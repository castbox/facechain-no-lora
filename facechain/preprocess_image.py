# coding=utf-8
# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Bascially the "train_text_to_image_lora.py without the lora training part.

Preprocess the image, generate the labels, filter them and then store them in the metadata.jsonl file.

"""

import argparse
import base64
import itertools
import json
import logging
import math
import os
import random
import shutil
from glob import glob
from pathlib import Path

import cv2
import datasets
import diffusers
import numpy as np
import onnxruntime
import PIL.Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor
from typing import List, Optional, Tuple, Union
import torchvision.transforms.functional as Ft
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import (AutoencoderKL, DDPMScheduler, DiffusionPipeline,
                       DPMSolverMultistepScheduler,
                       StableDiffusionInpaintPipeline, UNet2DConditionModel)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo, upload_folder
from facechain.utils import snapshot_download

from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from torch import multiprocessing
from transformers import CLIPTextModel, CLIPTokenizer

from facechain.inference import data_process_fn

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.14.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# set the cache path to the data disk, otherwise the models downloaded may be too large to handle
os.environ['MODELSCOPE_CACHE'] = "/root/autodl-tmp" 


class FaceCrop(torch.nn.Module):

    @staticmethod
    def get_params(img: Tensor) -> Tuple[int, int, int, int]:
        _, h, w = Ft.get_dimensions(img)
        if h != w:
            raise ValueError(f"The input image is not square.")
        ratio = torch.rand(size=(1,)).item() * 0.1 + 0.35
        yc = torch.rand(size=(1,)).item() * 0.15 + 0.35

        th = int(h / 1.15 * 0.35 / ratio)
        tw = th

        cx = int(0.5 * w)
        cy = int(0.5 / 1.15 * h)

        i = min(max(int(cy - yc * th), 0), h - th)
        j = int(cx - 0.5 * tw)

        return i, j, th, tw

    def __init__(self):
        super().__init__()

    def forward(self, img):
        i, j, h, w = self.get_params(img)

        return Ft.crop(img, i, j, h, w)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


def save_model_card(repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def softmax(x):
    x -= np.max(x, axis=0, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)
    return x


def get_rot(image):
    model_dir = snapshot_download('Cherrytest/rot_bgr',
                                  revision='v1.0.0')
    model_path = os.path.join(model_dir, 'rot_bgr.onnx')
    ort_session = onnxruntime.InferenceSession(model_path)

    img_cv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    img_clone = img_cv.copy()
    img_np = cv2.resize(img_cv, (224, 224))
    img_np = img_np.astype(np.float32)
    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape((1, 1, 3))
    norm = np.array([0.01742919, 0.017507, 0.01712475], dtype=np.float32).reshape((1, 1, 3))
    img_np = (img_np - mean) * norm
    img_tensor = torch.from_numpy(img_np)
    img_tensor = img_tensor.unsqueeze(0)
    img_nchw = img_tensor.permute(0, 3, 1, 2)
    ort_inputs = {ort_session.get_inputs()[0].name: img_nchw.numpy()}
    outputs = ort_session.run(None, ort_inputs)
    logits = outputs[0].reshape((-1,))
    probs = softmax(logits)
    rot_idx = np.argmax(probs)
    if rot_idx == 1:
        print('rot 90')
        img_clone = cv2.transpose(img_clone)
        img_clone = np.flip(img_clone, 1)
        return Image.fromarray(cv2.cvtColor(img_clone, cv2.COLOR_BGR2RGB))
    elif rot_idx == 2:
        print('rot 180')
        img_clone = cv2.flip(img_clone, -1)
        return Image.fromarray(cv2.cvtColor(img_clone, cv2.COLOR_BGR2RGB))
    elif rot_idx == 3:
        print('rot 270')
        img_clone = cv2.transpose(img_clone)
        img_clone = np.flip(img_clone, 0)
        return Image.fromarray(cv2.cvtColor(img_clone, cv2.COLOR_BGR2RGB))
    else:
        return image


def prepare_dataset(instance_images: list, output_dataset_dir):
    if not os.path.exists(output_dataset_dir):
        os.makedirs(output_dataset_dir)
    for i, temp_path in enumerate(instance_images):
        image = PIL.Image.open(temp_path)
        # image = PIL.Image.open(temp_path.name)
        '''
        w, h = image.size
        max_size = max(w, h)
        ratio =  1024 / max_size
        new_w = round(w * ratio)
        new_h = round(h * ratio)
        '''
        image = image.convert('RGB')
        image = get_rot(image)
        # image = image.resize((new_w, new_h))
        # image = image.resize((new_w, new_h), PIL.Image.ANTIALIAS)
        out_path = f'{output_dataset_dir}/{i:03d}.jpg'
        image.save(out_path, format='JPEG', quality=100)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier.",
    )
    parser.add_argument(
        "--sub_path",
        type=str,
        default=None,
        required=False,
        help="The sub model path of the `pretrained_model_name_or_path`",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The data images dir"
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dataset_name",
        type=str,
        default=None,
        help=(
            "The dataset dir after processing"
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")

    # lora args
    parser.add_argument("--use_peft", action="store_true", help="Whether to use peft to support lora")
    parser.add_argument("--lora_r", type=int, default=4, help="Lora rank, only used if use_lora is True")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Lora alpha, only used if lora is True")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Lora dropout, only used if use_lora is True")
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        help="Bias type for Lora. Can be 'none', 'all' or 'lora_only', only used if use_lora is True",
    )
    parser.add_argument(
        "--lora_text_encoder_r",
        type=int,
        default=4,
        help="Lora rank for text encoder, only used if `use_lora` and `train_text_encoder` are True",
    )
    parser.add_argument(
        "--lora_text_encoder_alpha",
        type=int,
        default=32,
        help="Lora alpha for text encoder, only used if `use_lora` and `train_text_encoder` are True",
    )
    parser.add_argument(
        "--lora_text_encoder_dropout",
        type=float,
        default=0.0,
        help="Lora dropout for text encoder, only used if `use_lora` and `train_text_encoder` are True",
    )
    parser.add_argument(
        "--lora_text_encoder_bias",
        type=str,
        default="none",
        help="Bias type for Lora. Can be 'none', 'all' or 'lora_only', only used if use_lora and `train_text_encoder` are True",
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None and args.output_dataset_name is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def main():

    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir)

    if args.dataset_name is not None:
        # if dataset_name is None, then it's called from the gradio
        # the data processing will be executed in the app.py to save the gpu memory.
        print('All input images:', args.dataset_name)
        args.dataset_name = [os.path.join(args.dataset_name, x) for x in os.listdir(args.dataset_name)]
        shutil.rmtree(args.output_dataset_name, ignore_errors=True)
        prepare_dataset(args.dataset_name, args.output_dataset_name)
        ## Our data process fn
        data_process_fn(input_img_dir=args.output_dataset_name, use_data_process=True)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
