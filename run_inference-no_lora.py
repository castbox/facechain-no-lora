# Copyright (c) Alibaba, Inc. and its affiliates.

"""
Input: Only one image of the user
Output: 5 images of the user with the selected style

Process: Use stable-diffusion and face fusion to generate the images, do not use face lora or style lora.

"""


import os

from facechain.inference import GenPortraitNoLora
import cv2
from facechain.utils import snapshot_download
from facechain.constants import neg_prompt, pos_prompt_with_cloth, pos_prompt_with_style, styles, base_models

# set the cache path to the data disk, otherwise the models downloaded may be too large to handle
os.environ['MODELSCOPE_CACHE'] = "/root/autodl-tmp" 

def generate_pos_prompt(style_model, prompt_cloth):
    if style_model in base_models[0]['style_list'][:-1] or style_model is None:
        pos_prompt = pos_prompt_with_cloth.format(prompt_cloth)
    else:
        matched = list(filter(lambda style: style_model == style['name'], styles))
        if len(matched) == 0:
            raise ValueError(f'styles not found: {style_model}')
        matched = matched[0]
        pos_prompt = pos_prompt_with_style.format(matched['add_prompt_style'])
    return pos_prompt


use_main_model = True
use_face_swap = True
use_post_process = True
use_stylization = False
use_depth_control = False
use_pose_model = False
pose_image = 'poses/man/pose1.png'
processed_dir = './processed/white_women'
num_generate = 10

base_model = base_models[2]
base_model_id = base_model['model_id']
revision = 'v2.0'

base_model_sub_dir = base_model['sub_path']
train_output_dir = './output/white_women'
output_dir = './generated/white_women_step_30_2'
style = styles[1]
style_model_id = style['model_id']

if base_model_id == None:
    style_model_path = None
    pos_prompt = generate_pos_prompt(style['name'], style['add_prompt_style'])
else:
    if os.path.exists(base_model_id):
        model_dir = base_model_id
    else:
        model_dir = snapshot_download(base_model_id, revision=style['revision'])
    pos_prompt = generate_pos_prompt(style['name'], style['add_prompt_style'])  # style has its own prompt

if not use_pose_model:
    pose_model_path = None
    use_depth_control = False
    pose_image = None
else:
    model_dir = snapshot_download('damo/face_chain_control_model', revision='v1.0.1')
    pose_model_path = os.path.join(model_dir, 'model_controlnet/control_v11p_sd15_openpose')

# hacky way to overwrite the positive prompt generated before
pos_prompt = style['add_prompt_style']

print("------------------")
print(f"Debug Info: the pos prompt used is: {pos_prompt} ")
print(f"Debug Info: the neg prompt used is: {neg_prompt} ")
print(f"Debug Info: the model dir is: {model_dir} ")
print("------------------")

gen_portrait = GenPortraitNoLora(pose_model_path, pose_image, use_depth_control, pos_prompt, neg_prompt,
                           use_main_model,
                           use_face_swap, use_post_process,
                           use_stylization)

outputs = gen_portrait(processed_dir, num_generate, base_model_id,
                       train_output_dir, base_model_sub_dir, revision)

os.makedirs(output_dir, exist_ok=True)

for i, out_tmp in enumerate(outputs):
    cv2.imwrite(os.path.join(output_dir, f'{i}.png'), out_tmp)

