from datetime import datetime
import os

import gradio as gr
import cv2
import ezexr
import numpy as np

import zerocomp.sdi_utils as sdi_utils
import glob

default_save_dir = 'results_shadowmap'
save_dir = default_save_dir
# dataset_root = 'E:/repos/shadowcomp_eval/rendered_w_shadowmap'
dataset_root = '.'


def upload(input_image, input_exr_path):

    if input_image is not None:
        pass

    elif input_exr_path is not None:
        global save_dir
        save_dir = os.path.dirname(input_exr_path)
        input_exr = ezexr.imread(input_exr_path, rgb='hybrid')
        bg = np.clip(input_exr['background'] ** (1 / 2.2), 0, 1)
        mask = input_exr['mask']
        fg = np.clip(input_exr['foreground'] ** (1 / 2.2), 0, 1)
        input_image = bg * (1 - mask) + fg * mask
        input_image = input_image[:, :, :3]

    return input_image, input_image


def save_shadowmap(input_obj_edit, input_obj_edit_negative):
    os.makedirs(save_dir, exist_ok=True)
    print('Saving to ' + save_dir)

    if save_dir != default_save_dir:
        name_list = glob.glob(os.path.join(save_dir, '*_shadowmap_neg_*.png'))
        name_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if len(name_list) > 0:
            # get the last number
            last_num = int(name_list[-1].split('_')[-1].split('.')[0])
            save_name = os.path.basename(save_dir + f'_shadowmap_{last_num + 1}')
            save_neg_name = os.path.basename(save_dir + f'_shadowmap_neg_{last_num + 1}')
        else:
            save_name = os.path.basename(save_dir + '_shadowmap_0')
            save_neg_name = os.path.basename(save_dir + '_shadowmap_neg_0')
    else:
        save_name = f'{datetime.now().strftime("%Y%m%d%H%M%S")}_shadowmap'
        save_neg_name = f'{datetime.now().strftime("%Y%m%d%H%M%S")}_shadowmap_neg'
    shadow_map = input_obj_edit['layers'][0]
    shadow_map = shadow_map[:, :, 3]
    shadow_map = cv2.imwrite(os.path.join(save_dir, save_name + '.png'), shadow_map)

    shadow_map_negative = input_obj_edit_negative['layers'][0]
    shadow_map_negative = shadow_map_negative[:, :, 3]
    shadow_map_negative = cv2.imwrite(os.path.join(save_dir, save_neg_name + '.png'), shadow_map_negative)


block = gr.Blocks().queue()
with block:
    with gr.Row():
        title = gr.Textbox(label="Input an png image or a multi-layer exr", placeholder="shadowmap")
    with gr.Row():
        input_image = gr.Image(label='Image', sources=['upload'], type="numpy")
        input_exr = gr.FileExplorer(label='EXR', file_count='single', root_dir=dataset_root)

    with gr.Row():
        upload_button = gr.Button(value="Upload")

    with gr.Row():
        input_obj_edit = gr.ImageEditor(label='Draw the positive shadow here', sources=['upload'], type="numpy")
        input_obj_edit_negative = gr.ImageEditor(label='Draw the negative shadow here', sources=['upload'], type="numpy")

    with gr.Row():
        save_shadowmap_btn = gr.Button(value="Save shadowmap")

    upload_button.click(upload, inputs=[input_image, input_exr], outputs=[input_obj_edit, input_obj_edit_negative])
    save_shadowmap_btn.click(save_shadowmap, inputs=[input_obj_edit, input_obj_edit_negative])

block.launch(server_name='127.0.0.1', server_port=786)
