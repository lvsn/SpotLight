import ezexr
import PIL.Image
import numpy as np
import glob
import os
from tqdm import tqdm
import torch
from typing import Literal
from PIL import ImageDraw
import subprocess
from classes.blender_dto import BlenderCropDto
import cv2
import json
from kornia.color import rgb_to_lab, lab_to_rgb
import tempfile
from justpfm import justpfm
from typing import Tuple
from multiprocessing import Pool


IMAGE_FOLDERS = {
    'shadow_comp_v3_kornia_3x3_edge_2': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2025-02-23_23-18-15_control_dataset_kornia_3x3_edge_2',
    'rgbx_cfg_3_intensity_2_inpaint_iccv': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\raw_method_outputs\rendered_eevee_ambiant_input\rgbx_cfg_3_intensity_2_iccv',
}
INTRINSICS_FOLDER = r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-11-05_23-02-48_control_dataset_default\intrinsics'
SIMULATED_GT_FOLDER = r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\singleimg_rendered_crops\rendered_2024-10-20_19-07-58_control'

crop_names = os.listdir(IMAGE_FOLDERS['sg_light']) # TODO: update

CONTROL_OUTPUTS_PNG = os.path.join('outputs', 'post_processed', 'png')
CONTROL_OUTPUTS_JPG = os.path.join('outputs', 'post_processed', 'jpg')
MAX_LIGHT_DIRECTIONS = 8 # TODO: change this
os.makedirs(CONTROL_OUTPUTS_PNG, exist_ok=True)
os.makedirs(CONTROL_OUTPUTS_JPG, exist_ok=True)


def alpha_composite(layers, premultiplied=True):
    out = np.zeros(layers[0].shape)
    out[:, :, :3] = layers[0][:, :, :3] * layers[0][:, :, 3:]
    out[:, :, 3] = layers[0][:, :, 3]
    for i in range(1, len(layers)):
        if premultiplied:
            out[:, :, :3] = out[:, :, :3] + layers[i][:, :, :3] * layers[i][:, :, 3:] * (1 - out[:, :, 3:])
            out[:, :, 3] = out[:, :, 3] + layers[i][:, :, 3] * (1 - out[:, :, 3])
        else:
            out[:, :, :3] = out[:, :, :3] + layers[i][:, :, :3] * layers[i][:, :, 3:]
            out[:, :, 3] = out[:, :, 3] + layers[i][:, :, 3]
    out = np.clip(out, 0, 1)
    return out

def srgb_to_intensity(image):
    return np.dot(image, [0.299, 0.587, 0.114])

def denoise_image(noisy_image, albedo, normals):
    with tempfile.TemporaryDirectory() as tmpdirname:
        noisy_image_path = os.path.join(tmpdirname, 'noisy_image.pfm')
        albedo_path = os.path.join(tmpdirname, 'albedo.pfm')
        normals_path = os.path.join(tmpdirname, 'normals.pfm')
        output_path = os.path.join(tmpdirname, 'output.pfm')
        justpfm.write_pfm(file_name=noisy_image_path, data=noisy_image.astype(np.float32))
        justpfm.write_pfm(file_name=albedo_path, data=albedo.astype(np.float32))
        justpfm.write_pfm(file_name=normals_path, data=normals.astype(np.float32))
        subprocess.run([r'oidnDenoise', '--device', 'cpu', '--srgb', '--ldr', noisy_image_path, '--albedo', albedo_path, '--normal', normals_path, '-o', output_path, '-t', 'float', '-q', 'high'],
                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        denoised_image = justpfm.read_pfm(output_path)
    return denoised_image

if __name__ == '__main__':
    for k, crop_name in enumerate(tqdm(crop_names)):
        image_names = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', 'gt_dir']
        
        background = np.clip(ezexr.imread(os.path.join(SIMULATED_GT_FOLDER, crop_name, f'{crop_name}_bundle0001.exr'), rgb='hybrid', whitelisted_channels=['background'])['background'][:, :, :3] ** (1/2.2), 0, 1)
        original_mask = (np.array(PIL.Image.open(os.path.join(IMAGE_FOLDERS['shadow_comp_v2_default'], crop_name, 'post_comp', f'mask.png'))) / 255)[:,:,0]
        mask = cv2.dilate(original_mask, np.ones((3,3), np.uint8), iterations=1)

        composite_albedo = np.array(PIL.Image.open(os.path.join(INTRINSICS_FOLDER, f'{crop_name}_comp_diffuse.png'))) / 255
        composite_normals = np.array(PIL.Image.open(os.path.join(INTRINSICS_FOLDER, f'{crop_name}_comp_normal.png'))) / 255

        for image_name in image_names:

            for other_technique_name, technique_path in IMAGE_FOLDERS.items():
                
                def save_np(other_technique_name, image):
                    png_path = os.path.join(CONTROL_OUTPUTS_PNG, other_technique_name, crop_name)
                    jpg_path = os.path.join(CONTROL_OUTPUTS_JPG, other_technique_name, crop_name)
                    os.makedirs(png_path, exist_ok=True)
                    os.makedirs(jpg_path, exist_ok=True)

                    pil_image = PIL.Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
                    pil_image.save(os.path.join(png_path, f'{image_name}.png'), compress_level=1)  # 0 for no compression, 1 for fastest compression
                    pil_image.save(os.path.join(jpg_path, f'{image_name}.jpg'), quality=90)

                if other_technique_name.startswith('shadow_comp'):
                    image_path = os.path.join(technique_path, crop_name, 'default', f'{image_name}.png')
                    pred = PIL.Image.open(image_path)
                    pred = np.array(pred) / 255
                    save_np(other_technique_name, pred)

                    pred_no_shadow = np.array(PIL.Image.open(os.path.join(technique_path, crop_name, 'post_comp', f'pred_no_shadow.png'))) / 255
                    pred_no_shadow_denoised = denoise_image(pred_no_shadow, composite_albedo, composite_normals)

                    shadow_opacity = np.clip((srgb_to_intensity(pred)+1e-6) / (srgb_to_intensity(pred_no_shadow) + 1e-6), 0, 1)
                                    
                    color_balance_factor = np.mean((1 - mask[:,:,None]) * background, axis=(0, 1), keepdims=True) / np.mean((1 - mask[:,:,None]) * pred_no_shadow, axis=(0, 1), keepdims=True)
                    post_comp_color_balanced = color_balance_factor * pred * mask[:,:,None] + background * (1 - mask[:,:,None]) * shadow_opacity[:,:,None]
                    save_np(f'{other_technique_name}_post_color_balanced', post_comp_color_balanced)
                    
                    pred_denoised = denoise_image(pred, composite_albedo, composite_normals)
                    
                    shadow_opacity_denoised = np.clip((srgb_to_intensity(pred_denoised)+1e-6) / (srgb_to_intensity(pred_no_shadow_denoised) + 1e-6), 0, 1)
                    color_balance_factor = np.mean((1 - mask[:,:,None]) * background, axis=(0, 1), keepdims=True) / np.mean((1 - mask[:,:,None]) * pred_no_shadow_denoised, axis=(0, 1), keepdims=True)
                    post_comp_color_balanced = color_balance_factor * pred_denoised * mask[:,:,None] + background * (1 - mask[:,:,None]) * shadow_opacity_denoised[:,:,None]
                    save_np(f'{other_technique_name}_post_color_balanced_denoised', post_comp_color_balanced)

                    
                elif other_technique_name.startswith('rgbx'):
                    image_path = os.path.join(technique_path, f'{crop_name}', 'default', f'{image_name}.png')
                    pred = PIL.Image.open(image_path)
                    pred = np.array(pred) / 255
                    save_np(other_technique_name, pred)

                    pred_no_shadow = np.array(PIL.Image.open(os.path.join(technique_path, f'{crop_name}',  'post_comp', f'{image_name}.png'))) / 255
                    pred_no_shadow_denoised = denoise_image(pred_no_shadow, composite_albedo, composite_normals)
                    
                    shadow_opacity = np.clip((srgb_to_intensity(pred)+1e-6) / (srgb_to_intensity(pred_no_shadow) + 1e-6), 0, 1)
                    color_balance_factor = np.mean((1 - mask[:,:,None]) * background, axis=(0, 1), keepdims=True) / np.mean((1 - mask[:,:,None]) * pred_no_shadow, axis=(0, 1), keepdims=True)
                    post_comp_color_balanced = color_balance_factor * pred * mask[:,:,None] + background * (1 - mask[:,:,None]) * shadow_opacity[:,:,None]
                    save_np(f'{other_technique_name}_post_color_balanced', post_comp_color_balanced)

                    pred_denoised = denoise_image(pred, composite_albedo, composite_normals)
                    
                    shadow_opacity_denoised = np.clip((srgb_to_intensity(pred_denoised)+1e-6) / (srgb_to_intensity(pred_no_shadow_denoised) + 1e-6), 0, 1)
                    color_balance_factor = np.mean((1 - mask[:,:,None]) * background, axis=(0, 1), keepdims=True) / np.mean((1 - mask[:,:,None]) * pred_no_shadow_denoised, axis=(0, 1), keepdims=True)
                    post_comp_color_balanced = color_balance_factor * pred_denoised * mask[:,:,None] + background * (1 - mask[:,:,None]) * shadow_opacity_denoised[:,:,None]
                    save_np(f'{other_technique_name}_post_color_balanced_denoised', post_comp_color_balanced)

                    
                else:
                    raise NotImplementedError
