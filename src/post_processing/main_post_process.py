import ezexr
import PIL.Image
import numpy as np
import os
from tqdm import tqdm
import subprocess
import cv2
import tempfile
from justpfm import justpfm
import argparse


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', help='Either "zerocomp" or "rgbx"', default='zerocomp')
    parser.add_argument('--dataset_dir', help='Directory of the base dataset', default='data/rendered_2024-10-20_19-07-58_control')
    parser.add_argument('--raw_outputs_dir', help='Directory containing raw outputs from SpotLight')
    parser.add_argument('--post_processed_outputs_dir', help='Directory which will contain the post-processed images', default='outputs_post_processed')
    args = parser.parse_args()

    crop_names = os.listdir(args.raw_outputs_dir)
    crop_names.remove('config.yaml')
    for k, crop_name in enumerate(tqdm(crop_names)):
        image_names = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', 'gt_dir']
        
        background = np.clip(ezexr.imread(os.path.join(args.dataset_dir, crop_name, f'{crop_name}_bundle0001.exr'), rgb='hybrid', whitelisted_channels=['background'])['background'][:, :, :3] ** (1/2.2), 0, 1)
        original_mask = (np.array(PIL.Image.open(os.path.join(args.raw_outputs_dir, crop_name, 'post_comp', f'mask.png'))) / 255)[:,:,0]
        mask = cv2.dilate(original_mask, np.ones((3,3), np.uint8), iterations=1)
        
        if args.backbone == 'zerocomp':
            composite_albedo = np.array(PIL.Image.open(os.path.join(args.raw_outputs_dir, crop_name, 'intrinsics', 'comp_diffuse.png'))) / 255
            composite_normals = np.array(PIL.Image.open(os.path.join(args.raw_outputs_dir, crop_name, 'intrinsics', 'comp_normal.png'))) / 255
        elif args.backbone == 'rgbx':
            composite_albedo = np.array(PIL.Image.open(os.path.join(args.raw_outputs_dir, crop_name, 'intrinsics', 'composited_albedo.png'))) / 255
            composite_normals = np.array(PIL.Image.open(os.path.join(args.raw_outputs_dir, crop_name, 'intrinsics', 'composited_normal.png'))) / 255
        else:
            raise Exception(f'Unknown backbone {args.backbone}')

        for image_name in image_names:
            def save_np(folder_name, image):
                png_path = os.path.join(args.post_processed_outputs_dir, folder_name, 'png', crop_name)
                jpg_path = os.path.join(args.post_processed_outputs_dir, folder_name, 'jpg', crop_name)
                os.makedirs(png_path, exist_ok=True)
                os.makedirs(jpg_path, exist_ok=True)

                pil_image = PIL.Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
                pil_image.save(os.path.join(png_path, f'{image_name}.png'), compress_level=1)  # 0 for no compression, 1 for fastest compression
                pil_image.save(os.path.join(jpg_path, f'{image_name}.jpg'), quality=90)

            image_path = os.path.join(args.raw_outputs_dir, crop_name, 'default', f'{image_name}.png')
            pred = PIL.Image.open(image_path)
            pred = np.array(pred) / 255
            save_np(f'{args.backbone}_raw', pred)

            if args.backbone == 'rgbx':
                pred_no_shadow = np.array(PIL.Image.open(os.path.join(args.raw_outputs_dir, crop_name, 'post_comp', f'{image_name}.png'))) / 255
            elif args.backbone == 'zerocomp':
                pred_no_shadow = np.array(PIL.Image.open(os.path.join(args.raw_outputs_dir, crop_name, 'post_comp', f'pred_no_shadow.png'))) / 255
            else:
                raise Exception(f'Unknown backbone {args.backbone}')
                

            shadow_opacity = np.clip((srgb_to_intensity(pred)+1e-6) / (srgb_to_intensity(pred_no_shadow) + 1e-6), 0, 1)
                            
            color_balance_factor = np.mean((1 - mask[:,:,None]) * background, axis=(0, 1), keepdims=True) / np.mean((1 - mask[:,:,None]) * pred_no_shadow, axis=(0, 1), keepdims=True)
            post_comp_color_balanced = color_balance_factor * pred * mask[:,:,None] + background * (1 - mask[:,:,None]) * shadow_opacity[:,:,None]
            save_np(f'{args.backbone}_post_color_balanced', post_comp_color_balanced)
            
            pred_no_shadow_denoised = denoise_image(pred_no_shadow, composite_albedo, composite_normals)
            pred_denoised = denoise_image(pred, composite_albedo, composite_normals)
            
            shadow_opacity_denoised = np.clip((srgb_to_intensity(pred_denoised)+1e-6) / (srgb_to_intensity(pred_no_shadow_denoised) + 1e-6), 0, 1)
            color_balance_factor = np.mean((1 - mask[:,:,None]) * background, axis=(0, 1), keepdims=True) / np.mean((1 - mask[:,:,None]) * pred_no_shadow_denoised, axis=(0, 1), keepdims=True)
            post_comp_color_balanced = color_balance_factor * pred_denoised * mask[:,:,None] + background * (1 - mask[:,:,None]) * shadow_opacity_denoised[:,:,None]
            save_np(f'{args.backbone}_post_color_balanced_denoised', post_comp_color_balanced)
