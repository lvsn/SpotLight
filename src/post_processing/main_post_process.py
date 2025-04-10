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
    ## 'shadow_comp_adjusted_rgb': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-10-30_23-40-22_control_dataset',
    ## 'shadow_comp': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-10-24_19-33-38_control_dataset',
    ## 'shadow_comp_zoedepth_cfg_1': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-10-31_14-26-47_control_dataset',
    ## 'shadow_comp_zoedepth_cfg_3': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-10-31_15-25-30_control_dataset_zoedepth_cfg_3',
    ## 'shadow_comp_combine_shading_mask_cfg_3': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-11-01_18-39-15_control_dataset_combine_shading_mask',
    # 'shadow_comp_denoised': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\22-oct-2024 (denoised shadowcomp outputs)\2024-10-14_21-09-25_control_dataset_denoised',
    'sg_light': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\singleimg_rendered_crops\rendered_2024-10-20_19-07-58_control_sg_light',
    # 'sg_light_no_obj': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\singleimg_rendered_crops\rendered_2024-10-20_19-07-58_control_sg_light_no_obj',
    # 'sg_light_edit_color': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\singleimg_rendered_crops\rendered_2024-09-23_09-41-45_control_sg_light_edit_color',
    # 'everlight': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\singleimg_rendered_crops\rendered_2024-10-20_19-07-58_control_everlight',
    # 'weber': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\singleimg_rendered_crops\rendered_2024-10-20_19-07-58_control_weber',
    #'dilightnet': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\raw_method_outputs\rendered_2024-10-20_19-07-58_control\dilightnet',
    ## 'single_sphere': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\singleimg_rendered_crops\rendered_2024-09-23_09-41-45_control_single_sphere',
    'neural_gaffer': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\raw_method_outputs\rendered_2024-10-20_19-07-58_control\output_fred_no_offset=PI_no_gaussian_erosion_3',
    
    # 'diffusionlight': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\singleimg_rendered_crops\rendered_2024-10-20_19-07-58_control_diffusionlight',
    #'rgbx_cfg_3_intensity_2': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\raw_method_outputs\rendered_2024-10-20_19-07-58_control\rgbx_cfg_3_intensity_2',
    'rgbx_cfg_3_intensity_2_inpaint': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\raw_method_outputs\rendered_2024-10-20_19-07-58_control\rgbx_cfg_3_intensity_2_inpaint',
    ##### 'rgbx_cfg_3_intensity_2_ok_noise': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\raw_method_outputs\rendered_2024-10-20_19-07-58_control\rgbx_cfg_3_intensity_2_ok_noise',
    #'ic_light_bg_guidance': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\raw_method_outputs\rendered_2024-10-20_19-07-58_control\output_IC_Light',
    # Models for the paper
    'shadow_comp_v2_default': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-11-05_23-02-48_control_dataset_default',
    'relight_shadow_comp_v2_default': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-11-05_23-02-48_control_dataset_default',
    'two_lights_shadow_comp_v2_default': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-11-05_23-02-48_control_dataset_default',
    # 'shadow_comp_v2_cfg_1': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-11-06_03-22-28_control_dataset_cfg_scale_1',
    #'shadow_comp_v2_cfg_5': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-11-06_21-41-16_control_dataset_cfg_scale_5',
    #'shadow_comp_v2_cfg_7': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-11-06_15-19-33_control_dataset_cfg_scale_7',
    # 'shadow_comp_v2_latent_weight_0.2': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-11-07_00-17-41_control_dataset_latent_mask_0.2',
    # 'shadow_comp_v2_latent_weight_0': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-11-06_11-39-11_control_dataset_latent_mask_0',
    'shadow_comp_v2_neg_no_shadow': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-11-08_05-59-20_control_dataset_neg_no_shadow',
    'shadow_comp_v2_neg_zerocomp': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-11-11_20-19-56_control_dataset_neg_zerocomp',
    'shadow_comp_v2_light_radius_0': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-11-14_06-50-20_control_dataset_light_source_radius_0',
    'shadow_comp_v2_light_radius_5': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-11-14_00-10-12_control_dataset_light_source_radius_5',
    #'shadow_comp_v2_zenith_70': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-11-21_00-07-38_control_dataset_force_zenith_70',
    'shadow_comp_v2_zenith_20': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-11-21_09-15-01_control_dataset_force_zenith_20',
    'zerocomp_coarse_shadow': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2025-01-23_23-44-24_control_dataset_zerocomp_coarse_shadow',
    'zerocomp_coarse_shadow_sd_edit_0.5': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2025-01-24_10-35-38_control_dataset_zerocomp_coarse_shadow_sd_edit_0.5',
    'zerocomp_coarse_shadow_sd_edit_0.8': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2025-01-27_15-23-43_control_dataset_zerocomp_coarse_shadow_sd_edit_0.8',
    'shadow_comp_v2_3x3_edge_x2': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2025-02-16_15-33-48_control_dataset_thin_shadows_3x3_edge_x2_erodeobj_batch0',
    # 'zerocomp': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-11-14_11-14-58_control_dataset_zerocomp'
    'zerocomp': r'D:\Downloads\results (1)\2024-11-14_20-37-16_control_dataset',
    'neural_gaffer_erode3_ambiant_eevee': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\raw_method_outputs\rendered_eevee_ambiant_input\neural_gaffer_erosion_3',
    'neural_gaffer_erode5_ambiant_eevee': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\raw_method_outputs\rendered_eevee_ambiant_input\neural_gaffer_erosion_5',
    'dilightnet_ambiant_eevee': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\raw_method_outputs\rendered_eevee_ambiant_input\dilightnet',
    'ic_light_bg_guidance_ambiant_eevee': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\raw_method_outputs\rendered_eevee_ambiant_input\output_IC_Light',
    'shadow_comp_v3_kornia_3x3_edge_2': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2025-02-23_23-18-15_control_dataset_kornia_3x3_edge_2',
    'white_mask_sentinel': '',
    'sentinel_no_control': '',
    'shading_only': '',
    'shadow_only': '',
    'shadow_comp_v3_cfg_1': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2025-02-25_00-20-43_control_dataset_cfg_scale_1',
    'shadow_comp_v3_cfg_7': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2025-02-24_18-12-25_control_dataset_cfg_scale_7',
     'shadow_comp_v3_latent_weight_0.2': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2025-02-25_03-33-28_control_dataset_latent_mask_0.2',
    'shadow_comp_v3_latent_weight_0': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2025-02-24_21-27-54_control_dataset_latent_mask_0',
     'shadow_comp_v3_light_radius_0': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2025-02-25_03-33-39_control_dataset_light_source_radius_0',
    'shadow_comp_v3_light_radius_5': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2025-02-24_21-28-05_control_dataset_light_source_radius_5',
    'rgbx_cfg_3_intensity_2_inpaint_iccv': r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\raw_method_outputs\rendered_eevee_ambiant_input\rgbx_cfg_3_intensity_2_iccv',
    'shadow_comp_v3_cfg_zero_star_mask_on': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2025-04-08_11-36-04_control_dataset_cfg_zero_star_mask_on',
    'shadow_comp_v3_cfg_zero_star_mask_off': r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2025-04-08_11-36-04_control_dataset_cfg_zero_star_mask_off'
}
INTRINSICS_FOLDER = r'D:\Desktop\Google Drive\uLaval\Maitrise\Results\24-sept-2024 (outputs control dataset)\2024-11-05_23-02-48_control_dataset_default\intrinsics'
SIMULATED_GT_FOLDER = r'D:\Documents\Projets\maitrise-code\compositing-blender\outputs\singleimg_rendered_crops\rendered_2024-10-20_19-07-58_control'

crop_names = os.listdir(IMAGE_FOLDERS['sg_light'])

CONTROL_OUTPUTS_PNG = os.path.join('outputs', 'c_o', 'png')
CONTROL_OUTPUTS_JPG = os.path.join('outputs', 'c_o', 'jpg')
CONTROL_OUTPUTS_MP4 = os.path.join('outputs', 'c_o', 'mp4')
MAX_LIGHT_DIRECTIONS = 8 # TODO: change this
os.makedirs(CONTROL_OUTPUTS_PNG, exist_ok=True)
os.makedirs(CONTROL_OUTPUTS_JPG, exist_ok=True)
os.makedirs(CONTROL_OUTPUTS_MP4, exist_ok=True)

# Create a folder for concatenated images/videos
CONTROL_OUTPUTS_RANDOM_CONCAT = os.path.join('outputs', 'c_o', 'png_random_concat')
os.makedirs(CONTROL_OUTPUTS_RANDOM_CONCAT, exist_ok=True)
CONTROL_OUTPUTS_MP4_CONCAT = os.path.join('outputs', 'c_o', 'mp4_concat')
os.makedirs(CONTROL_OUTPUTS_MP4_CONCAT, exist_ok=True)

FRAMES_PER_SECOND = 2

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

def draw_light_direction_on_image(image, light_xyz, *, stick_position=None, coordinate_system: Literal['blender', 'skylibs'] = 'skylibs'):
    assert len(light_xyz) == 3
    # assert norm of 1
    if type(light_xyz) == torch.Tensor:
        light_xyz = light_xyz.cpu().numpy()
    light_xyz = light_xyz / np.linalg.norm(light_xyz)

    if image is None:
        image = np.ones((512, 512, 3))
    assert len(image.shape) == 3
    assert image.shape[2] == 3

    image_shape = image.shape[:2]
    image = PIL.Image.fromarray((image * 255).astype(np.uint8))
  
    # draw stick
    if stick_position is not None:
        stick_end = np.array([stick_position[1], stick_position[0]])
    else:
        stick_end = np.array([image_shape[1] * 0.5, image_shape[0] * 0.75])

    # stick_start = np.array([image_shape[1] * 0.5, image_shape[0] * 0.5])
    stick_start = stick_end - np.array([0, 0.25 * image_shape[0]])
    stick_length = np.linalg.norm(stick_end - stick_start)
    
    # draw stick's shadow
    shadow_start = stick_end
    # NOTE: not perfect calculation of shadow length (doesn't account for light zenith)
    y_perspective_factor = 0.5
    if coordinate_system == 'blender':
        shadow_length = stick_length / light_xyz[2] * (1 - light_xyz[2] ** 2) ** 0.5
        shadow_end = shadow_start - np.array([light_xyz[0], -light_xyz[1] * y_perspective_factor]) * shadow_length
    elif coordinate_system == 'skylibs':
        shadow_length = stick_length / light_xyz[1] * (1 - light_xyz[1] ** 2) ** 0.5
        shadow_end = shadow_start - np.array([light_xyz[0], light_xyz[2] * y_perspective_factor]) * shadow_length
    # ImageDraw.Draw(image).line([tuple(shadow_start), tuple(shadow_end)], fill=(128, 128, 128), width=7, joint='curve')
    ImageDraw.Draw(image).line([tuple(shadow_start), tuple(shadow_end)], fill=(100, 100, 100), width=5, joint='curve')
    ImageDraw.Draw(image).line([tuple(stick_start), tuple(stick_end)], fill=(255, 0, 0), width=5, joint='curve')

    return np.array(image) / 255

def srgb_to_intensity(image):
    return np.dot(image, [0.299, 0.587, 0.114])

def match_color_lab(reference: np.ndarray, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    # inspired by yohan, modified by chatgpt

    # Convert images to torch tensors and reshape from HWC to CHW
    reference_tensor = torch.from_numpy(reference.transpose(2, 0, 1)).unsqueeze(0).float()
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    
    # Convert to LAB color space
    reference_lab = rgb_to_lab(reference_tensor)
    image_lab = rgb_to_lab(image_tensor)
    
    if mask is not None:
        # Convert mask to tensor and ensure it has the correct shape
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        # Compute masked mean and std
        ref_mean, ref_std = _masked_mean_std(reference_lab, mask_tensor)
        img_mean, img_std = _masked_mean_std(image_lab, mask_tensor)
    else:
        # Compute global mean and std
        ref_mean = reference_lab.mean(dim=(2, 3), keepdim=True)
        ref_std = reference_lab.std(dim=(2, 3), keepdim=True)
        img_mean = image_lab.mean(dim=(2, 3), keepdim=True)
        img_std = image_lab.std(dim=(2, 3), keepdim=True)
    
    # Avoid division by zero
    img_std = torch.where(img_std == 0, torch.tensor(1e-8), img_std)
    ref_std = torch.where(ref_std == 0, torch.tensor(1e-8), ref_std)
    
    # Perform color matching
    matched_lab = (image_lab - img_mean) / img_std * ref_std + ref_mean
    
    # Convert back to RGB
    matched_rgb = lab_to_rgb(matched_lab)
    
    # Remove batch dimension and rearrange axes back to HWC
    result_image = matched_rgb[0].permute(1, 2, 0).numpy()
    
    # Clip values to [0,1]
    result_image = np.clip(result_image, 0.0, 1.0)
    
    return result_image

def _masked_mean_std(tensor: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute masked mean and std
    masked_tensor = tensor * mask
    masked_sum = masked_tensor.sum(dim=(2, 3))  # Shape: [B, C]
    masked_count = mask.sum(dim=(2, 3))  # Shape: [B, 1]
    # Avoid division by zero
    masked_count = torch.where(masked_count == 0, torch.tensor(1e-8), masked_count)
    mean = (masked_sum / masked_count).unsqueeze(-1).unsqueeze(-1)
    var = ((tensor - mean) ** 2 * mask).sum(dim=(2, 3)) / masked_count
    std = torch.sqrt(var).unsqueeze(-1).unsqueeze(-1)
    return mean, std

def denoise_image(noisy_image, albedo, normals):
    with tempfile.TemporaryDirectory() as tmpdirname:
        noisy_image_path = os.path.join(tmpdirname, 'noisy_image.pfm')
        albedo_path = os.path.join(tmpdirname, 'albedo.pfm')
        normals_path = os.path.join(tmpdirname, 'normals.pfm')
        output_path = os.path.join(tmpdirname, 'output.pfm')
        justpfm.write_pfm(file_name=noisy_image_path, data=noisy_image.astype(np.float32))
        justpfm.write_pfm(file_name=albedo_path, data=albedo.astype(np.float32))
        justpfm.write_pfm(file_name=normals_path, data=normals.astype(np.float32))
        subprocess.run([r'D:\Downloads\oidn-2.3.0.x64.windows\oidn-2.3.0.x64.windows\bin\oidnDenoise.exe', '--device', 'cpu', '--srgb', '--ldr', noisy_image_path, '--albedo', albedo_path, '--normal', normals_path, '-o', output_path, '-t', 'float', '-q', 'high'],
                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        denoised_image = justpfm.read_pfm(output_path)
    return denoised_image

if __name__ == '__main__':
    if True:
        # crop_names = [
        #     "AG8A8563-e652dd5751_02_crop_B07B78RCT5",
        #     "AG8A8458-c640851baa_05_crop_B075HWR58K",
        #     "AG8A6551-16ae8050c9_06_crop_B075X4QFK2",
        #     "AG8A4702-1a77809d2c_01_crop_B07HZ6ZCZW",
        #     "AG8A4106-248c9b6695_00_crop_B075X33SC6",
        #     "AG8A4106-248c9b6695_00_crop_B07HZ1LZS1",
        #     "9C4A6765-55315335bf_07_crop_B07B4MPXKJ",
        #     "9C4A5673-d4ecde512a_05_crop_B075HR7DM7",
        #     "9C4A0395-ef28332472_02_crop_B07HSKBHBT",
        #     "9C4A0370-2d21311d85_02_crop_B075QDTYGK",
        # ]
        #crop_names = ["AG8A6551-16ae8050c9_06_crop_B075X4QFK2", "9C4A8239-7a9d7093e2_04_crop_B082VLJ7Y2", "9C4A0566-a088c98ccf_05_crop_B07VKM698C", "9C4A3419-2e5c9b4d85_08_crop_B07B4D8B2S", "9C4A4878-4dac321d8b_02_crop_B07HP93VDJ", "9C4A4861-19626f89e9_04_crop_B07B4MSYPS", "AG8A0630-e5622e17d2_01_crop_B07B85FJD5", "AG8A6719-fc8e1ea686_05_crop_B07DBJ1H18", "AG8A3196-8e1dfd8d95_08_crop_B07WMQ8P8Q", "AG8A8843-b91f89fff2_05_crop_B07B4YXNR3"]
        for k, crop_name in enumerate(tqdm(crop_names[15:])):
         

            image_names = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', 'gt_dir']
            # image_names = ['zerocomp']
            # image_names = ['gt_dir']
            if True:
                background = np.clip(ezexr.imread(os.path.join(SIMULATED_GT_FOLDER, crop_name, f'{crop_name}_bundle0001.exr'), rgb='hybrid', whitelisted_channels=['background'])['background'][:, :, :3] ** (1/2.2), 0, 1)
                original_mask = (np.array(PIL.Image.open(os.path.join(IMAGE_FOLDERS['shadow_comp_v2_default'], crop_name, 'post_comp', f'mask.png'))) / 255)[:,:,0]
                # mask = (np.array(PIL.Image.open(os.path.join(IMAGE_FOLDERS['zerocomp'], crop_name, 'post_comp', f'mask.png'))) / 255)[:,:,0]
                # dilate
                mask = cv2.dilate(original_mask, np.ones((3,3), np.uint8), iterations=1)

                composite_albedo = np.array(PIL.Image.open(os.path.join(INTRINSICS_FOLDER, f'{crop_name}_comp_diffuse.png'))) / 255
                composite_normals = np.array(PIL.Image.open(os.path.join(INTRINSICS_FOLDER, f'{crop_name}_comp_normal.png'))) / 255


            for image_name in image_names:
                out = [None, None, None]

                # blender_dto_path = os.path.join(IMAGE_FOLDERS['sg_light'], crop_name, f'{crop_name}_blender_dto_{image_name}.json')
                # blender_dto = BlenderCropDto.schema().loads(open(blender_dto_path, 'r').read())
                # shadow_image = draw_light_direction_on_image(None, np.array(blender_dto.lighting.single_sphere_position_object_space), coordinate_system='blender')
    # 
                # os.makedirs(os.path.join(CONTROL_OUTPUTS_PNG, 'shadow', crop_name), exist_ok=True)
                # PIL.Image.fromarray((shadow_image * 255).astype(np.uint8)).save(os.path.join(CONTROL_OUTPUTS_PNG, 'shadow', crop_name, f'{image_idx:04d}.png'))
                # os.makedirs(os.path.join(CONTROL_OUTPUTS_JPG, 'shadow', crop_name), exist_ok=True)
                # PIL.Image.fromarray((shadow_image * 255).astype(np.uint8)).save(os.path.join(CONTROL_OUTPUTS_JPG, 'shadow', crop_name, f'{image_idx:04d}.jpg'), quality=90)


                for other_technique_name, technique_path in IMAGE_FOLDERS.items():
                   #if not (other_technique_name.startswith('ic_light') or other_technique_name.startswith('dilightnet') or other_technique_name.startswith('neural_gaffer')):
                   #    continue
                    if not other_technique_name in ['shadow_comp_v3_cfg_zero_star_mask_on', 'shadow_comp_v3_cfg_zero_star_mask_off']:
                        continue
                    # print(f'Processing {other_technique_name} for {crop_name} {image_name}')
                    def save_np(other_technique_name, image):
                        png_path = os.path.join(CONTROL_OUTPUTS_PNG, other_technique_name, crop_name)
                        jpg_path = os.path.join(CONTROL_OUTPUTS_JPG, other_technique_name, crop_name)
                        os.makedirs(png_path, exist_ok=True)
                        os.makedirs(jpg_path, exist_ok=True)

                        pil_image = PIL.Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
                        pil_image.save(os.path.join(png_path, f'{image_name}.png'), compress_level=1)  # 0 for no compression, 1 for fastest compression
                        pil_image.save(os.path.join(jpg_path, f'{image_name}.jpg'), quality=90)

                    if other_technique_name in ['sg_light', 'sg_light_no_obj', 'sg_light_edit_color', 'everlight', 'weber', 'single_sphere', 'diffusionlight']:
                        image_path = os.path.join(technique_path, crop_name, f'{crop_name}_{image_name}_bundle0001.exr')
                        pred = np.clip(ezexr.imread(image_path, rgb='hybrid', whitelisted_channels=['composite'])['composite'][:, :, :3] ** (1/2.2), 0, 1)
                        save_np(other_technique_name, pred)
                    elif other_technique_name in ['ic_light_bg_guidance', 'dilightnet', 'neural_gaffer',
                                                  'neural_gaffer_erode5_ambiant_eevee', 'dilightnet_ambiant_eevee', 'ic_light_bg_guidance_ambiant_eevee']:
                        # that's legit just copy pasting
                        pred = np.array(PIL.Image.open(os.path.join(technique_path, crop_name, f'{image_name}.png'))) / 255
                        save_np(other_technique_name, pred)

                        pred_denoised = denoise_image(pred, composite_albedo, composite_normals)
                        pred_denoised = pred_denoised * mask[:,:,None] + pred * (1 - mask[:,:,None])
                        save_np(f'{other_technique_name}_denoised', pred_denoised)
                    elif other_technique_name in ['white_mask_sentinel']:
                        # that's legit just copy pasting
                        pred = np.array(PIL.Image.open(os.path.join(IMAGE_FOLDERS['ic_light_bg_guidance_ambiant_eevee'], crop_name, f'{image_name}.png'))) / 255
                        pred = pred * (1 - mask[:,:,None]) + mask[:,:,None]
                        save_np(other_technique_name, pred)
                    elif other_technique_name.startswith('shadow_comp'):
                        image_path = os.path.join(technique_path, crop_name, 'default', f'{image_name}.png')
                        pred = PIL.Image.open(image_path)
                        pred = np.array(pred) / 255
                        save_np(other_technique_name, pred)

                        pred_no_shadow = np.array(PIL.Image.open(os.path.join(technique_path, crop_name, 'post_comp', f'pred_no_shadow.png'))) / 255
                        pred_no_shadow_denoised = denoise_image(pred_no_shadow, composite_albedo, composite_normals)

                        # mask = (np.array(PIL.Image.open(os.path.join(technique_path, crop_name, 'post_comp', f'mask.png'))) / 255)[:,:,0]
                       # positive_shading_mask = np.array(PIL.Image.open(os.path.join(technique_path, crop_name, 'intermediate', f'positive_shading_image_{image_name}.png'))) / 255
                        
                        shadow_opacity = np.clip((srgb_to_intensity(pred)+1e-6) / (srgb_to_intensity(pred_no_shadow) + 1e-6), 0, 1)
                        if False:                            
                            post_comp = pred * (mask[:,:,None]) + background * (1 - (mask[:,:,None])) * shadow_opacity[:,:,None]
                            save_np(f'{other_technique_name}_post', post_comp)
                        
                        color_balance_factor = np.mean((1 - mask[:,:,None]) * background, axis=(0, 1), keepdims=True) / np.mean((1 - mask[:,:,None]) * pred_no_shadow, axis=(0, 1), keepdims=True)
                        post_comp_color_balanced = color_balance_factor * pred * mask[:,:,None] + background * (1 - mask[:,:,None]) * shadow_opacity[:,:,None]
                        save_np(f'{other_technique_name}_post_color_balanced', post_comp_color_balanced)
                        
                        pred_denoised = denoise_image(pred, composite_albedo, composite_normals)
                        
                        shadow_opacity_denoised = np.clip((srgb_to_intensity(pred_denoised)+1e-6) / (srgb_to_intensity(pred_no_shadow_denoised) + 1e-6), 0, 1)
                        color_balance_factor = np.mean((1 - mask[:,:,None]) * background, axis=(0, 1), keepdims=True) / np.mean((1 - mask[:,:,None]) * pred_no_shadow_denoised, axis=(0, 1), keepdims=True)
                        post_comp_color_balanced = color_balance_factor * pred_denoised * mask[:,:,None] + background * (1 - mask[:,:,None]) * shadow_opacity_denoised[:,:,None]
                        save_np(f'{other_technique_name}_post_color_balanced_denoised', post_comp_color_balanced)

                        
                        if False:
                            color_balanced_lab = match_color_lab(background, pred, mask)
                            save_np(f'{other_technique_name}_balanced_lab', color_balanced_lab)

                            post_comp_color_balanced_lab = color_balanced_lab * mask[:,:,None] + background * (1 - mask[:,:,None]) * shadow_opacity[:,:,None]
                            save_np(f'{other_technique_name}_post_color_balanced_lab', post_comp_color_balanced_lab)
                                
                            # # TODO: blur mask
                            # post_comp_color_balanced_masked = post_comp_color_balanced * (1 - positive_shading_mask) + background * positive_shading_mask
                            # save_np(f'{other_technique_name}_post_color_balanced_masked', post_comp_color_balanced_masked)
                    elif other_technique_name.startswith('two_lights_shadow_comp_v2_default'):
                        image_path = os.path.join(technique_path, crop_name, 'default', f'{image_name}.png')
                        pred = PIL.Image.open(image_path)
                        pred = np.array(pred) / 255
                        save_np(other_technique_name, pred)

                        pred_denoised = denoise_image(pred, composite_albedo, composite_normals)

                        pred_no_shadow = np.array(PIL.Image.open(os.path.join(technique_path, crop_name, 'post_comp', f'pred_no_shadow.png'))) / 255
                        pred_no_shadow_denoised = denoise_image(pred_no_shadow, composite_albedo, composite_normals)

                        shadow_opacity_denoised = np.clip((srgb_to_intensity(pred_denoised)+1e-6) / (srgb_to_intensity(pred_no_shadow_denoised) + 1e-6), 0, 1)
                        color_balance_factor = np.mean((1 - mask[:,:,None]) * background, axis=(0, 1), keepdims=True) / np.mean((1 - mask[:,:,None]) * pred_no_shadow_denoised, axis=(0, 1), keepdims=True)
                        post_comp_color_balanced = color_balance_factor * pred_denoised * mask[:,:,None] + background * (1 - mask[:,:,None]) * shadow_opacity_denoised[:,:,None]
                        save_np(f'{other_technique_name}_post_color_balanced_denoised', post_comp_color_balanced)

                        pred_static = np.array(PIL.Image.open(os.path.join(technique_path, crop_name, 'default', f'0004.png'))) / 255
                        pred_static_denoised = denoise_image(pred_static, composite_albedo, composite_normals)

                        shadow_opacity_denoised_static = np.clip((srgb_to_intensity(pred_static_denoised)+1e-6) / (srgb_to_intensity(pred_no_shadow) + 1e-6), 0, 1)
                        post_comp_color_balanced_static = color_balance_factor * pred_static * mask[:,:,None] + background * (1 - mask[:,:,None]) * shadow_opacity_denoised_static[:,:,None]
                        save_np(f'{other_technique_name}_post_color_balanced_static_denoised', post_comp_color_balanced_static)
                        # layer_0 = np.concatenate([np.zeros((512, 512, 3)), 1 - shadow_opacity_denoised[:,:,None]], axis=2)
                        # layer_1 = np.concatenate([np.zeros((512, 512, 3)), 1 - shadow_opacity_denoised_static[:,:,None]], axis=2)
                        # layer_2 = np.concatenate([background, np.ones((512, 512, 1))], axis=2)
                        # bg_composite = alpha_composite([layer_0, layer_1, layer_2], premultiplied=True)[:,:,:3]

                        # full_composite = original_mask[:,:,None] * (0.5 * post_comp_color_balanced_static + 0.5 * post_comp_color_balanced) + (1 - original_mask[:,:,None]) * bg_composite
                        gamma = 2.2
                        full_composite = (0.5 * (post_comp_color_balanced_static ** gamma) + 0.5 * (post_comp_color_balanced ** gamma)) ** (1/gamma)
                        save_np(f'{other_technique_name}_full_composite', full_composite)

                    elif other_technique_name.startswith('relight_shadow_comp'):
                        image_path = os.path.join(technique_path, crop_name, 'relight_bg', f'inv_{image_name}.png')
                        pred = PIL.Image.open(image_path)
                        pred = np.array(pred) / 255
                        save_np(f'{other_technique_name}_default', pred)

                        # positive_shading_mask_path = os.path.join(technique_path, crop_name, 'intermediate', f'positive_shading_mask_{image_name}.png')
                        # positive_shading_mask = 1 - np.array(PIL.Image.open(positive_shading_mask_path)) / 255
                        positive_shadow = os.path.join(technique_path, crop_name, 'intermediate', f'shadow_positive_{image_name}.png')
                        positive_shading_mask = np.array(PIL.Image.open(positive_shadow))[:,:,0] / 255

                        composite_mask = 1 - (1-mask) * (1-positive_shading_mask)

                        avg_image = []
                        for avg_image_name in ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007']:
                            curr_image_path = os.path.join(technique_path, crop_name, 'relight_bg', f'inv_{avg_image_name}.png')
                            curr_pred = PIL.Image.open(curr_image_path)
                            curr_pred = np.array(curr_pred) / 255
                            avg_image.append(curr_pred)
                        avg_image = np.mean(avg_image, axis=0)
                        save_np(f'{other_technique_name}_avg', avg_image)

                        color_balance_factor = np.mean((1 - mask[:,:,None]) * background, axis=(0, 1), keepdims=True) / np.mean((1 - mask[:,:,None]) * avg_image, axis=(0, 1), keepdims=True)

                        composed_image = (1 - composite_mask[:,:,None]) * (pred / avg_image) * background + composite_mask[:,:,None] * pred * color_balance_factor
                        save_np(f'{other_technique_name}_composed', composed_image)


                        # pred_no_shadow = np.array(PIL.Image.open(os.path.join(technique_path, crop_name, 'post_comp', f'pred_no_shadow.png'))) / 255
                        # pred_no_shadow_denoised = denoise_image(pred_no_shadow, composite_albedo, composite_normals)
# 
                        # # mask = (np.array(PIL.Image.open(os.path.join(technique_path, crop_name, 'post_comp', f'mask.png'))) / 255)[:,:,0]
                       ##  positive_shading_mask = np.array(PIL.Image.open(os.path.join(technique_path, crop_name, 'intermediate', f'positive_shading_image_{image_name}.png'))) / 255
                        # 
                        # shadow_opacity = np.clip((srgb_to_intensity(pred)+1e-6) / (srgb_to_intensity(pred_no_shadow) + 1e-6), 0, 1)
                        # if False:                            
                        #     post_comp = pred * (mask[:,:,None]) + background * (1 - (mask[:,:,None])) * shadow_opacity[:,:,None]
                        #     save_np(f'{other_technique_name}_post', post_comp)
                        # 
                        # color_balance_factor = np.mean((1 - mask[:,:,None]) * background, axis=(0, 1), keepdims=True) / np.mean((1 - mask[:,:,None]) * pred_no_shadow, axis=(0, 1), keepdims=True)
                        # post_comp_color_balanced = color_balance_factor * pred * mask[:,:,None] + background * (1 - mask[:,:,None]) * shadow_opacity[:,:,None]
                        # save_np(f'{other_technique_name}_post_color_balanced', post_comp_color_balanced)
                        # 
                        # pred_denoised = denoise_image(pred, composite_albedo, composite_normals)
                        # 
                        # shadow_opacity_denoised = np.clip((srgb_to_intensity(pred_denoised)+1e-6) / (srgb_to_intensity(pred_no_shadow_denoised) + 1e-6), 0, 1)
                        # color_balance_factor = np.mean((1 - mask[:,:,None]) * background, axis=(0, 1), keepdims=True) / np.mean((1 - mask[:,:,None]) * pred_no_shadow_denoised, axis=(0, 1), keepdims=True)
                        # post_comp_color_balanced = color_balance_factor * pred_denoised * mask[:,:,None] + background * (1 - mask[:,:,None]) * shadow_opacity_denoised[:,:,None]
                        # save_np(f'{other_technique_name}_post_color_balanced_denoised', post_comp_color_balanced)

                        
                    elif other_technique_name.startswith('zerocomp'):
                        image_path = os.path.join(technique_path, crop_name, 'default', f'{image_name}.png')
                        pred = PIL.Image.open(image_path)
                        pred = np.array(pred) / 255
                        save_np(other_technique_name, pred)

                        pred_no_shadow = np.array(PIL.Image.open(os.path.join(technique_path, crop_name, 'post_comp', f'pred_no_shadow.png'))) / 255
                        pred_no_shadow_denoised = denoise_image(pred_no_shadow, composite_albedo, composite_normals)

                        # mask = (np.array(PIL.Image.open(os.path.join(technique_path, crop_name, 'post_comp', f'mask.png'))) / 255)[:,:,0]
                       # positive_shading_mask = np.array(PIL.Image.open(os.path.join(technique_path, crop_name, 'intermediate', f'positive_shading_image_{image_name}.png'))) / 255
                        
                        shadow_opacity = np.clip((srgb_to_intensity(pred)+1e-6) / (srgb_to_intensity(pred_no_shadow) + 1e-6), 0, 1)
                        if False:                            
                            post_comp = pred * (mask[:,:,None]) + background * (1 - (mask[:,:,None])) * shadow_opacity[:,:,None]
                            save_np(f'{other_technique_name}_post', post_comp)
                        
                        color_balance_factor = np.mean((1 - mask[:,:,None]) * background, axis=(0, 1), keepdims=True) / np.mean((1 - mask[:,:,None]) * pred_no_shadow, axis=(0, 1), keepdims=True)
                        post_comp_color_balanced = color_balance_factor * pred * mask[:,:,None] + background * (1 - mask[:,:,None]) * shadow_opacity[:,:,None]
                        save_np(f'{other_technique_name}_post_color_balanced', post_comp_color_balanced)
                        
                        pred_denoised = denoise_image(pred, composite_albedo, composite_normals)
                        
                        shadow_opacity_denoised = np.clip((srgb_to_intensity(pred_denoised)+1e-6) / (srgb_to_intensity(pred_no_shadow_denoised) + 1e-6), 0, 1)
                        color_balance_factor = np.mean((1 - mask[:,:,None]) * background, axis=(0, 1), keepdims=True) / np.mean((1 - mask[:,:,None]) * pred_no_shadow_denoised, axis=(0, 1), keepdims=True)
                        post_comp_color_balanced = color_balance_factor * pred_denoised * mask[:,:,None] + background * (1 - mask[:,:,None]) * shadow_opacity_denoised[:,:,None]
                        save_np(f'{other_technique_name}_post_color_balanced_denoised', post_comp_color_balanced)

                        
                        if False:
                            color_balanced_lab = match_color_lab(background, pred, mask)
                            save_np(f'{other_technique_name}_balanced_lab', color_balanced_lab)

                            post_comp_color_balanced_lab = color_balanced_lab * mask[:,:,None] + background * (1 - mask[:,:,None]) * shadow_opacity[:,:,None]
                            save_np(f'{other_technique_name}_post_color_balanced_lab', post_comp_color_balanced_lab)
                                
                            # # TODO: blur mask
                            # post_comp_color_balanced_masked = post_comp_color_balanced * (1 - positive_shading_mask) + background * positive_shading_mask
                            # save_np(f'{other_technique_name}_post_color_balanced_masked', post_comp_color_balanced_masked)
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

                        # # TODO: blur mask
                        # positive_shading_mask = np.array(PIL.Image.open(os.path.join(IMAGE_FOLDERS['shadow_comp_v2_default'], crop_name, 'intermediate', f'positive_shading_image_{image_name}.png'))) / 255
                        # post_comp_color_balanced_masked = post_comp * (1 - positive_shading_mask) + background * positive_shading_mask
                        # save_np(f'{other_technique_name}_post_masked', post_comp_color_balanced_masked)
                        
                    else:
                        raise NotImplementedError

            # simulated GT image
            if False:
                image_path = os.path.join(SIMULATED_GT_FOLDER, crop_name, f'{crop_name}_bundle0001.exr')
                pred = np.clip(ezexr.imread(image_path, rgb='hybrid', whitelisted_channels=['composite'])['composite'][:, :, :3] ** (1/2.2), 0, 1)
                os.makedirs(os.path.join(CONTROL_OUTPUTS_PNG, 'simulated_gt', crop_name), exist_ok=True)
                PIL.Image.fromarray((pred * 255).astype(np.uint8)).save(os.path.join(CONTROL_OUTPUTS_PNG, 'simulated_gt', crop_name, f'gt_dir.png'))
                os.makedirs(os.path.join(CONTROL_OUTPUTS_JPG, 'simulated_gt', crop_name), exist_ok=True)
                PIL.Image.fromarray((pred * 255).astype(np.uint8)).save(os.path.join(CONTROL_OUTPUTS_JPG, 'simulated_gt', crop_name, f'gt_dir.jpg'), quality=90)


    if True:
        # Create concatenated images
        # output crop names to json


        use_center_image = False
        ours = 'shadow_comp_post'
        with open(os.path.join(CONTROL_OUTPUTS_MP4_CONCAT, 'crop_names.json'), 'w') as f:
            json.dump(crop_names, f, indent=4)
        for crop_name in tqdm(crop_names, desc='Creating concatenated images'):
            # Iterate over other techniques to compare with 'shadow_comp'
            for other_technique_name in ['diffusionlight']:
                if other_technique_name == ours:
                    continue
                
                for ordering in [0, 1]:
                    if ordering == 0:
                        left_technique_name = ours
                        right_technique_name = other_technique_name
                    else:
                        left_technique_name = other_technique_name
                        right_technique_name = ours
                        
                    sampled_direction = np.random.randint(0, MAX_LIGHT_DIRECTIONS)
                    left_image = os.path.join(CONTROL_OUTPUTS_PNG, left_technique_name, crop_name, f'{sampled_direction:04}.png')
                    right_image = os.path.join(CONTROL_OUTPUTS_PNG, right_technique_name, crop_name, f'{sampled_direction:04}.png')
                    output_video = os.path.join(CONTROL_OUTPUTS_MP4_CONCAT, f'{crop_name}.{left_technique_name}.{right_technique_name}.mp4')

                    # Check if all video files exist
                    if not all(os.path.exists(v) for v in [left_image, right_image]):
                        print(f"One of the video files is missing for crop '{crop_name}' and technique '{other_technique_name}'. Skipping concatenation.")
                        continue
                    
                    # numpy concat
                    left_image = PIL.Image.open(left_image)
                    right_image = PIL.Image.open(right_image)
                    
                    if use_center_image:
                        center_image = os.path.join(CONTROL_OUTPUTS_PNG, 'shadow', crop_name, f'{sampled_direction:04}.png')
                        center_image = PIL.Image.open(center_image)

                        concat_image = PIL.Image.new('RGB', (left_image.width + center_image.width + right_image.width, left_image.height))
                        concat_image.paste(left_image, (0, 0))
                        concat_image.paste(center_image, (left_image.width, 0))
                        concat_image.paste(right_image, (left_image.width + center_image.width, 0))
                    else:
                        concat_image = PIL.Image.new('RGB', (left_image.width + right_image.width, left_image.height))
                        concat_image.paste(left_image, (0, 0))
                        concat_image.paste(right_image, (left_image.width, 0))
                    # generate random int
                    rand_int = np.random.randint(0, 1e7)
                    concat_image.save(os.path.join(CONTROL_OUTPUTS_RANDOM_CONCAT, f'{rand_int:07}_{crop_name}.{left_technique_name}.{right_technique_name}.png'))   


    if False:
        for other_technique_name in [*IMAGE_FOLDERS, 'shadow']:
            technique_mp4_folder = os.path.join(CONTROL_OUTPUTS_MP4, other_technique_name)
            os.makedirs(technique_mp4_folder, exist_ok=True)
            for crop_name in tqdm(crop_names, desc=f'Creating videos for {other_technique_name}'):
                input_pattern = os.path.join(CONTROL_OUTPUTS_PNG, other_technique_name, crop_name, '%04d.png')  # Use PNG images
                output_video = os.path.join(technique_mp4_folder, f'{crop_name}.mp4')
                ffmpeg_command = [
                    'ffmpeg',
                    '-y',  # Overwrite output files without asking
                    '-framerate', str(FRAMES_PER_SECOND),  # Set input frame rate
                    '-i', input_pattern,  # Input files
                    '-c:v', 'libx264',  # Video codec
                    '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                    output_video  # Output file
                ]
                subprocess.run(ffmpeg_command)


    if False:
        # Create concatenated videos
        # output crop names to json
        with open(os.path.join(CONTROL_OUTPUTS_MP4_CONCAT, 'crop_names.json'), 'w') as f:
            json.dump(crop_names, f, indent=4)
        for crop_name in tqdm(crop_names, desc='Creating concatenated videos'):
            # Iterate over other techniques to compare with 'shadow_comp'
            for other_technique_name in IMAGE_FOLDERS.keys():
                if other_technique_name == 'shadow_comp':
                    continue
                
                for ordering in [0, 1]:
                    if ordering == 0:
                        left_technique_name = 'shadow_comp'
                        right_technique_name = other_technique_name
                    else:
                        left_technique_name = other_technique_name
                        right_technique_name = 'shadow_comp'
                        
                    
                    left_video = os.path.join(CONTROL_OUTPUTS_MP4, left_technique_name, f'{crop_name}.mp4')
                    center_video = os.path.join(CONTROL_OUTPUTS_MP4, 'shadow', f'{crop_name}.mp4')
                    right_video = os.path.join(CONTROL_OUTPUTS_MP4, right_technique_name, f'{crop_name}.mp4')
                    output_video = os.path.join(CONTROL_OUTPUTS_MP4_CONCAT, f'{crop_name}.{left_technique_name}.{right_technique_name}.mp4')

                    # Check if all video files exist
                    if not all(os.path.exists(v) for v in [left_video, center_video, right_video]):
                        print(f"One of the video files is missing for crop '{crop_name}' and technique '{other_technique_name}'. Skipping concatenation.")
                        continue
                    
                    # Use ffmpeg to concatenate videos side by side
                    ffmpeg_command = [
                        'ffmpeg',
                        '-y',  # Overwrite output files without asking
                        '-i', left_video,
                        '-i', center_video,
                        '-i', right_video,
                        '-filter_complex', '[0:v][1:v][2:v]hstack=inputs=3',
                        '-c:v', 'libx264',  # Video codec
                        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                        output_video
                    ]
                    subprocess.run(ffmpeg_command)
