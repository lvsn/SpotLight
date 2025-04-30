from torch.utils.data import Dataset
import torch
import os
import glob
from PIL import Image
import numpy as np
from hdrio import ezexr
import cv2



def gamma_correction(img, gamma=2.2):
    return (img.clip(0, 1) ** (1/gamma)).clip(0, 1)

def load_image(image_name, isGamma=False):
    image = cv2.imread(image_name, -1)
    image = np.asarray(image, dtype=np.float32)

    image = image/255.0
    if isGamma:
        image = image**2.2
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    if len(image.shape) == 3:
        image = image[:, :, ::-1]

    return np.ascontiguousarray(image)


class RenderDataset(Dataset):
    def __init__(self, root_dir, bg_estimates_dir=None, transforms=None, to_controlnet_input=None, shadows_dir=None, force_roughness_value=None, force_metallic_value=None, force_albedo_value=None):
        # Resize: If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        self.transforms = transforms
        self.to_controlnet_input = to_controlnet_input
        # One exr file contains all the images
        self.directory_files = sorted(glob.glob(os.path.join(root_dir, '*', '*bundle*.exr')))

        lack_list = []
        # Check if file exists
        assert len(self.directory_files) > 0 and len(self.directory_files) > 0, f'Filelist is empty!'
        for i, p in enumerate(self.directory_files):
            if not os.path.isfile(p):
                print(f'{p} does not exist, deprecating...')
                if i not in lack_list:
                    lack_list.append(i)
            if shadows_dir:
                if not os.path.exists(os.path.join(shadows_dir, os.path.basename(os.path.dirname(p)))):
                    if i not in lack_list:
                        lack_list.append(i)

        self.directory_files = [p for i, p in enumerate(self.directory_files) if i not in lack_list]

        for p in self.directory_files:
            assert os.path.isfile(p), f'{p} does not exist'

        self.bg_estimates_dir = bg_estimates_dir

        self.shadows_dir = shadows_dir
        self.force_roughness_value = force_roughness_value
        self.force_metallic_value = force_metallic_value
        self.force_albedo_value = force_albedo_value

    def __len__(self):
        return len(self.directory_files)

    def __getitem__(self, idx):
        # Source provide the object, destination provide the background
        src_exr = ezexr.imread(self.directory_files[idx], rgb="hybrid")
        # Get name of the file
        name = os.path.basename(self.directory_files[idx]).split('.')[0][:-11]
        # object brdf
        src_diffuse = src_exr['albedo']
        if self.force_albedo_value is not None:
            src_diffuse[:, :, 0] = self.force_albedo_value * src_diffuse[:, :, 3]
            src_diffuse[:, :, 1] = self.force_albedo_value * src_diffuse[:, :, 3]
            src_diffuse[:, :, 2] = self.force_albedo_value * src_diffuse[:, :, 3]
            
        src_roughness = src_exr['roughness'][:, :, 2:4] # value, alpha channel
        if self.force_roughness_value is not None:
            src_roughness[:, :, 0] = self.force_roughness_value * src_roughness[:, :, 1]
        src_metallic = src_exr['metallic'][:, :, 2:4] # value, alpha channel
        if self.force_metallic_value is not None:
            src_metallic[:, :, 0] = self.force_metallic_value * src_metallic[:, :, 1]
        # debug save diffuse
        # ezexr.imwrite(f'./tmp/diffuse_{idx}.exr', src_diffuse)
        # cv2.imwrite(f'./tmp/diffuse_{idx}.png', cv2.cvtColor((src_diffuse ** (1/2.2) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        src_depth = src_exr['depth'][:, :, 2:4] # value, alpha channel
        src_footprint_depth = src_exr['footprint_depth'][:, :, 2:4] # value, alpha channel
        src_mask = src_exr['mask'] # alpha channel of the rendered foreground (can be different from intrinsics)
        src_normal = src_exr['normals'] # RGBA
        src_normal[:, :, :3] = src_normal[:, :, :3] * 0.5 + 0.5 * src_normal[:, :, 3:4]

        src_obj = gamma_correction(src_exr['foreground'][:, :, :3])
        dst_bg = gamma_correction(src_exr['background'][:, :, :3])
        dst_comp = gamma_correction(src_exr['composite'][:, :, :3])


        sample = {
            'name': name,
            'depth': src_depth,
            'normal': src_normal,
            'diffuse': src_diffuse,
            'roughness': src_roughness,
            'metallic': src_metallic,
            'pixel_values': dst_bg,
            'mask': src_mask,
            'src_obj': src_obj,
            'comp': dst_comp,
            'footprint_depth': src_footprint_depth,
        }
        if self.shadows_dir:
            shadow_map_idx = 0
            while os.path.exists(os.path.join(self.shadows_dir, name, f'shadow_positive_{shadow_map_idx:04d}.png')):
                shadow_positive_shadow_path = os.path.join(self.shadows_dir, name, f'shadow_positive_{shadow_map_idx:04d}.png')
                shadow_negative_shadow_path = os.path.join(self.shadows_dir, name, f'shadow_negative_{shadow_map_idx:04d}.png')
                sample[f'shadow_positive_{shadow_map_idx:04d}'] = load_image(shadow_positive_shadow_path, isGamma=False)[:,:,:1]
                sample[f'shadow_negative_{shadow_map_idx:04d}'] = load_image(shadow_negative_shadow_path, isGamma=False)[:,:,:1]
                print(f'shadow_positive_{shadow_map_idx:04d}')
                shadow_map_idx += 1
            
            if os.path.exists(os.path.join(self.shadows_dir, name, f'shadow_positive_gt_dir.png')):
                shadow_positive_shadow_path = os.path.join(self.shadows_dir, name, f'shadow_positive_gt_dir.png')
                shadow_negative_shadow_path = os.path.join(self.shadows_dir, name, f'shadow_negative_gt_dir.png')
                sample[f'shadow_positive_gt_dir'] = load_image(shadow_positive_shadow_path, isGamma=False)[:,:,:1]
                sample[f'shadow_negative_gt_dir'] = load_image(shadow_negative_shadow_path, isGamma=False)[:,:,:1]
            

        intrinsic_name = name
        if self.bg_estimates_dir is not None:
            bg_depth_path = os.path.join(self.bg_estimates_dir, name, intrinsic_name + '_bg_depth.exr')
            if os.path.exists(bg_depth_path):
                bg_depth = ezexr.imread(bg_depth_path)[:, :, :1]
                sample['bg_depth'] = bg_depth
            else:
                print(f'{bg_depth_path} does not exist')

            bg_normal_path = os.path.join(self.bg_estimates_dir, name, intrinsic_name + '_bg_normal.png')
            if os.path.exists(bg_normal_path):
                bg_normal = load_image(bg_normal_path, isGamma=False)
                sample['bg_normal'] = bg_normal
            else:
                print(f'{bg_normal_path} does not exist')
                
            bg_diffuse_path = os.path.join(self.bg_estimates_dir, name, intrinsic_name + '_bg_diffuse.png')
            if os.path.exists(bg_diffuse_path):
                bg_diffuse = load_image(bg_diffuse_path, isGamma=False)
                sample['bg_diffuse'] = bg_diffuse
            else:
                print(f'{bg_diffuse_path} does not exist')

            bg_roughness_path = os.path.join(self.bg_estimates_dir, name, intrinsic_name + '_bg_roughness.png')
            if os.path.exists(bg_roughness_path):
                bg_roughness = load_image(bg_roughness_path, isGamma=False)
                sample['bg_roughness'] = bg_roughness
            else:
                print(f'{bg_roughness_path} does not exist')

            bg_metallic_path = os.path.join(self.bg_estimates_dir, name, intrinsic_name + '_bg_metallic.png')
            if os.path.exists(bg_metallic_path):
                bg_metallic = load_image(bg_metallic_path, isGamma=False)
                sample['bg_metallic'] = bg_metallic
            else:
                print(f'{bg_metallic_path} does not exist')

        if self.transforms:
            sample = self.transforms(sample)

        if self.to_controlnet_input:
            sample = self.to_controlnet_input(sample)

        return sample

