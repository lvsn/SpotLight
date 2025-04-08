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
    def __init__(self, root_dir, bg_estimates_dir, transforms=None, to_controlnet_input=None, scribbles_dir=None, force_roughness_value=None, force_metallic_value=None, force_albedo_value=None):
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
            if scribbles_dir:
                #print(os.path.join(scribbles_dir, os.path.basename(os.path.dirname(p))))
                if not os.path.exists(os.path.join(scribbles_dir, os.path.basename(os.path.dirname(p)))):
                    #print(f'{os.path.join(scribbles_dir, os.path.dirname(p))} does not exist, deprecating...')
                    if i not in lack_list:
                        lack_list.append(i)

        self.directory_files = [p for i, p in enumerate(self.directory_files) if i not in lack_list]

        for p in self.directory_files:
            assert os.path.isfile(p), f'{p} does not exist'

        self.bg_estimates_dir = bg_estimates_dir

        self.scribbles_dir = scribbles_dir
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
        if self.scribbles_dir:
            shadow_map_idx = 0
            while os.path.exists(os.path.join(self.scribbles_dir, name, f'{name}_shadowmap_{shadow_map_idx}.png')):
                
                positive_shadow_path = os.path.join(self.scribbles_dir, name, f'{name}_shadowmap_{shadow_map_idx}.png')
                negative_shadow_path = os.path.join(self.scribbles_dir, name, f'{name}_shadowmap_neg_{shadow_map_idx}.png')
                sample[f'positive_shadow_{shadow_map_idx}'] = load_image(positive_shadow_path, isGamma=False)[:,:,:1]
                sample[f'negative_shadow_{shadow_map_idx}'] = load_image(negative_shadow_path, isGamma=False)[:,:,:1]
                shadow_map_idx += 1

        intrinsic_name = name
        # Rebuttal
        # TODO: update depth
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



if __name__ == '__main__':
    dataloader = RenderDataset('../datasets/labo/GT_emission_envmap')
    save_dir = '../datasets/labo_png'
    from torch.utils.data import DataLoader
    import cv2
    from PIL import Image
    dataloader = DataLoader(dataloader, batch_size=1, shuffle=False, num_workers=4)

    def tensor_to_numpy(img, initial_range=(0, 1)):
        # scale to [0, 1]
        img = img - initial_range[0]
        img = img / (initial_range[1] - initial_range[0])
        if img.dim() == 4:
            img = img.squeeze(0)
        if img.shape[0] <= 3:
            img = img.permute(1, 2, 0)
        return np.clip(img.cpu().numpy(), 0, 1)

    def numpy_to_pil(img):
        img = (img * 255.0).astype("uint8")
        if img.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_img = Image.fromarray(img.squeeze(), mode="L")
        else:
            pil_img = Image.fromarray(img, mode="RGB")
        return pil_img

    def tensor_to_pil(img, initial_range=(0, 1)):
        img = tensor_to_numpy(img, initial_range)
        img = numpy_to_pil(img)
        return img

    import time
    from tqdm import tqdm
    t1 = time.time()
    for i, sample in enumerate(tqdm(dataloader)):
        # print(i, sample['pixel_values'].shape)
        # if i == 10:
        #     break

        # Dilated object mask
        # mask = sample['mask'].squeeze().numpy()
        # dilated_mask = cv2.dilate(mask, np.ones((15, 15)), iterations=1)
        # cv2.imwrite(f'./tmp/mask_{i}.png', mask * 255)
        # cv2.imwrite(f'./tmp/dilated_mask_{i}.png', dilated_mask * 255)

        # Output diffuse
        # ezexr.imwrite(f'./tmp/diffuse_{i}.exr', sample['diffuse'].squeeze().numpy())
        # df = sample['diffuse'].squeeze().numpy()
        # ezexr.imwrite(f'./tmp/diffuse_{i}.exr', df)
        # df_erosion = cv2.erode(df, np.ones((2, 2)), iterations=1)
        # ezexr.imwrite(f'./tmp/diffuse_erosion_{i}.exr', df_erosion)

        # # Naive composite
        # dst_bg = sample['pixel_values']
        # src_comp = sample['src_comp']
        # mask = sample['mask'].expand_as(dst_bg)

        # comp = dst_bg * (1 - mask) + src_comp * mask
        # comp_save = Image.fromarray((comp.squeeze().numpy() * 255).astype(np.uint8))
        # comp_save.save(f'./tmp/comp_{i}.png')

        # # Save images
        # name = f'{sample["name"][0]}'
        # img_dir = os.path.join(save_dir, name)
        # os.makedirs(img_dir, exist_ok=True)

        # # Background and its intrinsics
        # bg = sample['pixel_values']
        # # bg_df = sample['bg_diffuse']
        # # bg_rg = sample['bg_roughness']
        # # bg_mt = sample['bg_metallic']

        # # Object and its intrinsics
        # fg = sample['src_obj']
        # mask = sample['mask']
        # obj_dp = sample['depth']
        # obj_nm = sample['normal']
        # obj_df = sample['diffuse']
        # obj_df_gamma = sample['diffuse'] ** (1/2.2)
        # obj_rg = sample['roughness']
        # obj_mt = sample['metallic']

        # comp = sample['comp']

        # # Copy paste composite
        # cp_comp = bg.clone()
        # tmp_mask = mask.expand_as(cp_comp)
        # cp_comp[tmp_mask > 0.9] = fg[tmp_mask > 0.9]

        # tensor_to_pil(bg[0]).save(os.path.join(img_dir,      'bg.png'))
        # tensor_to_pil(mask[0]).save(os.path.join(img_dir,    'mask.png'))
        # # tensor_to_pil(bg_df[0]).save(os.path.join(img_dir, 'bg_df.png'))
        # # tensor_to_pil(bg_rg[0]).save(os.path.join(img_dir, 'bg_rg.png'))
        # # tensor_to_pil(bg_mt[0]).save(os.path.join(img_dir, 'bg_mt.png'))

        # tensor_to_pil(fg[0]).save(os.path.join(img_dir,      'obj.png'))
        # tensor_to_pil(obj_dp[0]).save(os.path.join(img_dir,  'obj_dp.png'))
        # tensor_to_pil(obj_nm[0]).save(os.path.join(img_dir,  'obj_nm.png'))
        # tensor_to_pil(obj_df[0]).save(os.path.join(img_dir,  'obj_df.png'))
        # tensor_to_pil(obj_df_gamma[0]).save(os.path.join(img_dir, 'obj_df_gamma.png'))
        # tensor_to_pil(obj_rg[0]).save(os.path.join(img_dir,  'obj_rg.png'))
        # tensor_to_pil(obj_mt[0]).save(os.path.join(img_dir,  'obj_mt.png'))

        # tensor_to_pil(cp_comp[0]).save(os.path.join(img_dir, 'cp.png'))
        # tensor_to_pil(comp[0]).save(os.path.join(img_dir,    'comp.png'))

        # break

        pass