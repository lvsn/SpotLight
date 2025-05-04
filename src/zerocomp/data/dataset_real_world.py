from torch.utils.data import Dataset
import torch
import os
import glob
from PIL import Image
import numpy as np
from hdrio import ezexr
import cv2


class RealWorldDataset(Dataset):
    def __init__(self, root_dir, transforms=None, to_controlnet_input=None, dataset_subfolder='comp', shadow_channel='dirss'):
        self.root_dir = root_dir
        # Resize: If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        self.transforms = transforms
        self.to_controlnet_input = to_controlnet_input
        # One exr file contains all the images
        self.dataset_subfolder = dataset_subfolder
        self.directory_files = sorted(glob.glob(os.path.join(root_dir, dataset_subfolder, '*', '*_comp.png')))
        self.shadow_channel = shadow_channel

        lack_list = []
        # Check if file exists
        assert len(self.directory_files) > 0 and len(self.directory_files) > 0, f'Filelist is empty!'
        for i, p in enumerate(self.directory_files):
            if p.endswith('neg_comp.png'):
                lack_list.append(i)
                continue
            if not os.path.isfile(p):
                print(f'{p} does not exist, deprecating...')
                if i not in lack_list:
                    lack_list.append(i)

        self.directory_files = [p for i, p in enumerate(self.directory_files) if i not in lack_list]

        for p in self.directory_files:
            assert os.path.isfile(p), f'{p} does not exist'

    def __len__(self):
        return len(self.directory_files)

    def __getitem__(self, idx):
        dst_comp_path = self.directory_files[idx]
        comp_folder = os.path.basename(os.path.dirname(dst_comp_path))
        bg_folder, bg_id = comp_folder.split('_')[0].split('-')
        obj_folder, obj_id = comp_folder.split('_')[1].split('-')
        parts = os.path.basename(dst_comp_path).split('_')
        name = '_'.join(parts[:-1])

        if self.dataset_subfolder == 'comp_relighting':
            bg_path = os.path.join(self.root_dir, bg_folder, f'{bg_id}_obj.png')
            bg_depth_path = os.path.join(self.root_dir, bg_folder, f'{bg_id}_obj_depth.exr')
            bg_normal_path = os.path.join(self.root_dir, bg_folder, f'{bg_id}_obj_normal.png')
            bg_diffuse_path = os.path.join(self.root_dir, bg_folder, f'{bg_id}_obj_diffuse.png')
        else:
            bg_path = os.path.join(self.root_dir, bg_folder, f'{bg_id}_bg.png')
            bg_depth_path = os.path.join(self.root_dir, bg_folder, f'{bg_id}_bg_depth.exr')
            bg_normal_path = os.path.join(self.root_dir, bg_folder, f'{bg_id}_bg_normal.png')
            bg_diffuse_path = os.path.join(self.root_dir, bg_folder, f'{bg_id}_bg_diffuse.png')

        obj_path = os.path.join(self.root_dir, obj_folder, f'{obj_id}_obj.png')
        obj_mask_path = os.path.join(self.root_dir, obj_folder, f'{obj_id}_mask.png')
        obj_depth_path = os.path.join(self.root_dir, obj_folder, f'{obj_id}_obj_depth.exr')
        obj_normal_path = os.path.join(self.root_dir, obj_folder, f'{obj_id}_obj_normal.png')
        obj_diffuse_path = os.path.join(self.root_dir, obj_folder, f'{obj_id}_obj_diffuse.png')

        bg = load_image(bg_path, isGamma=False)
        bg_depth = ezexr.imread(bg_depth_path)
        bg_normal = load_image(bg_normal_path, isGamma=False)
        bg_diffuse = load_image(bg_diffuse_path, isGamma=False)

        obj = load_image(obj_path, isGamma=False)
        obj_mask = load_image(obj_mask_path, isGamma=False)
        obj_depth = ezexr.imread(obj_depth_path)

        obj_normal = load_image(obj_normal_path, isGamma=False)
        # downsample to 512x512
        obj_normal = cv2.resize(obj_normal, (512, 512), interpolation=cv2.INTER_NEAREST)
        obj_normal = np.concatenate([obj_normal, np.ones_like(obj_normal[:, :, :1])], axis=2)
        obj_normal = obj_normal * obj_mask

        obj_diffuse = load_image(obj_diffuse_path, isGamma=False)
        obj_diffuse = np.concatenate([obj_diffuse, np.ones_like(obj_diffuse[:, :, :1])], axis=2)
        obj_diffuse = obj_diffuse * obj_mask

        dst_comp = load_image(dst_comp_path, isGamma=False)
        # final_ss = load_image(dst_comp_path.replace('_comp.png', '_finalss.png'), isGamma=False)
        pos_shadow = load_image(dst_comp_path.replace('_comp.png', f'_{self.shadow_channel}.png'), isGamma=False)[:,:,:1]
        neg_shadow = load_image(dst_comp_path.replace('_comp.png', f'_neg_{self.shadow_channel}.png'), isGamma=False)[:,:,:1]


        sample = {
            'name': name,
            'depth': obj_depth,
            'normal': obj_normal,
            'diffuse': obj_diffuse,
            'pixel_values': bg,
            'mask': obj_mask,
            'src_obj': obj,
            'comp': dst_comp,
            #'footprint_depth': None,
            'bg_depth': bg_depth,
            'bg_normal': bg_normal,
            'bg_diffuse': bg_diffuse,
            'pos_shadow': pos_shadow,
            'neg_shadow': neg_shadow,
            # 'final_ss': final_ss
        }

        if self.transforms:
            sample = self.transforms(sample)

        if self.to_controlnet_input:
            sample = self.to_controlnet_input(sample)

        return sample


def gamma_correction(img, gamma=2.2):
    return (img.clip(0, 1) ** (1 / gamma)).clip(0, 1)


def comp_normal_to_openrooms_normal(normal):
    # Input normal map should be in [-1, 1] range
    if normal.min() >= 0:
        normal = normal * 2 - 1
    normal[:, :, 2] = -normal[:, :, 2]
    # Transform it back to [0, 1] range
    normal = normal * 0.5 + 0.5
    return normal


def load_image(image_name, isGamma=False):
    image = cv2.imread(image_name, -1)
    image = np.asarray(image, dtype=np.float32)

    image = image / 255.0
    if isGamma:
        image = image**2.2
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = image[:, :, ::-1]

    return np.ascontiguousarray(image)


if __name__ == '__main__':
    dataloader = RealWorldDataset('/scratch/frfoc1@ulaval.ca/real_world')
    save_dir = 'test_real_world'
    from torch.utils.data import DataLoader
    import cv2
    from PIL import Image
    dataloader = DataLoader(dataloader, batch_size=1, shuffle=False, num_workers=0)

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

        pass
