import diffusers
import torch
from data.dataset_render import RenderDataset
from data.dataset_labo import LaboDataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as tf
import torch.nn.functional as F

import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import ezexr
import os

from controlnet_input_handle import match_depth_from_footprint, handle_depth_anything, handle_zoedepth, handle_depth_anything_v2_relative, handle_depth_anything_v2_metric

if __name__ == '__main__':
    mode = 'depth'
    device = 'cuda:0'
    # model_name = 'depthanything'
    # model_name = 'zoedepth'
    # model_name = 'marigold'
    model_name = 'depthanythingv2_relative'
    # model_name = 'depthanythingv2_metric'
    # dataset_name = 'render'
    # dataset_name = 'labo'
    dataset_name = 'fred'

    if model_name == 'depthanything':
        dp_model = handle_depth_anything().to(device)
        dp_model.eval()
    elif model_name == 'zoedepth':
        dp_model = handle_zoedepth().to(device)
        dp_model.eval()
    elif model_name == 'marigold':
        model_paper_kwargs = {
            diffusers.schedulers.DDIMScheduler: {
                "num_inference_steps": 10,
                "ensemble_size": 10,
            },
            diffusers.schedulers.LCMScheduler: {
                "num_inference_steps": 4,
                "ensemble_size": 5,
            },
        }
        vae = diffusers.AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", torch_dtype=torch.float16
        ).to(device)

        dp_model = diffusers.MarigoldDepthPipeline.from_pretrained(
            "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
        ).to(device)
        pipe_kwargs = model_paper_kwargs[type(dp_model.scheduler)]
        dp_model.vae = vae
    elif model_name == 'depthanythingv2_relative':
        dp_model = handle_depth_anything_v2_relative().to(device)
        dp_model.eval()
    elif model_name == 'depthanythingv2_metric':
        dp_model = handle_depth_anything_v2_metric().to(device)
        dp_model.eval()

    transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=[512, ], antialias=True),
        v2.CenterCrop([512, 512])
    ])

    if dataset_name == 'render':
        dataset_folder = '../datasets/render/GT_emission_envmap'
        save_folder = f'../datasets/render/reconstructed_obj/GT_emission_envmap_{model_name}'
        dataset = RenderDataset(dataset_folder, transforms=transforms)
    elif dataset_name == 'labo':
        dataset_folder = '../datasets/labo/GT_emission_envmap'
        save_folder = f'../datasets/labo/reconstructed_obj/GT_emission_envmap_{model_name}'
        dataset = LaboDataset(dataset_folder, transforms=transforms)
    else:
        dataset_folder = f'../datasets/{dataset_name}/GT_emission_envmap'
        save_folder = f'../datasets/{dataset_name}/reconstructed_obj/GT_emission_envmap_{model_name}'
        dataset = LaboDataset(dataset_folder, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    with torch.inference_mode():
        for i, sample in enumerate(tqdm(dataloader)):
            name = sample['name']

            bg_image = sample['pixel_values'].to(device)
            fp_depth = sample['footprint_depth'].to(device)
            obj_depth = sample['depth'].to(device)

            if model_name == 'depthanything':
                bg_depth = dp_model.infer(bg_image, pad_input=True, with_flip_aug=True)
                bg_depth_matched = match_depth_from_footprint(bg_depth, obj_depth, fp_depth, True)
            elif model_name == 'zoedepth':
                bg_depth = dp_model(bg_image)['metric_depth']
                bg_depth = F.interpolate(bg_depth, size=bg_image.shape[2:], mode='bilinear', align_corners=True)
                bg_depth_matched = match_depth_from_footprint(bg_depth, obj_depth, fp_depth, True)
            elif model_name == 'marigold':
                bg_depth = dp_model(bg_image, **pipe_kwargs).prediction
                bg_depth = torch.from_numpy(bg_depth).permute([0, 3, 1, 2])
                bg_depth_matched = match_depth_from_footprint(bg_depth, obj_depth, fp_depth, True)
            elif model_name == 'depthanythingv2_relative':
                bg_depth = dp_model.infer_tensor(bg_image)
                fp_depth_mask = fp_depth > 0
                fp_depth[fp_depth_mask] = 1 / fp_depth[fp_depth_mask]
                bg_depth_matched = match_depth_from_footprint(bg_depth, obj_depth, fp_depth, adjust_background=True)
                bg_depth_matched = 1 / bg_depth_matched
            elif model_name == 'depthanythingv2_metric':
                bg_depth = dp_model.infer_tensor(bg_image)
                bg_depth_matched = match_depth_from_footprint(bg_depth, obj_depth, fp_depth, True)

            for b in range(bg_image.shape[0]):
                bg_depth_b = bg_depth[b].cpu().squeeze().numpy()

                save_dir = os.path.join(save_folder, name[b], 'obj')
                os.makedirs(save_dir, exist_ok=True)
                ezexr.imwrite(os.path.join(save_dir, f'{name[b]}_bg_{mode}.exr'), bg_depth_b)

                bg_depth_matched_b = bg_depth_matched[b].cpu().squeeze().numpy()
                ezexr.imwrite(os.path.join(save_dir, f'{name[b]}_bg_{mode}_{model_name}_matched.exr'), bg_depth_matched_b)
