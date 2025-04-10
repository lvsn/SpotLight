import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
import logging
from PIL import Image
import PIL.Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from datetime import datetime
import re
import sdi_utils
import hydra
from tempfile import TemporaryDirectory
from controlnet_input_handle import ToControlNetInput, ToPredictors
from copy import deepcopy
import json
import compositor
from omegaconf import OmegaConf
import PIL.Image

EPS = 1e-6

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@hydra.main(config_path="../../configs", config_name="default", version_base='1.1')
def main(args):
    # use that folder format: %Y-%m-%d_%H-%M-%S
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"{args.eval.results_dir}/{current_time}_control_dataset"
    if args.eval.run_name != '':
        results_dir += f"_{args.eval.run_name}"
    os.makedirs(results_dir, exist_ok=True)

    OmegaConf.save(config=args, f=os.path.join(results_dir, "config.yaml"))

    sdi_utils.seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    to_controlnet_input = ToControlNetInput(
        device=args.eval.device,
        feed_empty_prompt=args.feed_empty_prompt,
        tokenizer=tokenizer,
        for_sdxl=False
    )
    val_dataloader = compositor.create_dataloader(args, to_controlnet_input, start_batch=args.eval.start_batch) # TODO: end batch too?

    for batch_idx, batch in enumerate(tqdm(val_dataloader)):
        with torch.inference_mode():
            with TemporaryDirectory() as tmp_dir:
                
                if args.dataset_name == 'real_world':
                    full_name = batch['name'][0]
                    match = re.match(r'([a-z]+-[0-9]+_obj-[0-9]+)_(.*)', full_name)
                    scene_name = match.group(1)
                    frame_name = match.group(2)
                else:
                    scene_name = batch['name'][0].replace('_bundle0001', '')

                if args.eval.obj_mask_dilation_shadowcomp > 0:
                    # usually a dilation of 1 is good, similar to ZeroComp (because of depth antialiasing)
                    dilated_obj_mask = F.max_pool2d(batch['mask'], kernel_size=args.eval.obj_mask_dilation_shadowcomp * 2 + 1, stride=1, padding=args.eval.obj_mask_dilation_shadowcomp)
                else:
                    dilated_obj_mask = batch['mask']
                dilated_obj_mask = dilated_obj_mask.to(args.eval.device)


                # auto_azimuth, auto_zenith = sh_light_utils.cartesian_to_spherical(*auto_light_position_skylibs)
                # auto_azimuth_deg, auto_zenith_deg = np.rad2deg(auto_azimuth), np.rad2deg(auto_zenith)
                
                if args.dataset_name == 'real_world':
                    frame_dict = {
                        'name': frame_name,
                    }
                    all_frames = [frame_dict]
                elif args.dataset_name == 'scribbles':
                    all_frames = []
                    
                    shadow_map_idx = 0
                    while os.path.exists(os.path.join(args.scribbles_dir, scene_name, f'{scene_name}_shadowmap_{shadow_map_idx}.png')):
                        all_frames.append({
                            'name': f'frame{shadow_map_idx}',
                            'shadow_map_idx': shadow_map_idx,
                        })
                        shadow_map_idx += 1

                else:
                    light_path_filename = os.path.join(args.dataset_dir, scene_name, f"{scene_name}_light_path_dto.json")
                    with open(light_path_filename, 'r') as f:
                        light_path = json.load(f)
                    all_frames = light_path['light_frames']
                    

                for frame_idx, frame_dict in enumerate(all_frames):
                    # defaults
                    frame_name = frame_dict['name']

                    if args.eval.render_only_gt_dir and frame_name != 'gt_dir':
                        continue

                    shadow_map_name = f"{scene_name}_light{frame_name}"
                    
                    print('rendering in blender... ')
                    if args.dataset_name == 'real_world':
                        shadow_custom = (1 - batch['pos_shadow']) * (1 - batch['mask'])
                    elif args.dataset_name == 'scribbles':
                        shadow_custom = batch[f'positive_shadow_{frame_dict["shadow_map_idx"]}'] * (1 - batch['mask'])
                    else:
                        shadow_custom = compositor.render_numpy_shadow(scene_name, shadow_map_name, frame_dict['blender_light_xyz_position_object_space'], args.dataset_dir, results_dir, background_mesh_dir=args.background_mesh_dir, objects_dir=args.objects_dir, 
                                                                       light_distance=args.eval.light_distance, light_radius=args.eval.light_radius)
                        shadow_custom = sdi_utils.numpy_to_tensor(shadow_custom).to(dtype=batch['mask'].dtype).to(batch['mask'].device) * (1 - batch['mask'])

                    if args.eval.negative_sample_type == 'opposite_azimuth':
                        blender_light_xyz_position_object_space_opposite = [
                            -frame_dict['blender_light_xyz_position_object_space'][0],
                            -frame_dict['blender_light_xyz_position_object_space'][1],
                            frame_dict['blender_light_xyz_position_object_space'][2]
                        ]
                        shadow_opposite = compositor.render_numpy_shadow(scene_name, shadow_map_name, blender_light_xyz_position_object_space_opposite, args.dataset_dir, results_dir, background_mesh_dir=args.background_mesh_dir, objects_dir=args.objects_dir,
                                                                         light_distance=args.eval.light_distance, light_radius=args.eval.light_radius)
                        shadow_opposite = sdi_utils.numpy_to_tensor(shadow_opposite).to(dtype=batch['mask'].dtype).to(batch['mask'].device) * (1 - batch['mask'])

                    elif args.eval.negative_sample_type == 'precomputed_shadowmap':
                        if args.dataset_name == 'real_world':
                            shadow_opposite = (1 - batch[f'neg_shadow'] )* (1 - batch['mask'])
                        elif args.dataset_name == 'scribbles':
                            shadow_opposite = batch[f'negative_shadow_{frame_dict["shadow_map_idx"]}'] * (1 - batch['mask'])
                    

                    print('run_inference... ')
                    if args.eval.negative_sample_type == 'opposite_azimuth':
                        shadow_negative_sample = shadow_opposite
                    elif args.eval.negative_sample_type == 'zerocomp':
                        shadow_negative_sample = torch.zeros_like(shadow_custom)
                    elif args.eval.negative_sample_type == 'no_shadow':
                        shadow_negative_sample = torch.zeros_like(shadow_custom)
                    elif args.eval.negative_sample_type == 'precomputed_shadowmap':
                        if args.dataset_name == 'real_world':
                            shadow_negative_sample = (1 - batch[f'neg_shadow']) * (1 - batch['mask'])
                        else:
                            shadow_negative_sample = batch[f'negative_shadow_{frame_dict["shadow_map_idx"]}'] * (1 - batch['mask'])
                        
                    background_coarse = batch['pixel_values'] * (1 - args.eval.shadow_opacity * torch.cat([shadow_custom, shadow_negative_sample]))
                    shadow_composites = compositor.alpha_blend(batch['diffuse'], background_coarse)

                    if args.eval.output_shadow_comp_intermediate_images:
                        shadow_positive_np = sdi_utils.tensor_to_numpy(shadow_custom)
                        shadow_negative_np = sdi_utils.tensor_to_numpy(shadow_negative_sample)
                        shadow_positive_composite_np = sdi_utils.tensor_to_numpy(shadow_composites[0])
                        shadow_negative_composite_np = sdi_utils.tensor_to_numpy(shadow_composites[1])

                        shadow_positive_outline = compositor.get_outline_visualization(background_coarse[0].unsqueeze(0), batch['mask'][0].unsqueeze(0))
                        shadow_positive_outline_np = sdi_utils.tensor_to_numpy(shadow_positive_outline)
                        shadow_negative_outline = compositor.get_outline_visualization(background_coarse[1].unsqueeze(0), batch['mask'][0].unsqueeze(0))
                        shadow_negative_outline_np = sdi_utils.tensor_to_numpy(shadow_negative_outline)

                        # export
                        os.makedirs(os.path.join(results_dir, scene_name, 'intermediate'), exist_ok=True) 
                        Image.fromarray((np.repeat(shadow_positive_np, 3, axis=2) * 255).astype(np.uint8)).save(os.path.join(results_dir, scene_name, 'intermediate', f"shadow_positive_{frame_name}.png"))
                        Image.fromarray((np.repeat(shadow_negative_np, 3, axis=2) * 255).astype(np.uint8)).save(os.path.join(results_dir, scene_name, 'intermediate', f"shadow_negative_{frame_name}.png"))
                        Image.fromarray((shadow_positive_composite_np * 255).astype(np.uint8)).save(os.path.join(results_dir, scene_name, 'intermediate', f"shadow_positive_composite_{frame_name}.png"))
                        Image.fromarray((shadow_negative_composite_np * 255).astype(np.uint8)).save(os.path.join(results_dir, scene_name, 'intermediate', f"shadow_negative_composite_{frame_name}.png"))
                        Image.fromarray((shadow_positive_outline_np * 255).astype(np.uint8)).save(os.path.join(results_dir, scene_name, 'intermediate', f"shadow_positive_outline_{frame_name}.png"))
                        Image.fromarray((shadow_negative_outline_np * 255).astype(np.uint8)).save(os.path.join(results_dir, scene_name, 'intermediate', f"shadow_negative_outline_{frame_name}.png"))

                    if args.eval.output_shadow_comp_intermediate_images:
                        weighted_latent_mask = pred_image_logs['weighted_latent_mask']
                        positive_latent_mask_np = sdi_utils.tensor_to_numpy(weighted_latent_mask[0])
                        negative_latent_mask_np = sdi_utils.tensor_to_numpy(weighted_latent_mask[1])
                        Image.fromarray((np.repeat(positive_latent_mask_np, 3, axis=2) * 255).astype(np.uint8)).save(os.path.join(results_dir, scene_name, 'intermediate', f"positive_latent_mask_{frame_name}.png"))
                        Image.fromarray((np.repeat(negative_latent_mask_np, 3, axis=2) * 255).astype(np.uint8)).save(os.path.join(results_dir, scene_name, 'intermediate', f"negative_latent_mask_{frame_name}.png"))


                    os.makedirs(os.path.join(results_dir, scene_name, 'default'), exist_ok=True)
                    img_disp = (sdi_utils.tensor_to_numpy(pred) * 255).astype(np.uint8)
                    img_disp = Image.fromarray(img_disp)
                    img_disp.save(os.path.join(results_dir, scene_name, 'default', f"{frame_name}.png"))
                    print('done default')

if __name__ == "__main__":
    main()
