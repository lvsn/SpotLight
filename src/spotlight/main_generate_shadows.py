import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
import logging
from PIL import Image
from tqdm import tqdm
import zerocomp.sdi_utils as sdi_utils
import hydra
import json
import zerocomp.compositor as compositor
from omegaconf import OmegaConf

EPS = 1e-6

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@hydra.main(config_path="../../configs", config_name="default", version_base='1.1')
def main(args):
    results_dir = args.shadows_dir
    assert not os.path.exists(results_dir), f"The output directory {results_dir} already exists, aborting."
    os.makedirs(results_dir)

    OmegaConf.save(config=args, f=os.path.join(results_dir, "config.yaml"))

    sdi_utils.seed_all(args.seed)

    val_dataloader = compositor.create_dataloader(args, to_controlnet_input=None, start_batch=args.eval.start_batch) # TODO: end batch too?

    for batch_idx, batch in enumerate(tqdm(val_dataloader)):
        scene_name = batch['name'][0].replace('_bundle0001', '')
        scene_dir = os.path.join(results_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True) 
        
        if args.eval.obj_mask_dilation_shadowcomp > 0:
            # usually a dilation of 1 is good, similar to ZeroComp (because of depth antialiasing)
            dilated_obj_mask = F.max_pool2d(batch['mask'], kernel_size=args.eval.obj_mask_dilation_shadowcomp * 2 + 1, stride=1, padding=args.eval.obj_mask_dilation_shadowcomp)
        else:
            dilated_obj_mask = batch['mask']
        dilated_obj_mask = dilated_obj_mask.to(args.eval.device)

        light_path_filename = os.path.join(args.dataset_dir, scene_name, f"{scene_name}_light_path_dto.json")
        with open(light_path_filename, 'r') as f:
            light_path = json.load(f)
        all_frames = light_path['light_frames']
        
        for frame_idx, frame_dict in enumerate(all_frames):
            # defaults
            frame_name = frame_dict['name']

            shadow_map_name = f"{scene_name}_light{frame_name}"

            shadow_positive = compositor.render_numpy_shadow(scene_name, shadow_map_name, frame_dict['blender_light_xyz_position_object_space'], args.dataset_dir, scene_dir, background_mesh_dir=args.background_mesh_dir, objects_dir=args.objects_dir, 
                                                            light_distance=args.eval.light_distance, light_radius=args.eval.light_radius, debug_mode=args.shadow_generation_debug_mode)
            shadow_positive = sdi_utils.numpy_to_tensor(shadow_positive).to(dtype=batch['mask'].dtype).to(batch['mask'].device) * (1 - batch['mask'])

            blender_light_xyz_position_object_space_opposite = [
                -frame_dict['blender_light_xyz_position_object_space'][0],
                -frame_dict['blender_light_xyz_position_object_space'][1],
                frame_dict['blender_light_xyz_position_object_space'][2]
            ]
            shadow_opposite = compositor.render_numpy_shadow(scene_name, shadow_map_name, blender_light_xyz_position_object_space_opposite, args.dataset_dir, scene_dir, background_mesh_dir=args.background_mesh_dir, objects_dir=args.objects_dir,
                                                                light_distance=args.eval.light_distance, light_radius=args.eval.light_radius, debug_mode=args.shadow_generation_debug_mode)
            shadow_opposite = sdi_utils.numpy_to_tensor(shadow_opposite).to(dtype=batch['mask'].dtype).to(batch['mask'].device) * (1 - batch['mask'])
                
            background_coarse = batch['pixel_values'] * (1 - args.eval.shadow_opacity * torch.cat([shadow_positive, shadow_opposite]))
            shadow_composites = compositor.alpha_blend(batch['diffuse'], background_coarse)

            shadow_positive_np = sdi_utils.tensor_to_numpy(shadow_positive)
            shadow_negative_np = sdi_utils.tensor_to_numpy(shadow_opposite)
            shadow_positive_composite_np = sdi_utils.tensor_to_numpy(shadow_composites[0])
            shadow_negative_composite_np = sdi_utils.tensor_to_numpy(shadow_composites[1])

            shadow_positive_outline = compositor.get_outline_visualization(background_coarse[0].unsqueeze(0), batch['mask'][0].unsqueeze(0))
            shadow_positive_outline_np = sdi_utils.tensor_to_numpy(shadow_positive_outline)
            shadow_negative_outline = compositor.get_outline_visualization(background_coarse[1].unsqueeze(0), batch['mask'][0].unsqueeze(0))
            shadow_negative_outline_np = sdi_utils.tensor_to_numpy(shadow_negative_outline)

            # export
            Image.fromarray((np.repeat(shadow_positive_np, 3, axis=2) * 255).astype(np.uint8)).save(os.path.join(scene_dir, f"shadow_positive_{frame_name}.png"))
            Image.fromarray((np.repeat(shadow_negative_np, 3, axis=2) * 255).astype(np.uint8)).save(os.path.join(scene_dir, f"shadow_negative_{frame_name}.png"))
            Image.fromarray((shadow_positive_composite_np * 255).astype(np.uint8)).save(os.path.join(scene_dir, f"shadow_positive_composite_{frame_name}.png"))
            Image.fromarray((shadow_negative_composite_np * 255).astype(np.uint8)).save(os.path.join(scene_dir, f"shadow_negative_composite_{frame_name}.png"))
            Image.fromarray((shadow_positive_outline_np * 255).astype(np.uint8)).save(os.path.join(scene_dir, f"shadow_positive_outline_{frame_name}.png"))
            Image.fromarray((shadow_negative_outline_np * 255).astype(np.uint8)).save(os.path.join(scene_dir, f"shadow_negative_outline_{frame_name}.png"))


if __name__ == "__main__":
    main()
