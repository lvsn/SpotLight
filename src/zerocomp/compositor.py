import torch
import torch.nn.functional as v2
import numpy as np
import cv2
import open3d as o3d
import sdi_utils
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from packaging import version
from torchvision.transforms import v2
from torchvision.ops import masks_to_boxes
from transformers import CLIPTextModel
import cv2
from diffusers.utils.torch_utils import randn_tensor
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_xformers_available
from data.dataset_render import RenderDataset
from data.dataset_real_world import RealWorldDataset
from controlnet_input_handle import collate_fn, match_depth_from_footprint
import kornia.morphology
import sdi_utils
import open3d as o3d
import subprocess
from controlnet_input_handle import compute_shading
import ezexr

def recursive_info(data):
    if isinstance(data, dict):
        return {key: recursive_info(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [recursive_info(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(recursive_info(item) for item in data)
    elif isinstance(data, torch.Tensor):
        return {"type": "PyTorch Tensor", "shape": list(data.shape)}
    elif isinstance(data, np.ndarray):
        return {"type": "NumPy ndarray", "shape": list(data.shape)}
    else:
        return {"type": type(data).__name__, "value": data}


def load_models(args, tokenizer):
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=None
    )
    
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=None)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=None
    )

    controlnet = ControlNetModel.from_pretrained(args.eval.controlnet_model_name_or_path, subfolder="controlnet")

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    controlnet.eval()

    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        import xformers
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
    elif args.enable_xformers_memory_efficient_attention:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

    return vae, unet, text_encoder, controlnet

def create_dataloader(args, to_controlnet_input, start_batch=0, shuffle=False):
    val_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=[args.resolution, ], antialias=True),
        v2.CenterCrop([args.resolution, args.resolution])
    ])
    
    if args.dataset_name == 'render':
        val_dataset = RenderDataset(args.dataset_dir, args.background_estimated_intrinsics_dir, transforms=val_transforms, to_controlnet_input=to_controlnet_input, force_metallic_value=args.eval.force_metallic_value, force_roughness_value=args.eval.force_roughness_value, force_albedo_value=args.eval.force_albedo_value)
    elif args.dataset_name == 'scribbles':
        val_dataset = RenderDataset(args.dataset_dir, args.background_estimated_intrinsics_dir, transforms=val_transforms, to_controlnet_input=to_controlnet_input, scribbles_dir=args.scribbles_dir, force_metallic_value=args.eval.force_metallic_value, force_roughness_value=args.eval.force_roughness_value, force_albedo_value=args.eval.force_albedo_value)
    elif args.dataset_name == 'real_world':
        val_dataset = RealWorldDataset(args.dataset_dir, transforms=val_transforms, to_controlnet_input=to_controlnet_input, dataset_subfolder=args.dataset_subfolder,
                                       shadow_channel=args.eval.shadow_channel)

    val_dataset = Subset(val_dataset, range(start_batch, len(val_dataset)))
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=shuffle,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn
    )
    return val_dataloader

def alpha_blend(obj_map, bg_map, *, color_channels=3):
    # Alpha blend (assuming premultiplied alpha)

    assert len(obj_map.shape) == 4
    assert len(bg_map.shape) == 4
    assert obj_map.shape[1] == color_channels + 1
    assert bg_map.shape[1] == color_channels
    
    return obj_map[:, :color_channels, :, :] + bg_map[:, :color_channels, :, :] * (1 - obj_map[:, color_channels:(color_channels+1), :, :])

def composite_batch(batch, args, to_predictors,  EPS=1e-6, *, forced_shading_mask=None, override_image_for_shading_computation=None,
                    return_shading_mask=False):
    bs = batch['pixel_values'].shape[0]
    assert bs == 1
    image_logs = [{} for _ in range(bs)]

    for k, v in batch.items():
        if hasattr(v, 'to'):
            batch[k] = v.to(device=args.eval.device)

    obj_mask = batch["mask"]
    obj_batch = {
        "depth": batch["depth"],
        "footprint_depth": batch["footprint_depth"] if "footprint_depth" in batch else None,
        "normal": batch["normal"],
        "diffuse": batch["diffuse"] if not args.eval.use_rgb_as_diffuse else batch['src_obj'],
        "mask": batch["mask"],
    }
    if args.eval.use_rgb_as_diffuse:
        obj_batch["diffuse"][:, :3, :, :] = batch["src_obj"][:, :3, :, :]
    
    dst_batch = {
        "pixel_values": batch["pixel_values"],
        "input_ids": batch["input_ids"],
        "caption": batch["caption"],
        "bg_normal": batch["bg_normal"] if "bg_normal" in batch else None,
        "bg_diffuse": batch["bg_diffuse"] if "bg_diffuse" in batch else None,
        # TODO: put those back if needed
        # "bg_roughness": batch["bg_roughness"] if "bg_roughness" in batch else None,
        # "bg_metallic": batch["bg_metallic"] if "bg_metallic" in batch else None,
    }
    if 'bg_roughness' in batch:
        dst_batch['bg_roughness'] = batch['bg_roughness']
        obj_batch['roughness'] = batch['roughness']
    if 'bg_metallic' in batch:
        dst_batch['bg_metallic'] = batch['bg_metallic']
        obj_batch['metallic'] = batch['metallic']

    dst_batch = to_predictors(dst_batch)
    # TODO: cleaner
    if override_image_for_shading_computation is not None:
        dst_batch["controlnet_inputs"]["shading"] = compute_shading(override_image_for_shading_computation, alpha_blend(obj_batch['diffuse'], dst_batch['controlnet_inputs']['diffuse']))

    dst_normal = dst_batch["controlnet_inputs"]["normal"].clone()
    # ezexr.imwrite(os.path.join(f'/scratch/fredfc/{batch["name"][0]}_normal1.exr'), sdi_utils.tensor_to_numpy(dst_normal))
    dst_diffuse = dst_batch["controlnet_inputs"]["diffuse"].clone()
    dst_diffuse += EPS
    dst_shading = dst_batch["controlnet_inputs"]["shading"].clone()

    bg_shading = dst_shading.clone()

    src_obj_list = sdi_utils.tensor_to_pil_list(batch["src_obj"])
    dst_comp_list = sdi_utils.tensor_to_pil_list(batch["comp"])
    bg_list = sdi_utils.tensor_to_pil_list(dst_batch["pixel_values"] * 0.5 + 0.5)
    bg_image_for_balance = dst_batch["pixel_values"].clone() * 0.5 + 0.5
    for sample_idx in range(bs):
        image_logs[sample_idx].update({"src_obj": src_obj_list[sample_idx],
                                       "src_mask": obj_mask[sample_idx],
                                       "dst_bg": bg_list[sample_idx],
                                       "dst_comp": dst_comp_list[sample_idx],
                                       "name": batch["name"][sample_idx]} if "name" in batch else False)

    # TODO: match disparity
    # obj_batch['depth'] = match_depth_from_footprint(dst_depth, obj_batch['depth'], obj_batch['footprint_depth'])
    
    if 'depth' in dst_batch["controlnet_inputs"]:
        dst_depth = dst_batch["controlnet_inputs"]["depth"].clone()
        if args.eval.predictor_names[0] == 'depth_anything_relative':
            dst_batch["controlnet_inputs"]['depth'] = match_depth_from_footprint(dst_depth, obj_batch['depth'], obj_batch['footprint_depth'], adjust_background=True, match_disparity=True)
        elif args.eval.predictor_names[0] == 'zoedepth':
            obj_batch['depth'] = match_depth_from_footprint(dst_depth, obj_batch['depth'], obj_batch['footprint_depth'], adjust_background=False, match_disparity=False)
        else:
            raise NotImplementedError

    comp_batch = {}
    for k, v in dst_batch["controlnet_inputs"].items():
        if k == 'depth':
            dst_pil_list = sdi_utils.tensor_to_pil_list(v, [v.min(), v.max()])
        else:
            dst_pil_list = sdi_utils.tensor_to_pil_list(v)
        for sample_idx in range(len(dst_pil_list)):
            image_logs[sample_idx].update({f"dst_{k}": dst_pil_list[sample_idx]})

        tmp_mask = obj_mask.expand_as(v)
        if k == 'shading':
            shading_maskout_mode = args.eval.shading_maskout_mode
            shading_maskout_bbox_dilation = args.eval.shading_maskout_bbox_dilation
            shading_maskout_bbox_depth_range = args.eval.shading_maskout_bbox_depth_range

            shading_mask_binarization_threshold = 0.1 # used to be 0.9
            if forced_shading_mask is not None and args.eval.forced_shading_mask_mode == 'override':
                v[:, :, forced_shading_mask] = args.aug.fill_value
            else:
                for b in range(bs):
                    if shading_maskout_mode == 'None':
                        pass

                    elif 'BBox' in shading_maskout_mode:
                        bbox = masks_to_boxes(obj_mask.squeeze(dim=1)).int()
                        _, _, h, w = v.shape

                        x1, y1, x2, y2 = bbox[b]
                        x1 = max(x1 - shading_maskout_bbox_dilation, 0)
                        y1 = max(y1 - shading_maskout_bbox_dilation, 0)
                        x2 = min(x2 + shading_maskout_bbox_dilation, w)
                        y2 = min(y2 + shading_maskout_bbox_dilation, h)

                        v[b, :, y1:y2, x1:x2] = args.aug.fill_value

                        if shading_maskout_mode == 'BBoxWithDepth':
                            avg_obj_depth = obj_batch["depth"][b, :, :, :][obj_mask[b, :, :, :] > shading_mask_binarization_threshold].mean()
                            bg_depth = dst_batch["controlnet_inputs"]["depth"][b, :, :, :]
                            avg_obj_depth = avg_obj_depth.expand_as(bg_depth)
                            out_of_depth_range_mask = torch.abs(bg_depth - avg_obj_depth) > shading_maskout_bbox_depth_range

                            if args.eval.shading_maskout_pc_above_cropping_type == 'abovebbox':
                                out_of_depth_range_mask[:, :y1, :] = True
                            elif args.eval.shading_maskout_pc_above_cropping_type == 'argmin':
                                obj_mask_argmax = torch.argmax(obj_mask[b, :, :, :], dim=1, keepdim=True)
                                for j in range(w):
                                    out_of_depth_range_mask[b, :, :obj_mask_argmax[0, 0, j], j] = True

                            out_of_depth_range_mask = torch.logical_and(out_of_depth_range_mask, ~(obj_mask[b, :, :, :].bool()))
                            out_of_depth_range_mask = out_of_depth_range_mask.expand_as(bg_shading[b, :, :, :])
                            v[b, out_of_depth_range_mask] = bg_shading[b, out_of_depth_range_mask]

                    elif shading_maskout_mode == 'PointCloud':
                        bg_depth = dst_batch["controlnet_inputs"]["depth"][b, :, :, :]
                        bg_point_cloud = sdi_utils.depth_map_to_point_cloud(bg_depth, fov=args.eval.point_cloud_fov).permute(1, 2, 0).reshape(-1, 3)
                        obj_depth = comp_batch['depth'][b, :, :, :]
                        obj_point_cloud = sdi_utils.depth_map_to_point_cloud(obj_depth, fov=args.eval.point_cloud_fov)
                        obj_point_cloud = obj_point_cloud.permute(1, 2, 0)[(obj_mask[b, 0, :, :] > shading_mask_binarization_threshold), :]
                        dists = sdi_utils.compute_distance_bgpc_objpc(bg_point_cloud.cpu().numpy(), obj_point_cloud.cpu().numpy())
                        dists = dists.reshape(bg_depth.shape[1], bg_depth.shape[2], 1)
                        dists = torch.from_numpy(dists).to(args.eval.device).permute(2, 0, 1)
                        pc_crop_mask = None
                        if args.eval.shading_maskout_pc_type == 'absolute':
                            pc_crop_mask = dists < args.eval.shading_maskout_pc_range
                        elif args.eval.shading_maskout_pc_type == 'relative':
                            object_height = obj_point_cloud[:, 0].max() - obj_point_cloud[:, 0].min()
                            pc_crop_mask = dists < object_height * args.eval.shading_maskout_pc_range_relative
                        else:
                            raise NotImplementedError

                        _, _, h, w = v.shape
                        if args.eval.shading_maskout_pc_above_cropping_type == 'abovebbox':
                            bbox = masks_to_boxes(obj_mask.squeeze(dim=1)).int()
                            x1, y1, x2, y2 = bbox[b]
                            pc_crop_mask[:, :y1, :] = False
                        elif args.eval.shading_maskout_pc_above_cropping_type == 'argmin':
                            obj_mask_argmax = torch.argmax(obj_mask[b, :, :, :], dim=1, keepdim=True)
                            for j in range(w):
                                if obj_mask_argmax[0, 0, j] > 0:
                                    pc_crop_mask[:, :obj_mask_argmax[0, 0, j], j] = False

                        pc_crop_mask = pc_crop_mask.expand_as(bg_shading[b, :, :, :])
                        obj_mask_b = obj_mask > shading_mask_binarization_threshold
                        crop_mask = torch.logical_or(pc_crop_mask, obj_mask_b[b, :, :, :])
                        v[b, crop_mask] = args.aug.fill_value

                    elif shading_maskout_mode == 'Cone':
                        bg_depth = dst_batch["controlnet_inputs"]["depth"][b, :, :, :]
                        bg_point_cloud = sdi_utils.depth_map_to_point_cloud(bg_depth, fov=args.eval.point_cloud_fov).permute(1, 2, 0).reshape(-1, 3)
                        obj_depth = comp_batch['depth'][b, :, :, :]
                        obj_point_cloud = sdi_utils.depth_map_to_point_cloud(obj_depth, fov=args.eval.point_cloud_fov)
                        obj_point_cloud = obj_point_cloud.permute(1, 2, 0)[(obj_mask[b, 0, :, :] > shading_mask_binarization_threshold), :]

                        bg_point_cloud = bg_point_cloud.cpu().numpy()
                        bg_cloud = o3d.geometry.PointCloud()
                        bg_cloud.points = o3d.utility.Vector3dVector(bg_point_cloud)
                        bg_cloud_tree = o3d.geometry.KDTreeFlann(bg_cloud)

                        obj_point_cloud = obj_point_cloud.cpu().numpy()
                        obj_cloud = o3d.geometry.PointCloud()
                        obj_cloud.points = o3d.utility.Vector3dVector(obj_point_cloud)
                        obj_cloud_tree = o3d.geometry.KDTreeFlann(obj_cloud)

                        cone_crop_mask = np.zeros([bg_point_cloud.shape[0], 1], dtype=bool)
                        down_vector = np.array([-1, 0, 0])
                        for i in range(bg_point_cloud.shape[0]):
                            query_point = bg_point_cloud[i, :][:, None]
                            [k_, idx, _] = obj_cloud_tree.search_knn_vector_3d(query_point, 1)
                            if k_ > 0:
                                current_point = bg_point_cloud[i, :]
                                obj_closest_point = obj_point_cloud[idx[0], :]
                                cone_vector = obj_closest_point - current_point
                                distance = np.linalg.norm(cone_vector, ord=2)
                                if distance > args.eval.shading_maskout_cone_radius:
                                    continue
                                cone_vector_unit = cone_vector / (distance + EPS)
                                angle = np.dot(cone_vector_unit, down_vector)
                                if np.isnan(angle).any():
                                    continue
                                angle = np.arccos(angle) * 180 / np.pi
                                if angle < args.eval.shading_maskout_cone_angle:
                                    cone_crop_mask[i, :] = True

                        cone_crop_mask = cone_crop_mask.reshape(bg_depth.shape[1], bg_depth.shape[2], 1)
                        cone_crop_mask = torch.from_numpy(cone_crop_mask).to(args.eval.device).permute(2, 0, 1)

                        obj_mask_b = obj_mask > shading_mask_binarization_threshold
                        crop_mask = torch.logical_or(cone_crop_mask, obj_mask_b[b, :, :, :])

                        v[b, crop_mask.expand_as(v[b, :, :, :])] = args.aug.fill_value

                    if args.eval.shading_maskout_obj_dilation > 0:
                        ks = args.eval.shading_maskout_obj_dilation
                        obj_mask_dilated = cv2.dilate(obj_mask[b, 0, :, :].cpu().numpy(), np.ones((ks, ks)), iterations=1)
                        obj_mask_dilated = torch.from_numpy(obj_mask_dilated).to(args.eval.device)
                        obj_mask_dilated = obj_mask_dilated.expand_as(v[b, :, :, :])
                        v[b, obj_mask_dilated > shading_mask_binarization_threshold] = args.aug.fill_value

            if forced_shading_mask is not None and args.eval.forced_shading_mask_mode == 'combine':
                v[:, :, forced_shading_mask] = args.aug.fill_value

            v = torch.clamp(v, max=1e3)

        elif k == 'depth':
            obj_area = obj_batch[k]
            bg_area = v

            assert obj_area.shape[1] == 2
            
            v[...] = alpha_blend(obj_batch[k], v, color_channels=1)
            v[...] = torch.clamp(v, min=0, max=20) # TODO: still needed?
        elif k in ['metallic', 'roughness']:
            assert obj_batch[k].shape[1] == 2, f"obj_batch[{k}].shape[1] = {obj_batch[k].shape[1]}"
            v[...] = alpha_blend(obj_batch[k], v, color_channels=1)
        elif k == 'masked_bg':
            pass
        elif k == 'mask':
            pass
        else:
            assert obj_batch[k].shape[1] == 4, f"obj_batch[{k}].shape[1] = {obj_batch[k].shape[1]}"
            
            v = alpha_blend(obj_batch[k].clone(), v.clone())

        comp_batch[k] = v

    controlnet_inputs = []
    for k, v in comp_batch.items():
        if k == 'mask':
            v = torch.ones_like(v)
            shading = comp_batch['shading']
            v[shading[:, 0:1, :, :] == args.aug.fill_value] = args.aug.fill_value
            v = v.float()
            comp_batch[k] = v
            shading_mask = v
        elif k == 'masked_bg':
            shading = comp_batch['shading']
            v[shading[:, :, :, :] == args.aug.fill_value] = args.aug.fill_value
            comp_batch[k] = v

        controlnet_inputs.append(v)

        if k == "depth":
            v_pil_list = sdi_utils.tensor_to_pil_list(v, [v.min(), v.max()])
        else:
            v_pil_list = sdi_utils.tensor_to_pil_list(v)

        for sample_idx in range(len(dst_pil_list)):
            image_logs[sample_idx].update({f"comp_{k}": v_pil_list[sample_idx]})

    if args.conditioning_channels == 12:
        depth_valid_mask = torch.ones_like(obj_mask)
        controlnet_inputs.append(depth_valid_mask)
    conditioning = torch.cat(controlnet_inputs, dim=1)

    # os.makedirs('/scratch/fredfc/depth_outputs', exist_ok=True)
    # ezexr.imwrite(os.path.join(f'/scratch/fredfc/depth_outputs/{batch["name"][0]}_depth.exr'), sdi_utils.tensor_to_numpy(comp_batch["depth"], clip=False))
    if return_shading_mask:
        return conditioning, dst_batch, obj_mask, bg_image_for_balance, image_logs, shading_mask
    else:
        return conditioning, dst_batch, obj_mask, bg_image_for_balance, image_logs


def run_inference(conditioning, dst_batch, pipeline, args, bg_image_for_balance, obj_mask, *, 
                controlnet_conditioning_scale=1.0, guess_mode=False, guidance_scale=0.0,
                latent_mask_weight=[0.005, 0.005],
                seed=0, # could have used other seed, such as 469
                num_inference_steps=20,
                shadow_guidance_kernel_size = 16 * 2 + 1,
                relighting_guidance=None, relighting_guidance_start_provided=False, shadow_guidances=None, shadow_composites=None, relighting_guidance_scale=0.0, relighting_guidance_mask=None,
                float_32_autocast=False,
                sd_edit_cleanup_percentage=0.0, conditioning_sd_edit=None, output_image_logs=False,
                ):
    bs = conditioning.shape[0]
    validation_prompt = dst_batch["caption"]

    if shadow_guidances is not None and relighting_guidance is not None:
        assert len(shadow_guidances.shape) == 4
        assert shadow_guidances.shape[0] == 2
            
        shadowed_latents = pipeline.vae.encode(shadow_composites.to(pipeline.vae.dtype) * 2 - 1).latent_dist.sample() * pipeline.vae.config.scaling_factor
        
        if args.eval.dilate_shadow_backbone == 'kornia':
            dilation_kernel_3x3 = torch.tensor([[0, 1, 0],
                                                [1, 1, 1],
                                                [0, 1, 0]], device=args.eval.device).float()
            
            latent_shadow_guidances = F.interpolate(shadow_guidances, size=(64, 64), mode='bilinear')
            obj_mask_downsampled = F.interpolate(obj_mask, size=(64, 64), mode='bilinear')

            epsilon_shadow_mask = 0.05
            
            binary_shadow_mask = (latent_shadow_guidances > epsilon_shadow_mask).float()
            binary_obj_mask = (obj_mask_downsampled > epsilon_shadow_mask).float()

            dilated_latent_shadow_mask = kornia.morphology.dilation(binary_shadow_mask, dilation_kernel_3x3)
            eroded_latent_shadow_mask = kornia.morphology.erosion(binary_shadow_mask, dilation_kernel_3x3)
            edge_latent_mask = dilated_latent_shadow_mask - eroded_latent_shadow_mask
            
            final_latent_mask = eroded_latent_shadow_mask + edge_latent_mask * args.eval.latent_mask_edge_boost
            final_latent_mask *= 1 - binary_obj_mask
            weighted_latent_mask = torch.cat([final_latent_mask[0:1] * latent_mask_weight[0], final_latent_mask[1:2] * latent_mask_weight[1]])
            # final_latent_mask = ((kornia.morphology.dilation(latent_shadow_guidances, dilation_kernel_7x7) * (1 - obj_mask_downsampled)) > 0.05).float()          
        else:
            raise NotImplementedError
    
    if seed is None:
        generator = None
    else:
        generator = torch.Generator(device=args.eval.device).manual_seed(seed)

    with torch.autocast("cuda", dtype=torch.float32 if float_32_autocast else None):
        noise_latents = randn_tensor([1, pipeline.unet.config.in_channels, args.resolution // pipeline.vae_scale_factor, args.resolution // pipeline.vae_scale_factor],
                                        generator=generator, device=pipeline.device, dtype=pipeline.dtype)
            
        def latent_shadow_blend_callback(pipeline, i, t, callback_kwargs):
            if i == num_inference_steps - 1:
                # we don't want to add shadows without at the last step, since it can't be refined by the denoiser
                return {
                    'latents': callback_kwargs['latents'][0:1, ...],
                }
            
            random_noise = randn_tensor([1, pipeline.unet.config.in_channels, args.resolution // pipeline.vae_scale_factor, args.resolution // pipeline.vae_scale_factor],
                                        generator=generator, device=pipeline.device, dtype=pipeline.dtype) # TODO: make it different always
            random_noise = torch.cat([random_noise, random_noise], dim=0)
            noised_hard_shadow_latent = pipeline.scheduler.add_noise(shadowed_latents, random_noise, t) # TODO: shift t

            # sdi_utils.tensor_to_pil(pipeline.vae.decode(callback_kwargs['pred_original_sample'][:1] / pipeline.vae.config.scaling_factor, return_dict=False)[0] * 0.5 + 0.5).save(os.path.join(results_dir, f'decoded_{batch_idx}', 'original_pred', f'step_{t:05}_original_image.png'))
            # sdi_utils.tensor_to_pil(pipeline.vae.decode(callback_kwargs['latents'][:1] / pipeline.vae.config.scaling_factor, return_dict=False)[0] * 0.5 + 0.5).save(os.path.join(results_dir, f'decoded_{batch_idx}', 'current', f'step_{t:05}_current_image.png'))
            

            output_latents = weighted_latent_mask * noised_hard_shadow_latent + (1 - weighted_latent_mask) * callback_kwargs['latents']
            if i == num_inference_steps - 1:
                output_latents = output_latents[0:1, ...]
            return {
                'latents': output_latents,
            }

        if shadow_guidances is not None and relighting_guidance is not None:
            callback_on_step_end = latent_shadow_blend_callback
            callback_on_step_end_tensor_inputs = ['latents', 'pred_original_sample']
        else:
            callback_on_step_end = None
            callback_on_step_end_tensor_inputs = None
            
        out = pipeline(
            validation_prompt, conditioning, num_inference_steps=num_inference_steps, generator=generator, latents=noise_latents, guidance_scale=guidance_scale, guess_mode=guess_mode,
            output_type='pt', class_labels=dst_batch['dominant_light_center'],
            callback_on_step_end=callback_on_step_end, callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            relighting_guidance=relighting_guidance, relighting_guidance_start_provided=relighting_guidance_start_provided, relighting_guidance_scale=relighting_guidance_scale, relighting_guidance_mask=relighting_guidance_mask,
            relighting_guidance_rescale=args.eval.relighting_guidance_rescale,
        )
        if sd_edit_cleanup_percentage > 0.0:
            assert conditioning_sd_edit is not None
            out = pipeline(
                validation_prompt, conditioning_sd_edit, num_inference_steps=num_inference_steps, generator=generator, latents=noise_latents, guidance_scale=guidance_scale, guess_mode=guess_mode,
                output_type='pt',
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                sd_edit_cleanup_percentage=sd_edit_cleanup_percentage,
                sd_edit_image=(out.images * 2 - 1)
            )

    current_images = torch.nan_to_num(out.images, nan=0, posinf=0, neginf=0)

    if output_image_logs:
        image_logs = {
            'final_latent_mask': final_latent_mask,
            'weighted_latent_mask': weighted_latent_mask,
        }
        return current_images[0], image_logs
    return current_images[0]


def _render_numpy_shadow(scene_name, shadow_map_name, light_position_blender, dataset_dir, results_dir, *, background_mesh_dir, objects_dir, light_distance, light_radius=0.0, hide_object=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
    # Set up the environment variables for the subprocess
    debug_mode = True # warning: turning this on usually slows quite a bit the transfer time.

    new_env = os.environ.copy()
    new_env['DEBUG'] = '1' if debug_mode else '0'
    new_env['SCENE_NAME'] = scene_name
    new_env['SHADOW_MAP_NAME'] = shadow_map_name
    new_env['DATASET_DIR'] = dataset_dir
    new_env['OUTPUT_DIR'] = os.path.join(results_dir)
    new_env['BACKGROUND_MESH_DIR'] = background_mesh_dir
    new_env['OBJECTS_DIR'] = objects_dir
    new_env['LIGHT_X'] = str(light_position_blender[0] * light_distance)
    new_env['LIGHT_Y'] = str(light_position_blender[1] * light_distance)
    new_env['LIGHT_Z'] = str(light_position_blender[2] * light_distance)
    new_env['LIGHT_RADIUS'] = str(light_radius)
    new_env['HIDE_OBJECT'] = str(hide_object)
    
    # Run the Blender process to generate the shadow map
    subprocess.run(['blender','--background', '--python', 'src/blender/shadow_map_generation.py'], env=new_env)

    # Check if the shadow map was generated
    shadow_map_path = os.path.join(results_dir, f'ShadowMap_{shadow_map_name}_0001.exr')
    if not os.path.exists(shadow_map_path):
        raise FileNotFoundError(f"Shadow map not found at {shadow_map_path}")

    # Read the generated shadow map
    output = ezexr.imread(shadow_map_path)

    # # Remove the shadow map file
    if not debug_mode:
        os.remove(shadow_map_path)

    return output

def render_numpy_shadow(scene_name, shadow_map_name, light_position_blender, dataset_dir, results_dir, *, background_mesh_dir, objects_dir, light_distance, light_radius=0.0, render_engine='BLENDER_EVEE', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
    if render_engine == 'BLENDER_EVEE':
        render_with_object = _render_numpy_shadow(scene_name, f'{shadow_map_name}_a', light_position_blender, dataset_dir, results_dir, light_distance=light_distance, background_mesh_dir=background_mesh_dir, objects_dir=objects_dir, light_radius=light_radius, hide_object=False, stdout=stdout, stderr=stderr)[:, :, 0:1]
        render_without_object = _render_numpy_shadow(scene_name, f'{shadow_map_name}_b', light_position_blender, dataset_dir, results_dir, light_distance=light_distance, background_mesh_dir=background_mesh_dir, objects_dir=objects_dir, light_radius=light_radius, hide_object=True, stdout=stdout, stderr=stderr)[:, :, 0:1]
        shadow_gain = render_with_object / (render_without_object + 1e-6)
        shadow_alpha = 1 - shadow_gain
    elif render_engine == 'CYCLES':
        shadow_alpha = _render_numpy_shadow(scene_name, shadow_map_name, light_position_blender, dataset_dir, results_dir, background_mesh_dir=background_mesh_dir, objects_dir=objects_dir, light_distance=light_distance, light_radius=light_radius, hide_object=False, stdout=stdout, stderr=stderr)[:, :, 0:1]

    return shadow_alpha

def get_outline_visualization(shadowed_bg: torch.Tensor, obj_mask: torch.Tensor):
    # Get the outline of the object by dilating+eroding the mask and making it white in the output, using kornia
    dilation_kernel = torch.tensor([[0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 1, 1, 1, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 0],
                                    [1, 1, 1, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 1, 1, 0],
                                    [0, 0, 1, 1, 1, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0]]).float().to(obj_mask.device)
    eroded_obj_mask = kornia.morphology.erosion(obj_mask, dilation_kernel)
    # dilated_obj_mask = kornia.morphology.dilation(obj_mask, dilation_kernel)
    outline = obj_mask - eroded_obj_mask
    outline = outline.expand_as(shadowed_bg)
    out = shadowed_bg * (1 - outline) + outline
    return out