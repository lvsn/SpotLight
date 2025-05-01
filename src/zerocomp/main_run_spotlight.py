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
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
)
from pipeline_controlnet_spotlight_zc import ControlNetSpotLightZeroCompPipeline
from controlnet_input_handle import ToControlNetInput, ToPredictors
import sdi_utils
import hydra
from tempfile import TemporaryDirectory
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
    torch.set_grad_enabled(False)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"{args.eval.results_dir}/{current_time}_control_dataset"
    if args.eval.run_name != '':
        results_dir += f"_{args.eval.run_name}"
    os.makedirs(results_dir, exist_ok=True)

    OmegaConf.save(config=args, f=os.path.join(results_dir, "config.yaml"))

    sdi_utils.seed_all(args.seed)

    # light_estimator = compositor.initialize_light_estimator(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    vae, unet, text_encoder, controlnet = compositor.load_models(args)

    to_controlnet_input = ToControlNetInput(
        device=args.eval.device,
        feed_empty_prompt=args.feed_empty_prompt,
        tokenizer=tokenizer,
        for_sdxl=False
    )
# 
    to_predictors = ToPredictors(args.eval.device,
                                 args.scale_destination_composite_to_minus_one_to_one,
                                 conditioning_maps=args.conditioning_maps,
                                 predictor_names=args.eval.predictor_names,)

    val_dataloader = compositor.create_dataloader(args, to_controlnet_input=to_controlnet_input, start_batch=args.eval.start_batch) # TODO: end batch too?

    weight_dtype = torch.float32
    if args.eval.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif args.eval.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(args.eval.device, dtype=weight_dtype)
    # vae.to(args.eval.device, dtype=torch.float32)
    unet.to(args.eval.device, dtype=weight_dtype)
    text_encoder.to(args.eval.device, dtype=weight_dtype)
    controlnet.to(args.eval.device, dtype=weight_dtype)

    pipeline = ControlNetSpotLightZeroCompPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=None
    )
    pipeline.scheduler = DDIMScheduler.from_config(
        pipeline.scheduler.config,
        **args.val_scheduler.kwargs,
    )
    pipeline = pipeline.to(args.eval.device, dtype=weight_dtype)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    
    if args.eval.relight_bg:
        controlnet_inv = ControlNetModel.from_pretrained(args.eval.controlnet_model_name_or_path_inv, subfolder="controlnet",
                                                        light_parametrization=args.light_parametrization)
        pipeline_inv = ControlNetSpotLightZeroCompPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet_inv,
            safety_checker=None,
            revision=None
        )
        pipeline_inv.scheduler = DDIMScheduler.from_config(
            pipeline_inv.scheduler.config,
            **args.val_scheduler.kwargs,
        )
        pipeline_inv = pipeline_inv.to(args.eval.device, dtype=weight_dtype)
        pipeline_inv.set_progress_bar_config(disable=True)

        if args.enable_xformers_memory_efficient_attention:
            pipeline_inv.enable_xformers_memory_efficient_attention()

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
                    # Note: max_pool2d should be changed for a proper dilation
                    dilated_obj_mask = F.max_pool2d(batch['mask'], kernel_size=args.eval.obj_mask_dilation_shadowcomp * 2 + 1, stride=1, padding=args.eval.obj_mask_dilation_shadowcomp)
                else:
                    dilated_obj_mask = batch['mask']
                dilated_obj_mask = dilated_obj_mask.to(args.eval.device)

                relighting_guidance_mask = F.interpolate(batch['mask'], size=(64, 64), mode='bilinear', antialias=True).to(args.eval.device)
 
                image_logs = None
                if args.dataset_name != 'real_world':
                    conditioning_zerocomp, dst_batch, obj_mask, bg_image_for_balance, image_logs, shading_mask_zero_comp = compositor.composite_batch(
                        batch=batch,
                        args=args, 
                        to_predictors=to_predictors,
                        return_shading_mask=True,
                    )
                    if args.eval.negative_sample_type == 'negative_shading_mask':
                        negative_shading_mask = shading_mask_zero_comp
                        negative_image_logs = image_logs

                if image_logs is not None and args.eval.output_zero_comp_intrinsics:
                    for k, image in image_logs[0].items():
                        os.makedirs(os.path.join(results_dir, scene_name, 'intrinsics'), exist_ok=True)
                        intrinsic_png_path = os.path.join(results_dir, scene_name, 'intrinsics', f"{k}.png")
                        if type(image) == torch.Tensor:
                            sdi_utils.tensor_to_pil_list(image)[0].save(intrinsic_png_path)
                        elif type(image) == PIL.Image.Image:
                            image.save(intrinsic_png_path)

                if args.eval.negative_sample_type == 'zerocomp' or args.eval.render_zerocomp_only:
                    conditioning_zerocomp, dst_batch, obj_mask, bg_image_for_balance, image_logs, shading_mask_zero_comp = compositor.composite_batch(
                        batch=batch,
                        args=args, 
                        to_predictors=to_predictors,
                        return_shading_mask=True,
                    )
                    negative_shading_mask = shading_mask_zero_comp
                    negative_image_logs = image_logs
                    conditioning_negative = conditioning_zerocomp
                    
                    pred_zerocomp = compositor.run_inference(
                        conditioning=conditioning_zerocomp, 
                        dst_batch=dst_batch, 
                        pipeline=pipeline, 
                        args=args,
                        bg_image_for_balance=bg_image_for_balance, 
                        obj_mask=obj_mask,
                    )
                    pred_zerocomp_np = sdi_utils.tensor_to_numpy(torch.nan_to_num(pred_zerocomp, nan=0.0, posinf=0.0, neginf=0.0))

                    os.makedirs(os.path.join(results_dir, scene_name, 'default'), exist_ok=True)
                    img_disp = (pred_zerocomp_np * 255).astype(np.uint8)
                    img_disp = Image.fromarray(img_disp)
                    img_disp.save(os.path.join(results_dir, scene_name, 'default', f"zerocomp.png"))
                    
                    if args.eval.render_zerocomp_only:
                        continue

                if args.eval.generate_images_for_post_compositing:
                    conditioning_no_shadow, dst_batch, obj_mask, bg_image_for_balance, _ = compositor.composite_batch(
                        batch=batch,
                        args=args, 
                        to_predictors=to_predictors,
                        forced_shading_mask=(dilated_obj_mask[0, 0] != 0.0),
                        return_shading_mask=False,
                    )
                    pred_no_shadow = compositor.run_inference(
                        conditioning=conditioning_no_shadow, 
                        dst_batch=dst_batch, 
                        pipeline=pipeline, 
                        args=args,
                        bg_image_for_balance=bg_image_for_balance, 
                        obj_mask=obj_mask,
                    )
                    os.makedirs(os.path.join(results_dir, scene_name, 'post_comp'), exist_ok=True)
                    img_disp = (sdi_utils.tensor_to_numpy(pred_no_shadow) * 255).astype(np.uint8)
                    img_disp = Image.fromarray(img_disp)
                    img_disp.save(os.path.join(results_dir, scene_name, 'post_comp', f"pred_no_shadow.png"))
                    
                    mask_disp = (np.repeat(sdi_utils.tensor_to_numpy(batch['mask']), 3, 2) * 255).astype(np.uint8)
                    mask_disp = Image.fromarray(mask_disp)
                    mask_disp.save(os.path.join(results_dir, scene_name, 'post_comp', f"mask.png"))

                if args.eval.negative_sample_type == 'no_shadow':
                    conditioning_negative, _, _, _, negative_image_logs, negative_shading_mask = compositor.composite_batch(
                        batch=batch,
                        args=args,
                        to_predictors=to_predictors,
                        forced_shading_mask=(dilated_obj_mask[0, 0] != 0.0),
                        return_shading_mask=True
                    )

                all_frames = []
                shadow_map_idx = 0
                while os.path.exists(os.path.join(args.shadows_dir, scene_name, f'shadow_positive_{shadow_map_idx:04d}.png')):
                    all_frames.append({
                        'name': f'frame_{shadow_map_idx}',
                        'shadow_map_idx': f'{shadow_map_idx:04d}'
                    })
                    shadow_map_idx += 1
                if os.path.exists(os.path.join(args.shadows_dir, scene_name, f'{scene_name}_shadow_positive_gt_dir.png')):
                    all_frames.append({
                        'name': f'gt_dir',
                        'shadow_map_idx': f'gt_dir'
                    })

                for frame_idx, frame_dict in enumerate(all_frames):
                    # defaults
                    frame_name = frame_dict['name']

                    if args.eval.render_only_gt_dir and frame_name != 'gt_dir':
                        continue

                    shadow_custom = batch[f'shadow_positive_{frame_dict["shadow_map_idx"]}'] * (1 - batch['mask'])

                    print('composite_batch... ')
                    
                    # pretty much same as: args.eval.shading_maskout_obj_dilation // 2 - 1
                    
                    shadow_opposite = batch[f'shadow_negative_{frame_dict["shadow_map_idx"]}'] * (1 - batch['mask'])
                    conditioning_negative, _, _, _, negative_image_logs, negative_shading_mask = compositor.composite_batch(
                        batch=batch, 
                        args=args, 
                        to_predictors=to_predictors, 
                        forced_shading_mask=(shadow_opposite[0,0] > 0.1) | (dilated_obj_mask[0, 0] != 0.0),
                        return_shading_mask=True
                    )
                
                    if args.eval.use_coarse_shadow_as_is:
                        assert args.eval.relighting_guidance_scale == 1.0, 'negative sample not supported for coarse shadow (could easily be added)'
                        background_coarse = batch['pixel_values'] * (1 - args.eval.shadow_opacity * shadow_custom)
                        conditioning_custom, dst_batch, obj_mask, bg_image_for_balance, positive_image_logs, positive_shading_mask = compositor.composite_batch(
                            batch=batch, 
                            args=args, 
                            to_predictors=to_predictors, 
                            forced_shading_mask=(dilated_obj_mask[0, 0] != 0.0),
                            override_image_for_shading_computation=background_coarse,
                            return_shading_mask=True
                        )
                    else:
                        conditioning_custom, dst_batch, obj_mask, bg_image_for_balance, positive_image_logs, positive_shading_mask = compositor.composite_batch(
                            batch=batch, 
                            args=args, 
                            to_predictors=to_predictors, 
                            forced_shading_mask=(shadow_custom[0,0] > 0.1) | (dilated_obj_mask[0, 0] != 0.0),
                            return_shading_mask=True
                        )

                    # if args.dataset_name == 'real_world' and frame_idx == 0:
                    #     if args.eval.output_zero_comp_intrinsics:
                    #         for k, image in positive_image_logs[0].items():
                    #             os.makedirs(os.path.join(results_dir, 'intrinsics'), exist_ok=True)
                    #             intrinsic_png_path = os.path.join(results_dir, 'intrinsics', f"{scene_name}_{k}.png")
                    #             if type(image) == torch.Tensor:
                    #                 sdi_utils.tensor_to_pil_list(image)[0].save(intrinsic_png_path)
                    #             elif type(image) == PIL.Image.Image:
                    #                 image.save(intrinsic_png_path)
                    print('done')


                    print('run_inference... ')
                    if args.eval.negative_sample_type == 'zerocomp':
                        shadow_opposite = torch.zeros_like(shadow_custom)
                    elif args.eval.negative_sample_type == 'no_shadow':
                        shadow_opposite = torch.zeros_like(shadow_custom)
                        
                    shadow_guidances = torch.cat([shadow_custom, shadow_opposite])
                    background_coarse = batch['pixel_values'] * (1 - args.eval.shadow_opacity * torch.cat([shadow_custom, shadow_opposite]))
                    shadow_composites = compositor.alpha_blend(batch['diffuse'], background_coarse)

                    if args.eval.output_shadow_comp_intermediate_images:
                        # save positive and negative shading masks
                        os.makedirs(os.path.join(results_dir, scene_name, 'intermediate'), exist_ok=True)
                        Image.fromarray((np.repeat(sdi_utils.tensor_to_numpy(positive_shading_mask), 3, axis=2) * 255).astype(np.uint8)).save(os.path.join(results_dir, scene_name, 'intermediate', f"positive_shading_mask_{frame_name}.png"))
                        Image.fromarray((np.repeat(sdi_utils.tensor_to_numpy(negative_shading_mask), 3, axis=2) * 255).astype(np.uint8)).save(os.path.join(results_dir, scene_name, 'intermediate', f"negative_shading_mask_{frame_name}.png"))
                        negative_image_logs[0]['comp_shading'].save(os.path.join(results_dir, scene_name, 'intermediate', f"negative_shading_image_{frame_name}.png"))
                        positive_image_logs[0]['comp_shading'].save(os.path.join(results_dir, scene_name, 'intermediate', f"positive_shading_image_{frame_name}.png"))

                    if args.eval.sd_edit_cleanup_percentage > 0.0:
                        conditioning_sd_edit, _, _, _,_ = compositor.composite_batch(
                            batch=batch,
                            args=args, 
                            to_predictors=to_predictors, 
                            forced_shading_mask=torch.ones_like(shadow_custom[0,0], dtype=torch.bool),
                            return_shading_mask=False
                        )
                    else:
                        conditioning_sd_edit = None
                        
                    pred, pred_image_logs = compositor.run_inference(
                        conditioning=conditioning_custom,
                        dst_batch=dst_batch, 
                        pipeline=pipeline, 
                        args=args,            
                        bg_image_for_balance=bg_image_for_balance, 
                        obj_mask=obj_mask,
                        controlnet_conditioning_scale=args.eval.controlnet_conditioning_scale,
                        guess_mode=args.eval.guess_mode,
                        # fred stuff
                        relighting_guidance=[conditioning_custom, conditioning_negative] if args.eval.relighting_guidance_scale is not None else None,
                        shadow_guidances=shadow_guidances,
                        shadow_composites=shadow_composites,
                        guidance_scale=args.eval.guidance_scale,
                        relighting_guidance_mask=relighting_guidance_mask,
                        latent_mask_weight=[args.eval.latent_mask_weight, args.eval.latent_mask_weight],
                        relighting_guidance_scale=args.eval.relighting_guidance_scale,
                        sd_edit_cleanup_percentage=args.eval.sd_edit_cleanup_percentage,
                        conditioning_sd_edit=conditioning_sd_edit,
                        output_image_logs=True,
                    )

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

                    if args.eval.relight_bg:
                        print('relight bg')
                        override_image_for_shading_computation_bg_new_shadow = pred.unsqueeze(0)
                        forced_shading_mask_bg_new_shadow = ~((shadow_custom[0,0] > 0.1) | (dilated_obj_mask[0, 0] != 0.0))

                        
                        conditioning_bg_new_shadow, dst_batch, obj_mask, bg_image_for_balance, image_logs = compositor.composite_batch(
                            batch=batch, 
                            args=args, 
                            to_predictors=to_predictors,
                            override_image_for_shading_computation=override_image_for_shading_computation_bg_new_shadow,
                            forced_shading_mask=forced_shading_mask_bg_new_shadow,
                        )
                        if args.eval.output_zero_comp_intrinsics and frame_idx == 0:
                            for k, image in image_logs[0].items():
                                os.makedirs(os.path.join(results_dir, 'intrinsics_relight_bg'), exist_ok=True)
                                intrinsic_png_path = os.path.join(results_dir, 'intrinsics_relight_bg', f"{scene_name}_{k}.png")
                                if type(image) == torch.Tensor:
                                    sdi_utils.tensor_to_pil_list(image)[0].save(intrinsic_png_path)
                                elif type(image) == PIL.Image.Image:
                                    image.save(intrinsic_png_path)

                        pred_relight_bg_inv = compositor.run_inference(
                            conditioning=conditioning_bg_new_shadow,
                            dst_batch=dst_batch, 
                            pipeline=pipeline_inv, 
                            args=args, 
                            bg_image_for_balance=bg_image_for_balance, 
                            obj_mask=obj_mask,
                            guidance_scale=args.eval.guidance_scale,
                            controlnet_conditioning_scale=args.eval.controlnet_conditioning_scale,
                            guess_mode=args.eval.guess_mode,
                        )
                        os.makedirs(os.path.join(results_dir, scene_name, 'relight_bg'), exist_ok=True)
                        img_relight_bg_inv_disp = (sdi_utils.tensor_to_numpy(pred_relight_bg_inv) * 255).astype(np.uint8)
                        img_relight_bg_inv_disp = Image.fromarray(img_relight_bg_inv_disp)
                        img_relight_bg_inv_disp.save(os.path.join(results_dir, scene_name, 'relight_bg', f"inv_{frame_name}.png"))

                        if args.eval.relight_bg_no_obj_for_post_comp:

                            batch_mod = deepcopy(batch)
                            # set obj's alpha to 0 everywhere
                            batch_mod['depth'][...] = 0.0
                            batch_mod['mask'][...] = 0.0
                            batch_mod['normal'][...] = 0.0
                            batch_mod['diffuse'][...] = 0.0
                            if 'roughness' in batch_mod:
                                batch_mod['roughness'][...] = 0.0
                            if 'metallic' in batch_mod:
                                batch_mod['metallic'][...] = 0.0

                            forced_shading_mask_no_obj = torch.ones_like(forced_shading_mask_bg_new_shadow, dtype=torch.bool)
                            conditioning_bg_no_obj, dst_batch, obj_mask, bg_image_for_balance, image_logs = compositor.composite_batch(
                                batch=batch_mod,
                                args=args, 
                                to_predictors=to_predictors,
                                forced_shading_mask=forced_shading_mask_no_obj,
                            )

                            if args.eval.output_zero_comp_intrinsics and frame_idx == 0:
                                for k, image in image_logs[0].items():
                                    os.makedirs(os.path.join(results_dir, 'intrinsics_relight_bg_no_obj'), exist_ok=True)
                                    intrinsic_png_path = os.path.join(results_dir, 'intrinsics_relight_bg_no_obj', f"{scene_name}_{k}.png")
                                    if type(image) == torch.Tensor:
                                        sdi_utils.tensor_to_pil_list(image)[0].save(intrinsic_png_path)
                                    elif type(image) == PIL.Image.Image:
                                        image.save(intrinsic_png_path)

                            pred_relight_bg_inv_no_obj = compositor.run_inference(
                                conditioning=conditioning_bg_no_obj,
                                dst_batch=dst_batch, 
                                pipeline=pipeline_inv, 
                                args=args, 
                                bg_image_for_balance=bg_image_for_balance, 
                                obj_mask=obj_mask,
                                guidance_scale=args.eval.guidance_scale,
                                controlnet_conditioning_scale=args.eval.controlnet_conditioning_scale,
                                guess_mode=args.eval.guess_mode,
                            )
                            os.makedirs(os.path.join(results_dir, scene_name, 'relight_bg'), exist_ok=True)
                            img_relight_bg_inv_no_obj_disp = (sdi_utils.tensor_to_numpy(pred_relight_bg_inv_no_obj) * 255).astype(np.uint8)
                            img_relight_bg_inv_no_obj_disp = Image.fromarray(img_relight_bg_inv_no_obj_disp)
                            img_relight_bg_inv_no_obj_disp.save(os.path.join(results_dir, scene_name, 'relight_bg', f"inv_no_obj_{frame_name}.png"))

if __name__ == "__main__":
    main()
