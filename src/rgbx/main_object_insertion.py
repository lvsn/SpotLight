import os
import torch
import argparse
import numpy as np
from diffusers import DDIMScheduler
from rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from x2rgb_inpainting.pipeline_x2rgb_inpainting import StableDiffusionAOVDropoutPipeline as StableDiffusionAOVDropoutPipelineInpainting
import torchvision.transforms as T
from PIL import Image
import ezexr  # Ensure ezexr is installed: pip install ezexr
from kornia.morphology import dilation
import torch.nn.functional as F
import time
# Import the image loading functions
from x2rgb_inpainting.load_image import load_exr_image, load_ldr_image
import torch

def mask_to_bbox_mask(mask):
    """
    Converts a binary mask to a bounding box mask, where the bounding box region is set to 1s.

    Parameters:
    mask (torch.Tensor): A binary mask with non-zero values representing the object.

    Returns:
    torch.Tensor: A binary mask with the bounding box region set to 1s.
    """
    # Find non-zero elements in the mask
    non_zero_indices = torch.nonzero(mask[0] > 0, as_tuple=True)
    
    if len(non_zero_indices[0]) == 0:
        # No non-zero pixels found; return an empty mask of the same shape
        return torch.zeros_like(mask)
    
    # Get the bounding box coordinates
    y_min, y_max = torch.min(non_zero_indices[0]), torch.max(non_zero_indices[0])
    x_min, x_max = torch.min(non_zero_indices[1]), torch.max(non_zero_indices[1])

    # Create a new mask with the bounding box region set to 1s
    bbox_mask = torch.zeros_like(mask)
    bbox_mask[:, y_min:y_max+1, x_min:x_max+1] = 1

    return bbox_mask

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process EXR files from RGB to X to RGB.', fromfile_prefix_chars='@')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with EXR files.')
    parser.add_argument('--shadows_dir', type=str, required=True, help='Input shadows directory.')
    parser.add_argument('--output_dir', type=str, default='output_rgb', help='Output directory to save reconstructed RGB images and intrinsic composites.')
    parser.add_argument('--drop_aovs', nargs='*', default=[], help='List of intrinsic maps to drop (e.g., --drop_aovs irradiance).')
    parser.add_argument('--seed', type=int, default=469, help='Random seed.')
    parser.add_argument('--inference_steps', type=int, default=50, help='Number of inference steps.')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate.')
    parser.add_argument('--relighting_guidance_scale', type=float, default=3.0, help='SpotLight relight guidance scale.')
    parser.add_argument('--latent_mask_weight', type=float, default=0.05, help='Latent shadow composite weight.')
    parser.add_argument('--do_wb_x_rgb', action='store_true', help='do_wb_x_rgb.')
    parser.add_argument('--do_wb_rgb_x', action='store_true', help='do_wb_rgb_x.')
    parser.add_argument('--intensity_rescale', type=float, default=1.0, help='Intensity rescale.')
    parser.add_argument('--use_inpaint_model', action='store_true', help='Use inpainting model.')
    parser.add_argument('--use_rectangular_mask', action='store_true', help='Use rect mask for inpaint model.')
    parser.add_argument('--dilation_size', type=int, default=3, help='dilation kernel size (e.g. 3 for 3x3).')

    args = parser.parse_args()
    args.output_dir = args.output_dir + '/' + time.strftime("%Y-%m-%d-%H-%M")

    os.makedirs(args.output_dir, exist_ok=True)

    # Write configurations to a text file
    config_file_path = f"{args.output_dir}/configurations.txt"
    with open(config_file_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    print(f"Configurations saved to {config_file_path}")

    # List of all possible AOVs
    required_aovs = ['albedo', 'normal', 'roughness', 'metallic', 'irradiance']

    # Compute required_aovs by excluding drop_aovs (for estimation)
    fed_aovs = [aov for aov in required_aovs if aov not in args.drop_aovs]
    # For x2rgb pipeline, we always pass all intrinsic maps

    # Load models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load rgb2x pipeline
    rgb2x_pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        "zheng95z/rgb-to-x",
        torch_dtype=torch.float16,
    ).to(device)
    rgb2x_pipe.scheduler = DDIMScheduler.from_config(
        rgb2x_pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    rgb2x_pipe.set_progress_bar_config(disable=True)

    # Load x2rgb pipeline
    if args.use_inpaint_model:
        x2rgb_pipe = StableDiffusionAOVDropoutPipelineInpainting.from_pretrained(
            "zheng95z/x-to-rgb-inpainting",
            torch_dtype=torch.float16,
        ).to("cuda")
    else:
        x2rgb_pipe = StableDiffusionAOVDropoutPipeline.from_pretrained(
            "zheng95z/x-to-rgb",
            torch_dtype=torch.float16,
        ).to(device)
    x2rgb_pipe.scheduler = DDIMScheduler.from_config(
        x2rgb_pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    x2rgb_pipe.set_progress_bar_config(disable=True)

    # Loop over subdirectories in input_dir
    for subdir_name in os.listdir(args.input_dir):
        subdir_path = os.path.join(args.input_dir, subdir_name)
        if os.path.isdir(subdir_path):
            # Find EXR files in the subdirectory
            exr_files = [f for f in os.listdir(subdir_path) if f.lower().endswith('.exr')]
            for exr_file in exr_files:
                exr_path = os.path.join(subdir_path, exr_file)
                composite_name = os.path.splitext(exr_file)[0].replace('_bundle0001', '')
                output_subdir = os.path.join(args.output_dir, composite_name)
                os.makedirs(output_subdir, exist_ok=True)

                # Create subdirectories for outputs
                default_output_dir = os.path.join(output_subdir, "default")
                post_comp_output_dir = os.path.join(output_subdir, "post_comp")
                intrinsics_output_dir = os.path.join(output_subdir, "intrinsics")
                os.makedirs(default_output_dir, exist_ok=True)
                os.makedirs(post_comp_output_dir, exist_ok=True)
                os.makedirs(intrinsics_output_dir, exist_ok=True)

                print(f'Estimating intrinsics for {exr_path}')
                # Estimate intrinsics once for the EXR file
                composited_intrinsics, bg_intrinsics, bg_processed_rgb_x, bg_processed_x_rgb, obj_mask_tensor = estimate_intrinsics(exr_path, rgb2x_pipe, fed_aovs, args)

                # Save background and composited intrinsic maps as PNGs
                save_intrinsics(composited_intrinsics, bg_intrinsics, intrinsics_output_dir)

                # Process each light direction using the composited intrinsics
                for light_direction in ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', 'gt_dir']:
                    print(f'Processing {exr_path} with light direction {light_direction}')
                    process_light_direction(exr_path, x2rgb_pipe, default_output_dir, composited_intrinsics, bg_processed_x_rgb, obj_mask_tensor, light_direction, args)

                    # Run a second x2rgb prediction with the background-only intrinsics
                    process_light_direction(exr_path, x2rgb_pipe, post_comp_output_dir, bg_intrinsics, bg_processed_x_rgb, obj_mask_tensor, light_direction, args, render_for_post_comp=True)

def estimate_intrinsics(exr_path, rgb2x_pipe, fed_aovs, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Read the EXR file once using ezexr
    exr_data = ezexr.imread(exr_path, rgb='hybrid')

    # Extract background image and tonemap it
    bg_linear = exr_data['background'][:, :, :3]
    bg_clipped = np.clip(bg_linear * args.intensity_rescale, 0, 1)
    bg_clipped_tensor = torch.from_numpy(bg_clipped.transpose(2, 0, 1)).to(device).float()

    if args.do_wb_rgb_x:
        color_balance = torch.mean(bg_clipped_tensor, (1, 2)) / torch.dot(torch.mean(bg_clipped_tensor, (1, 2)), torch.tensor([0.299, 0.587, 0.114]).to(bg_clipped_tensor.device))
        bg_clipped_tensor_rgb_x = torch.clamp(1 / color_balance[:, None, None] * bg_clipped_tensor, max=1.0)
    else:
        bg_clipped_tensor_rgb_x = bg_clipped_tensor

    if args.do_wb_x_rgb:
        color_balance = torch.mean(bg_clipped_tensor, (1, 2)) / torch.dot(torch.mean(bg_clipped_tensor, (1, 2)), torch.tensor([0.299, 0.587, 0.114]).to(bg_clipped_tensor.device))
        bg_clipped_tensor_x_rgb = torch.clamp(1 / color_balance[:, None, None] * bg_clipped_tensor, max=1.0)
    else:
        bg_clipped_tensor_x_rgb = bg_clipped_tensor
    
    # Resize and crop the background image
    bg_processed_rgb_x = preprocess_image(bg_clipped_tensor_rgb_x)
    bg_processed_x_rgb = preprocess_image(bg_clipped_tensor_x_rgb)

    # Prepare prompts for required_aovs
    prompts = {
        "albedo": "Albedo (diffuse basecolor)",
        "normal": "Camera-space Normal",
        "roughness": "Roughness",
        "metallic": "Metallicness",
        "irradiance": "Irradiance (diffuse lighting)",
    }

    height = width = 512

    # Estimate background's intrinsic maps using rgb2x pipeline
    generator = torch.Generator(device=device).manual_seed(args.seed)
    bg_intrinsic_maps = {}
    for aov_name in fed_aovs:
        prompt = prompts[aov_name]
        output = rgb2x_pipe(
            prompt=prompt,
            photo=bg_processed_rgb_x,
            num_inference_steps=args.inference_steps,
            height=height,
            width=width,
            generator=generator,
            required_aovs=[aov_name],
        )
        generated_pil_image = output.images[0][0]
        generated_tensor = T.ToTensor()(generated_pil_image).to(device)
        if aov_name == 'albedo':
            bg_intrinsic_maps[aov_name] = generated_tensor ** 2.2
        else:
            bg_intrinsic_maps[aov_name] = generated_tensor

    # Extract object's mask
    obj_mask = exr_data['mask'][:, :, :1]
    obj_mask_tensor = torch.from_numpy(obj_mask.transpose(2, 0, 1)).to(device).float()

    # Composite object's intrinsic maps over background's estimated intrinsic maps
    composited_intrinsics = {}
    for aov_name in fed_aovs:
        if aov_name == 'normal':
            exr_aov_name = 'normals'
            obj_map = exr_data[exr_aov_name][:, :, :4]
            obj_map[:,:,:3] = obj_map[:,:,:3] * 0.5 + 0.5
        else:
            exr_aov_name = aov_name
            obj_map = exr_data[exr_aov_name][:, :, :4]
                    
        bg_map = bg_intrinsic_maps.get(aov_name)
        obj_map_tensor = torch.from_numpy(obj_map.transpose(2, 0, 1)).to(device).float()
        alpha = obj_map_tensor[3:4, :, :]
        obj_rgb = obj_map_tensor[:3, :, :]
        composite_map = obj_rgb * alpha + bg_map * (1 - alpha)
        composited_intrinsics[aov_name] = composite_map

    return composited_intrinsics, bg_intrinsic_maps, bg_processed_rgb_x, bg_processed_x_rgb, obj_mask_tensor

def save_intrinsics(composited_intrinsics, bg_intrinsics, intrinsics_output_dir):
    for aov_name, intrinsic_map in composited_intrinsics.items():
        composite_image = intrinsic_map.cpu().clamp(0, 1)
        composite_pil = T.ToPILImage()(composite_image)
        composite_image_path = os.path.join(intrinsics_output_dir, f'composited_{aov_name}.png')
        composite_pil.save(composite_image_path)

    for aov_name, intrinsic_map in bg_intrinsics.items():
        bg_image = intrinsic_map.cpu().clamp(0, 1)
        bg_pil = T.ToPILImage()(bg_image)
        bg_image_path = os.path.join(intrinsics_output_dir, f'bg_{aov_name}.png')
        bg_pil.save(bg_image_path)


def process_light_direction(exr_path, x2rgb_pipe, output_dir, composited_intrinsics, bg_processed_x_rgb, obj_mask_tensor, light_direction, args, render_for_post_comp=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    composite_name = os.path.basename(exr_path).replace('_bundle0001.exr', '')

    positive_guidance_image = load_ldr_image(os.path.join(args.shadows_dir, composite_name, f'shadow_positive_composite_{light_direction}.png'))
    negative_guidance_image = load_ldr_image(os.path.join(args.shadows_dir, composite_name, f'shadow_negative_composite_{light_direction}.png'))
    positive_mask = load_ldr_image(os.path.join(args.shadows_dir, composite_name, f'shadow_positive_{light_direction}.png'))
    negative_mask = load_ldr_image(os.path.join(args.shadows_dir, composite_name, f'shadow_negative_{light_direction}.png'))

    # Prepare parameters for x2rgb pipeline
    albedo_image = composited_intrinsics.get('albedo', None)
    normal_image = composited_intrinsics.get('normal', None) * 2 - 1
    roughness_image = composited_intrinsics.get('roughness', None)
    metallic_image = composited_intrinsics.get('metallic', None)
    irradiance_image = composited_intrinsics.get('irradiance', None)

    # Process with x2rgb pipeline
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Added part that does inpainting
    encoded_background = x2rgb_pipe.vae.encode((bg_processed_x_rgb * 2 - 1).unsqueeze(0).half()).latent_dist.mode() * x2rgb_pipe.vae.config.scaling_factor
    inpainting_mask = (T.Resize((64, 64))(obj_mask_tensor) > 0.0).float()
    inpainting_mask = dilation(inpainting_mask.unsqueeze(0), torch.ones(3, 3).to(inpainting_mask.device)).squeeze(0)
    inpainting_mask_up = T.Resize((512, 512), interpolation=T.InterpolationMode.NEAREST)(inpainting_mask).float()

    if args.use_inpaint_model:
        if args.dilation_size > 1:
            dilated_obj_mask = F.max_pool2d((obj_mask_tensor > 0).float(), kernel_size=args.dilation_size, stride=1, padding=args.dilation_size//2)
        else:
            dilated_obj_mask = (obj_mask_tensor > 0).float()
        positive_mask_inpaint = 1 - (1 - positive_mask) * (1 - (dilated_obj_mask > 0).float()).cpu()
        positive_mask_inpaint = (positive_mask_inpaint.to(device)[:1,:,:] > 0.5).float()
        if args.use_rectangular_mask:
            positive_mask_inpaint = mask_to_bbox_mask(positive_mask_inpaint)
        positive_mask_inpaint = 1 - positive_mask_inpaint
        positive_masked_image = positive_mask_inpaint * bg_processed_x_rgb.unsqueeze(0) * 2 - 1
    
        # Creating the negative mask
        negative_mask_inpaint = 1 - (1 - negative_mask) * (1 - (dilated_obj_mask > 0).float()).cpu()
        negative_mask_inpaint = (negative_mask_inpaint.to(device)[:1,:,:] > 0.5).float()
        if args.use_rectangular_mask:
            negative_mask_inpaint = mask_to_bbox_mask(negative_mask_inpaint)
        negative_mask_inpaint = 1 - negative_mask_inpaint
        negative_masked_image = negative_mask_inpaint * bg_processed_x_rgb.unsqueeze(0) * 2 - 1

        # print('positive_masked_image.min(), positive_masked_image.max()', positive_masked_image.min(), positive_masked_image.max())
        # print('positive_mask.min(), negative_mask.max()', positive_mask.min(), negative_mask.max())

        output = x2rgb_pipe(
            prompt='',
            albedo=albedo_image,
            normal=normal_image,
            roughness=roughness_image,
            metallic=metallic_image,
            irradiance=irradiance_image,
            num_inference_steps=args.inference_steps,
            height=bg_processed_x_rgb.shape[1],
            width=bg_processed_x_rgb.shape[2],
            generator=generator,
            required_aovs=['albedo', 'normal', 'roughness', 'metallic', 'irradiance'],
            guidance_scale=0,
            image_guidance_scale=0,
            guidance_rescale=0.7,
            output_type='numpy',
            encoded_background=encoded_background,
            inpainting_mask=inpainting_mask,
            positive_guidance_image=positive_guidance_image.unsqueeze(0).to(device),
            negative_guidance_image=negative_guidance_image.unsqueeze(0).to(device),
            positive_mask=positive_mask.unsqueeze(0).to(device), # TODO: clean up
            negative_mask=negative_mask.unsqueeze(0).to(device), # TODO: clean up
            object_mask=obj_mask_tensor.unsqueeze(0).to(device),
            latent_mask_weight=[args.latent_mask_weight, args.latent_mask_weight],
            relighting_guidance_scale=args.relighting_guidance_scale if not render_for_post_comp else 1.0,
            render_for_post_comp=render_for_post_comp,
           #mask=torch.stack([positive_mask, negative_mask], dim=0),
           #masked_image=torch.cat([positive_masked_image, negative_masked_image], dim=0)
           mask=torch.stack([positive_mask_inpaint, negative_mask_inpaint], dim=0),
           masked_image=torch.cat([positive_masked_image, negative_masked_image], dim=0)
        )
    else:
        output = x2rgb_pipe(
            prompt='',
            albedo=albedo_image,
            normal=normal_image,
            roughness=roughness_image,
            metallic=metallic_image,
            irradiance=irradiance_image,
            num_inference_steps=args.inference_steps,
            height=bg_processed_x_rgb.shape[1],
            width=bg_processed_x_rgb.shape[2],
            generator=generator,
            required_aovs=['albedo', 'normal', 'roughness', 'metallic', 'irradiance'],
            guidance_scale=0,
            image_guidance_scale=0,
            guidance_rescale=0.7,
            output_type='numpy',
            encoded_background=encoded_background,
            inpainting_mask=inpainting_mask,
            positive_guidance_image=positive_guidance_image.unsqueeze(0).to(device),
            negative_guidance_image=negative_guidance_image.unsqueeze(0).to(device),
            positive_mask=positive_mask.unsqueeze(0).to(device), # TODO: clean up
            negative_mask=negative_mask.unsqueeze(0).to(device), # TODO: clean up
            object_mask=obj_mask_tensor.unsqueeze(0).to(device),
            latent_mask_weight=[args.latent_mask_weight, args.latent_mask_weight],
            relighting_guidance_scale=args.relighting_guidance_scale if not render_for_post_comp else 1.0,
            render_for_post_comp=render_for_post_comp
        )
    generated_image = output.images[0]

    if args.do_wb_x_rgb:
        generated_image = np.clip(((generated_image ** 2.2) * color_balance.cpu().numpy()[None, None, :]) ** (1 / 2.2), 0, 1)

    if args.intensity_rescale:
        generated_image = np.clip(((generated_image ** 2.2) / args.intensity_rescale) ** (1 / 2.2), 0, 1)

    output_image_path = os.path.join(output_dir, f'{light_direction}_pos_masked_image.png')
    save_np_image((positive_masked_image[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) ** (1 / 2.2), output_image_path)
    output_image_path = os.path.join(output_dir, f'{light_direction}_neg_masked_image.png')
    save_np_image((negative_masked_image[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) ** (1 / 2.2), output_image_path)

    # Save the output image to output_dir
    output_image_path = os.path.join(output_dir, f'{light_direction}.png')
    save_np_image(generated_image, output_image_path)

    output_image_path = os.path.join(output_dir, f'inpainting_mask_upsampled.png')
    save_np_image(inpainting_mask_up.cpu().permute(1, 2, 0).numpy().repeat(3, 2), output_image_path)

def preprocess_image(image_tensor):
    # Resize so that the smallest dimension is 512, maintain aspect ratio
    resize_size = 512
    c, h, w = image_tensor.shape
    if h < w:
        scale_factor = resize_size / h
    else:
        scale_factor = resize_size / w
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    resize_transform = T.Resize((new_h, new_w))
    image_resized = resize_transform(image_tensor)

    # Center crop to 512x512
    crop_size = 512
    center_crop = T.CenterCrop(crop_size)
    image_cropped = center_crop(image_resized)

    return image_cropped

def save_np_image(image_array, image_path):
    # image_array is expected to be a numpy array of shape (H, W, C) in [0, 1]
    image = np.clip(image_array, 0, 1)
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(image_path)

if __name__ == '__main__':
    main()
