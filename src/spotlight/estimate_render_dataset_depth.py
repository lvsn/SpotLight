import diffusers
import torch
from src.zerocomp.data.dataset_render import RenderDataset
from torchvision.transforms import v2
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import ezexr
import os
import numpy as np
from src.zerocomp.controlnet_input_handle import match_depth_from_footprint, handle_depth_anything, handle_zoedepth, handle_depth_anything_v2_relative, handle_depth_anything_v2_metric

EPS = 1e-5


def fov_to_focal_length(fov_degrees, width, height):
    """
    Convert field of view to focal length.

    Parameters:
    - fov_degrees: Field of view in degrees
    - width: Width of the image
    - height: Height of the image

    Returns:
    - fx, fy: Focal lengths in the x and y directions
    """
    fov_radians = np.deg2rad(fov_degrees)
    fx = width / (2 * np.tan(fov_radians / 2))
    fy = height / (2 * np.tan(fov_radians / 2))
    return fx, fy


def depth_map_to_point_cloud(depth_map, fov):
    if isinstance(depth_map, np.ndarray):
        if len(depth_map.shape) == 2:
            height, width = depth_map.shape
        else:
            height, width, _ = depth_map.shape
        fov_rad = np.radians(fov)
        focal_length = width / (2 * np.tan(fov_rad / 2))

        i, j = np.meshgrid(np.arange(height), np.arange(width))
        i = i - height / 2
        j = j - width / 2

        y = (i * depth_map) / focal_length
        x = (j * depth_map) / focal_length
        z = depth_map

        point_cloud = np.stack([x, -y, -z], axis=-1)
    elif isinstance(depth_map, torch.Tensor):
        c, height, width = depth_map.shape
        fov_rad = torch.tensor(np.radians(fov), dtype=depth_map.dtype, device=depth_map.device)
        focal_length = width / (2 * torch.tan(fov_rad / 2))

        i, j = torch.meshgrid(torch.arange(height, dtype=depth_map.dtype, device=depth_map.device), torch.arange(width, dtype=depth_map.dtype, device=depth_map.device))
        i = i - height / 2
        j = j - width / 2

        y = (i * depth_map) / focal_length
        x = (j * depth_map) / focal_length
        z = depth_map

        point_cloud = torch.cat([x, -y, -z], dim=0)
    else:
        raise NotImplementedError

    point_cloud = point_cloud.reshape(-1, 3)

    return point_cloud


def depth_map_to_point_cloud_old(depth_map, fov):
    if isinstance(depth_map, np.ndarray):
        if len(depth_map.shape) == 2:
            height, width = depth_map.shape
        else:
            height, width, _ = depth_map.shape
        fov_rad = np.radians(fov)
        focal_length = width / (2 * np.tan(fov_rad / 2))

        i, j = np.meshgrid(np.arange(width), np.arange(height))
        i = i - width / 2
        j = j - height / 2

        x = (i * depth_map) / focal_length
        y = (j * depth_map) / focal_length
        z = depth_map

        point_cloud = np.stack([x, y, z], axis=-1)
    elif isinstance(depth_map, torch.Tensor):
        c, height, width = depth_map.shape
        fov_rad = torch.tensor(np.radians(fov), dtype=depth_map.dtype, device=depth_map.device)
        focal_length = width / (2 * torch.tan(fov_rad / 2))

        i, j = torch.meshgrid(torch.arange(width, dtype=depth_map.dtype, device=depth_map.device), torch.arange(height, dtype=depth_map.dtype, device=depth_map.device))
        i = i - width / 2
        j = j - height / 2

        x = (i * depth_map) / focal_length
        y = (j * depth_map) / focal_length
        z = depth_map

        point_cloud = torch.cat([x, y, z], dim=0)
    else:
        raise NotImplementedError

    point_cloud = point_cloud.reshape(-1, 3)

    return point_cloud


def get_face_indices(depth_shape):
    # Assume that neighboring pixels in the depth map are connected in the mesh
    face_indices = []

    h, w = depth_shape

    for i in range(h - 1):
        for j in range(w - 1):
            v0 = i * w + j
            v1 = i * w + j + 1
            v2 = (i + 1) * w + j
            v3 = (i + 1) * w + j + 1
            face_indices.append([v0, v1, v2])
            face_indices.append([v1, v3, v2])

    return face_indices


def save_obj_file(point_cloud, face_indices, output_path):
    with open(output_path, 'w') as f:
        for p in point_cloud:
            f.write(f'v {p[0]} {p[1]} {p[2]}\n')
        for face in face_indices:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')


if __name__ == '__main__':
    mode = 'depth'
    device = 'cuda:0'
    use_matched = True
    model_name = 'depthanythingv2_relative'
    fov_degrees = 50.0
    
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

    dataset_folder = 'data/rendered_2024-10-20_19-07-58_control'
    save_folder = f'data/bg_depth_{model_name}'
    dataset = RenderDataset(dataset_folder, transforms=transforms)
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    face_indices = None

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

                # Convert depth map to mesh
                # point_cloud = depth_map_to_point_cloud(depth, fov_degrees)
                point_cloud = depth_map_to_point_cloud_old(bg_depth_matched_b, fov_degrees)

                # # Also save the point cloud to an XYZ file
                # output_path = os.path.join(folder, f'{name}_point_cloud.xyz')
                # np.savetxt(output_path, point_cloud, fmt='%.6f')

                face_indices = get_face_indices(bg_depth_matched_b.shape) if face_indices is None else face_indices

                # Save the mesh to an OBJ file
                os.makedirs(os.path.join(save_folder, name[b], 'obj'), exist_ok=True)
                obj_file_name = f'{name[b]}_{model_name}_matched' if use_matched else f'{name[b]}_{model_name}'
                obj_file_name = obj_file_name + '.obj'
                output_path = os.path.join(save_folder, name[b], 'obj', obj_file_name)
                save_obj_file(point_cloud, face_indices, output_path)
