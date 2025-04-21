import numpy as np
import torch
import os
import sys
import ezexr
from tqdm import tqdm

# pwdpath = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(-1, os.path.join(pwdpath, 'predictors'))

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


def planer_depth_2_dist(depth, focal_length):
    # Generate the image plane coordinates
    height, width = depth.shape
    npy_image_plane_x = np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width).reshape(1, width).repeat(height, 0).astype(np.float32)[:, :, None]
    npy_image_plane_y = np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height).reshape(height, 1).repeat(width, 1).astype(np.float32)[:, :, None]
    npy_image_plane_z = np.full([height, width, 1], focal_length, np.float32)
    npy_image_plane = np.concatenate([npy_image_plane_x, npy_image_plane_y, npy_image_plane_z], 2)

    euclidean_distance = np.linalg.norm(npy_image_plane, axis=2) / focal_length * depth

    return euclidean_distance


def dist_2_planar_depth(dist, focal_length):
    height, width = dist.shape
    npy_image_plane_x = np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width).reshape(1, width).repeat(height, 0).astype(np.float32)[:, :, None]
    npy_image_plane_y = np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height).reshape(height, 1).repeat(width, 1).astype(np.float32)[:, :, None]
    npy_image_plane_z = np.full([height, width, 1], focal_length, np.float32)
    npy_image_plane = np.concatenate([npy_image_plane_x, npy_image_plane_y, npy_image_plane_z], 2)

    planar_depth = dist / np.linalg.norm(npy_image_plane, axis=2) * focal_length

    return planar_depth


def match_depth_from_footprint(background_depth, object_depth, object_footprint_depth, adjust_background=False):
    assert type(background_depth) == type(object_depth) == type(object_footprint_depth) == torch.Tensor
    assert background_depth.shape == object_depth.shape == object_footprint_depth.shape
    assert len(background_depth.shape) == 4
    assert background_depth.shape[1] == 1

    adjusted_object_depth = torch.empty_like(object_depth)
    adjusted_background_depth = torch.empty_like(background_depth)
    for batch_idx in range(background_depth.shape[0]):
        # we have to loop this, since the number of masked elements differs between batch elements
        footprint_mask = object_footprint_depth[batch_idx, 0, :, :] > 0
        flattened_background_depths = background_depth[batch_idx, 0, footprint_mask]
        flattened_footprint_depths = object_footprint_depth[batch_idx, 0, footprint_mask]

        # solve using least squares
        A = torch.vstack([flattened_footprint_depths, torch.ones(len(flattened_footprint_depths), device=flattened_footprint_depths.device)]).T
        y = flattened_background_depths
        m, c = torch.linalg.lstsq(A, y).solution
        adjusted_object_depth[batch_idx] = object_depth[batch_idx] * m + c
        adjusted_background_depth[batch_idx] = (background_depth[batch_idx] - c) / m

    if adjust_background:
        return adjusted_background_depth
    else:
        return adjusted_object_depth


def match_depth_from_footprint_numpy(background_depth, object_depth, object_footprint_depth, adjust_background=False):
    assert type(background_depth) == type(object_depth) == type(object_footprint_depth) == np.ndarray
    assert background_depth.shape == object_depth.shape == object_footprint_depth.shape
    assert len(background_depth.shape) == 2

    adjusted_object_depth = np.empty_like(object_depth)
    adjusted_background_depth = np.empty_like(background_depth)

    footprint_mask = object_footprint_depth > 0
    flattened_background_depths = background_depth[footprint_mask]
    flattened_footprint_depths = object_footprint_depth[footprint_mask]

    # solve using least squares
    # Ax + b = y, x=bg_depth, y=obj_fp_depth
    A = np.vstack([flattened_footprint_depths, np.ones(len(flattened_footprint_depths))]).T
    m, c = np.linalg.lstsq(A, flattened_background_depths, rcond=None)[0]
    adjusted_object_depth = object_depth * m + c
    adjusted_background_depth = (background_depth - c) / m

    if adjust_background:
        return adjusted_background_depth
    else:
        return adjusted_object_depth


if __name__ == "__main__":
    # All model depth outputs should not be matched #
    # model_name = 'depthanything'
    # model_name = 'zoedepth'
    # model_name = 'marigold'
    model_name = 'depthanythingv2_relative'
    # model_name = 'depthanythingv2_metric'

    use_matched = True

    dataset_name = 'labo'
    dataset_name = 'fred'
    bg_depth_folder = f'../datasets/{dataset_name}/reconstructed_obj/GT_emission_envmap_{model_name}'
    obj_folder = f'../datasets/{dataset_name}/GT_emission_envmap'
    save_folder = f'../datasets/{dataset_name}/reconstructed_obj/GT_emission_envmap_{model_name}_obj'
    # save_folder = f'./{model_name}_dist'

    face_indices = None

    for name in tqdm(os.listdir(bg_depth_folder)):
        folder = os.path.join(bg_depth_folder, name)
        file_name = f'{name}_bg_depth_{model_name}_matched.exr' if use_matched else f'{name}_bg_depth_{model_name}.exr'
        bg_depth_path = os.path.join(folder, 'obj', file_name)
        try:
            bg_depth = ezexr.imread(bg_depth_path).squeeze()
        except:
            print(f"Failed to read {bg_depth_path}")
            continue

        # Field of view (example value, replace with actual value)
        fov_degrees = 50.0
        width, height = bg_depth.shape
        fx, _ = fov_to_focal_length(fov_degrees, width, height)

        if use_matched:
            obj_path = os.path.join(obj_folder, name, f'{name}_bundle0001.exr')
            try:
                obj_exr = ezexr.imread(obj_path, rgb="hybrid")
            except:
                print(f"Failed to read {obj_path}")
                continue

            obj_depth = obj_exr['depth'][:, :, 0]
            fp_depth = obj_exr['footprint_depth'][:, :, 0]

            bg_depth = match_depth_from_footprint_numpy(bg_depth, obj_depth, fp_depth, adjust_background=True)

        # Save the distance map
        save_path = os.path.join(save_folder, name)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, f'{name}_bg_distance_{model_name}.exr')
        ezexr.imwrite(save_path, bg_depth)

        # Convert depth map to mesh
        # point_cloud = depth_map_to_point_cloud(depth, fov_degrees)
        point_cloud = depth_map_to_point_cloud_old(bg_depth, fov_degrees)

        # # Also save the point cloud to an XYZ file
        # output_path = os.path.join(folder, f'{name}_point_cloud.xyz')
        # np.savetxt(output_path, point_cloud, fmt='%.6f')

        face_indices = get_face_indices(bg_depth.shape) if face_indices is None else face_indices

        # Save the mesh to an OBJ file
        os.makedirs(os.path.join(save_folder, name, 'obj'), exist_ok=True)
        obj_file_name = f'{name}_{model_name}_matched' if use_matched else f'{name}_{model_name}'
        obj_file_name = obj_file_name + '.obj'
        output_path = os.path.join(save_folder, name, 'obj', obj_file_name)
        save_obj_file(point_cloud, face_indices, output_path)
