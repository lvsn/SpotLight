import numpy as np
import torch
import skimage.io
import random
from PIL import Image
import skimage.transform
import cv2
try:
    import open3d as o3d
except:
    print("Open3D not installed \n")


# def get_conditioning_channels(conditioning_maps):
#     if conditioning_maps is None:
#         raise ValueError("conditioning_maps is None")
#     num_conditioning_channels = 0
#     for key in conditioning_maps:
#         if key == 'normal' or key == 'diffuse' or key == 'shading':
#             num_conditioning_channels += 3
#         elif key == 'depth' or key == 'mask' or key == 'roughness' or key == 'metallic':
#             num_conditioning_channels += 1
#     return num_conditioning_channels


def compute_distance_bgpc_objpc(bgpc, objpc):
    # Create Open3D PointCloud objects
    bg_cloud = o3d.geometry.PointCloud()
    bg_cloud.points = o3d.utility.Vector3dVector(bgpc)

    obj_cloud = o3d.geometry.PointCloud()
    obj_cloud.points = o3d.utility.Vector3dVector(objpc)

    # Compute distances using Open3D KDTree
    dists = bg_cloud.compute_point_cloud_distance(obj_cloud)
    dists = np.asarray(dists)

    return dists


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

    return point_cloud


def depth_map_to_point_cloud_with_rgb(depth_map, rgb_image, fov):
    # Get point cloud
    point_cloud = depth_map_to_point_cloud(depth_map, fov)

    # Combine depth map and RGB image
    if isinstance(rgb_image, np.ndarray):
        point_cloud_rgb = np.concatenate([point_cloud, rgb_image], axis=-1)
    elif isinstance(rgb_image, torch.Tensor):
        point_cloud_rgb = torch.cat([point_cloud, rgb_image], dim=0)

    return point_cloud_rgb


def color_rebalance(out_image, bg_image):
    bg_avg_color = torch.mean(bg_image, dim=(2, 3), keepdim=True)
    out_avg_color = torch.mean(out_image, dim=(2, 3), keepdim=True)
    ratio = (bg_avg_color / out_avg_color)
    return ratio


def comp_normal_to_openrooms_normal(nm):
    new_nm = nm.copy()
    # Input normal map should be in [-1, 1] range
    if new_nm.min() >= 0:
        new_nm = new_nm * 2 - 1
    new_nm[:, :, 2] = -new_nm[:, :, 2]
    # Transform it back to [0, 1] range
    new_nm = new_nm * 0.5 + 0.5
    return new_nm.clip(0, 1)

# 
# def comp_normal_to_openrooms_normal_tensor(nm):
#     new_nm = nm.clone()
#     # Input normal map should be in [-1, 1] range
#     if new_nm.min() >= 0:
#         new_nm = new_nm * 2 - 1
#     new_nm[:, 2, :, :] = -new_nm[:, 2, :, :]
#     # Transform it back to [0, 1] range
#     new_nm = new_nm * 0.5 + 0.5
#     return new_nm.clip(0, 1)
# 

def omnidata_normal_to_openrooms_normal(normal):
    # Input normal map should be in [-1, 1] range
    if normal.min() >= 0:
        normal = normal * 2 - 1
    normal[:, 1, :, :] = -normal[:, 1, :, :]
    normal[:, 2, :, :] = -normal[:, 2, :, :]
    # Transform it back to [0, 1] range
    normal = normal * 0.5 + 0.5
    return normal


def stablenormal_normal_to_openrooms_normal(normal):
    # Input normal map should be in [-1, 1] range
    if normal.min() >= 0:
        normal = normal * 2 - 1
    normal[:, 0, :, :] = -normal[:, 0, :, :]
    # Transform it back to [0, 1] range
    normal = normal * 0.5 + 0.5
    return normal


def normalized_vector_img_to_rgb_img(vector_img):
    return np.clip(np.array([
        vector_img[..., 0] * 0.5 + 0.5,
        vector_img[..., 1] * 0.5 + 0.5,
        vector_img[..., 2] * 0.5 + 0.5,
    ]), 0, 1).transpose(1, 2, 0)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log_image_to_disk(image, path):
    if len(image.shape) == 4:
        image = image.detach().cpu().numpy()[0].transpose(1, 2, 0)
    else:
        image = image.detach().cpu().numpy().transpose(1, 2, 0)
    if image.shape[2] == 1:
        image = np.tile(image, (1, 1, 3))
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    skimage.io.imsave(path, image)


# def tensor_to_numpy(img, initial_range=(0, 1)):
#     # scale to [0, 1]
#     img = img - initial_range[0]
#     img = img / (initial_range[1] - initial_range[0])
#     if img.dim() == 4:
#         img = img.squeeze(0)
#     return np.clip(img.permute(1, 2, 0).detach().cpu().numpy(), 0, 1)

def tensor_to_numpy(img, initial_range=(0, 1), clip=False):
    # scale to [0, 1]
    # TODO: should clipping be enabled by default
    img = img.detach() - initial_range[0]
    img = img / (initial_range[1] - initial_range[0])
    if img.dim() == 4:
        img = img.squeeze(0)
    img = img.permute(1, 2, 0).cpu().numpy()
    if clip:
        img = np.clip(img, 0, 1)
    return img


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


def tensor_to_pil_list(images, initial_range=(0, 1)):
    images = tensor_to_numpy_list(images, initial_range)
    images = numpy_to_pil_list(images)
    return images


def tensor_to_numpy_list(images, initial_range=(0, 1)):
    # scale to [0, 1]
    images = images - initial_range[0]
    images = images / (initial_range[1] - initial_range[0])
    if images.dim() == 3:
        images = images.unsqueeze(0)
    return np.clip(images.permute(0, 2, 3, 1).cpu().numpy(), 0, 1)


def numpy_to_pil_list(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255.0).astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image, mode="RGB") for image in images]

    return pil_images


def impath_to_numpy(image_name, is_Gamma=False):
    image = cv2.imread(image_name, -1)
    image = np.asarray(image, dtype=np.float32)

    # image = image/255.0
    
    if not image_name.endswith('.exr'):
        image = image/255.0

    if is_Gamma:
        image = image**2.2
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    if len(image.shape) == 3:
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        image = image[:, :, ::-1]

    return np.ascontiguousarray(image)


def numpy_to_tensor(img):
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).float()
    return img


def impath_to_tensor(image_name, is_Gamma=False):
    img = impath_to_numpy(image_name, is_Gamma)
    img = numpy_to_tensor(img)
    return img


def minmax_norm(img):
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img))


def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean.item())
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


def load_model_from_checkpoint(model, fpath):
    ckpt = torch.load(fpath, map_location='cpu')
    ckpt = ckpt['model']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v
    model.load_state_dict(load_dict)
    return model


def load_state_dict(model, state_dict):
    """Load state_dict into model, handling DataParallel and DistributedDataParallel. Also checks for "model" key in state_dict.

    DataParallel prefixes state_dict keys with 'module.' when saving.
    If the model is not a DataParallel model but the state_dict is, then prefixes are removed.
    If the model is a DataParallel model but the state_dict is not, then prefixes are added.
    """
    state_dict = state_dict.get('model', state_dict)
    # if model is a DataParallel model, then state_dict keys are prefixed with 'module.'

    do_prefix = isinstance(
        model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    state = {}
    for k, v in state_dict.items():
        if k.startswith('module.') and not do_prefix:
            k = k[7:]

        if not k.startswith('module.') and do_prefix:
            k = 'module.' + k

        state[k] = v

    model.load_state_dict(state, strict=True)
    print("Loaded successfully")
    return model


def load_wts(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    return load_state_dict(model, ckpt)


def load_ckpt(model, checkpoint):
    model = load_wts(model, checkpoint)
    print("Loaded weights from {0}".format(checkpoint))
    return model
