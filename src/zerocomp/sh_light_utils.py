import hdrio
import cv2
import matplotlib.pyplot as plt
import imageio
import numpy as np
import skylibs
import envmap
from envmap import EnvironmentMap
from scipy.ndimage import morphology
import ezexr
import imageio.v3 as imageio
import PIL.Image
from PIL import ImageDraw
from typing import Literal
import os
import glob
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import itertools
import skimage.morphology
from skimage.transform import resize
import random
from datetime import datetime
import struct
import cv2
from sdi_utils import depth_map_to_point_cloud
import torch
import sdi_utils
from scipy.spatial.transform import Rotation


def generate_gaussian_image(size=40, sigma=1.0):
    """Generate a 2D Gaussian distribution centered in the image."""
    x = np.linspace(-size//2, size//2, size)
    y = np.linspace(-size//2, size//2, size)
    x, y = np.meshgrid(x, y)
    
    # Center the Gaussian at the middle of the image
    center_x = center_y = 0
    gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    
    return gaussian

def plot_blurry_dot(ax, size=200, sigma=2.0):
    gaussian_image = generate_gaussian_image(size=size, sigma=sigma)
    
    ax.imshow(gaussian_image, cmap='Blues', extent=[-size//2, size//2, -size//2, size//2])
    ax.set_aspect('equal')


def light_position_to_direction(light_xyz, center):
    light_direction = light_xyz - center
    light_direction = light_direction / np.linalg.norm(light_direction)
    return light_direction

def spherical_to_cartesian(azimuth, zenith):
    # to skylibs coordinate system
    x = np.sin(zenith) * np.cos(azimuth)
    y = np.cos(zenith)
    z = np.sin(zenith) * np.sin(azimuth)
    return np.array([x, y, z]).T

def cartesian_to_spherical(x, y, z):
    azimuth = np.arctan2(z, x)
    zenith = np.arccos(y / np.sqrt(x**2 + y**2 + z**2)) # if unit vector = arccos(y)
    return np.array([azimuth, zenith])


def draw_light_direction_on_image(image, light_xyz, *, stick_position=None, coordinate_system: Literal['blender', 'skylibs'] = 'skylibs'):
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    assert len(light_xyz) == 3
    # assert norm of 1
    if type(light_xyz) == torch.Tensor:
        light_xyz = light_xyz.cpu().numpy()
    light_xyz = light_xyz / np.linalg.norm(light_xyz)


    image_shape = image.shape[:2]
    image = PIL.Image.fromarray((image * 255).astype(np.uint8))
  
    # draw stick
    if stick_position is not None:
        stick_end = np.array([stick_position[1], stick_position[0]])
    else:
        stick_end = np.array([image_shape[1] * 0.5, image_shape[0] * 0.75])

    # stick_start = np.array([image_shape[1] * 0.5, image_shape[0] * 0.5])
    stick_start = stick_end - np.array([0, 0.25 * image_shape[0]])
    stick_length = np.linalg.norm(stick_end - stick_start)
    
    # draw stick's shadow
    shadow_start = stick_end
    # NOTE: not perfect calculation of shadow length (doesn't account for light zenith)
    y_perspective_factor = 0.5
    if coordinate_system == 'blender':
        shadow_length = stick_length / light_xyz[2] * (1 - light_xyz[2] ** 2) ** 0.5
        shadow_end = shadow_start - np.array([light_xyz[0], -light_xyz[1] * y_perspective_factor]) * shadow_length
    elif coordinate_system == 'skylibs':
        shadow_length = stick_length / light_xyz[1] * (1 - light_xyz[1] ** 2) ** 0.5
        shadow_end = shadow_start - np.array([light_xyz[0], light_xyz[2] * y_perspective_factor]) * shadow_length
    ImageDraw.Draw(image).line([tuple(shadow_start), tuple(shadow_end)], fill=(128, 128, 128, 128), width=7, joint='curve')
    ImageDraw.Draw(image).line([tuple(shadow_start), tuple(shadow_end)], fill=(0, 0, 0, 128), width=5, joint='curve')
    ImageDraw.Draw(image).line([tuple(stick_start), tuple(stick_end)], fill=(255, 0, 0), width=5, joint='curve')

    return np.array(image) / 255

def emphasize_masked_region(image, mask, *, method: Literal['dim_outside', 'contour'] = 'dim_outside'):
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    if len(mask.shape) == 3:
        mask = mask[..., 0]
    assert image.shape[:2] == mask.shape

    if method == 'dim_outside':
        image = np.where(mask[..., np.newaxis], image, image * 0.5)
    elif method == 'contour':
        mask_dilated = morphology.grey_dilation(mask, size=5, mode='wrap')
        mask_contour = mask_dilated & ~mask
        image = np.where(mask_contour[..., np.newaxis], np.clip(np.array([0, 0.5, 0]) + image, 0, 1), image)
    return image


def reinhard(x):
    return x / (1 + x)

MASK_SIZE = 0.1
MASK_RANGES = []

MASK_SIZE = 0.3
for k in range(3):
    MASK_RANGES.append(((0.6, 0.9), (MASK_SIZE * k, MASK_SIZE * (k + 1))))
for k in range(3):
    MASK_RANGES.append(((0.8, 0.99), (MASK_SIZE * k, MASK_SIZE * (k + 1))))

MASK_CONFIGS = [None]

MASK_SIZE = 0.4
for k in range(2):
    MASK_RANGES.append(((0.6, 0.99), (MASK_SIZE * k, MASK_SIZE * (k + 1))))

if os.getenv('CC_CLUSTER'):
    OPENROOMS_DIR = os.path.join(os.getenv('SLURM_TMPDIR'), 'datasets', 'openrooms_main_xml1')
    IMAGE_DIR = os.path.join(OPENROOMS_DIR, 'Image')
    LIGHT_SOURCE_DIR = os.path.join(OPENROOMS_DIR, 'LightSource')
    GEOMETRY_DIR = os.path.join(OPENROOMS_DIR, 'Geometry')
    MATERIAL_DIR = os.path.join(OPENROOMS_DIR, 'Material')

    OUTPUT_DIR = os.path.join(os.getenv('SCRATCH'), 'outputs-sh', datetime.now().strftime("%Y%m%d-%H%M%S"))
else:
    OPENROOMS_DIR = 'OpenRooms'
    IMAGE_DIR = os.path.join(OPENROOMS_DIR, 'Image/main_xml')
    LIGHT_SOURCE_DIR = os.path.join(OPENROOMS_DIR, 'LightSource/main_xml')
    GEOMETRY_DIR = os.path.join(OPENROOMS_DIR, 'Geometry/main_xml')
    MATERIAL_DIR = os.path.join(OPENROOMS_DIR, 'Material/main_xml')

    OUTPUT_DIR = os.path.join('outputs-sh', datetime.now().strftime("%Y%m%d-%H%M%S"))


def skylibs_to_blender_xyz(skylibs_xyz):
    return np.array([skylibs_xyz[0], -skylibs_xyz[2], skylibs_xyz[1]])

def blender_to_skylibs_xyz(blender_xyz):
    return np.array([blender_xyz[0], blender_xyz[2], -blender_xyz[1]])


# def draw_light_direction_on_image(image, blender_light_xyz, camera_rot_x):
#     assert len(image.shape) == 3
#     assert image.shape[2] == 3
#     assert len(blender_light_xyz) == 3
# 
#     image_shape = image.shape[:2]
#     image = PIL.Image.fromarray((image * 255).astype(np.uint8))
#   
#     # draw stick
#     stick_start = np.array([image_shape[1] * 0.5, image_shape[0] * 0.5])
#     stick_end = np.array([image_shape[1] * 0.5, image_shape[1] * 0.75])
#     ImageDraw.Draw(image).line([tuple(stick_start), tuple(stick_end)], fill=(255, 0, 0), width=5, joint='curve')
# 
#     
#     # draw stick's shadow
#     shadow_start = stick_end
#     # NOTE: not perfect calculation of shadow length (doesn't account for light zenith)
#     blender_light_xyz = np.array(blender_light_xyz) / np.linalg.norm(np.array(blender_light_xyz)[:2])
#     shadow_end = shadow_start - np.array([blender_light_xyz[0], -blender_light_xyz[1]  * np.cos(camera_rot_x - np.deg2rad(15))]) * min(image_shape[1], image_shape[0]) / 8
#     ImageDraw.Draw(image).line([tuple(shadow_start), tuple(shadow_end)], fill=(0, 0, 0, 128), width=5, joint='curve')
#     return np.array(image) / 255
# 

def draw_light_direction_on_image(image, light_xyz, camera_rot_x, *, stick_position=None, coordinate_system: Literal['blender', 'skylibs'] = 'skylibs'):
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    assert len(light_xyz) == 3

    image_shape = image.shape[:2]
    image = PIL.Image.fromarray((image * 255).astype(np.uint8))
  
    # draw stick
    if stick_position is not None:
        stick_end = np.array([stick_position[1], stick_position[0]])
    else:
        stick_end = np.array([image_shape[1] * 0.5, image_shape[0] * 0.75])

    # stick_start = np.array([image_shape[1] * 0.5, image_shape[0] * 0.5])
    stick_start = stick_end - np.array([0, 0.25 * image_shape[0]])
    ImageDraw.Draw(image).line([tuple(stick_start), tuple(stick_end)], fill=(255, 0, 0), width=5, joint='curve')

    
    # draw stick's shadow
    shadow_start = stick_end
    # NOTE: not perfect calculation of shadow length (doesn't account for light zenith)
    if coordinate_system == 'blender':
        shadow_end = shadow_start - np.array([light_xyz[0], -light_xyz[1]  * np.cos(camera_rot_x - np.deg2rad(15))]) * min(image_shape[1], image_shape[0]) / 8
    elif coordinate_system == 'skylibs':
        shadow_end = shadow_start - np.array([light_xyz[0], light_xyz[2]  * 0.8]) * min(image_shape[1], image_shape[0]) / 8
    ImageDraw.Draw(image).line([tuple(shadow_start), tuple(shadow_end)], fill=(255, 255, 255, 128), width=7, joint='curve')
    ImageDraw.Draw(image).line([tuple(shadow_start), tuple(shadow_end)], fill=(0, 0, 0, 128), width=5, joint='curve')
    return np.array(image) / 255


def dumb_get_light_source_from_envmap(e):
    # argmin
    pos = np.unravel_index(np.argmax(e.data), e.data.shape)
    x, y, z, _ = e.pixel2world(pos[1], pos[0])
    return x, y, z

def approx_grayscale_lambertian_shading(image, diffuse, *, handle_missing_diffuse_channels=True):
    # luminance = np.dot(image, [0.2126, 0.7152, 0.0722])
    luminance_constants = np.array([0.2126, 0.7152, 0.0722])[np.newaxis, np.newaxis, :]
    diffuse_valid_mask = diffuse > 1e-3
    valid_diffuse_mask = np.all(diffuse_valid_mask, axis=2)

    if handle_missing_diffuse_channels:
        approx_grayscale_shading = image / diffuse # it should return nans here, which is okay
        approx_grayscale_shading = np.sum(approx_grayscale_shading * diffuse_valid_mask * luminance_constants, axis=2, keepdims=True) / np.clip(
            np.sum(luminance_constants * diffuse_valid_mask, axis=2, keepdims=True),
            1e-5, None)
    else:
        approx_grayscale_shading = np.sum((image / np.clip(diffuse, 1e-5, None)) * luminance_constants, axis=2, keepdims=True)

    return approx_grayscale_shading, valid_diffuse_mask

def reshade(image, normals, environment_map, *, mask=None):
    assert image.shape[:2] == normals.shape[:2] == mask.shape[:2]
    assert image.shape[2] == 1

    normals = normals.reshape(-1, 3)
    u, v = environment_map.world2image(normals[:, 0], normals[:, 1], normals[:, 2])
    u, v = u.reshape(image.shape[:2]), v.reshape(image.shape[:2])

    reshaded = environment_map.copy().interpolate(u, v, order=1).data
    if mask is not None:

        if len(mask.shape) == 2:
            mask = mask[..., np.newaxis]
        reshaded = mask*reshaded + (1 - mask)*image

    return reshaded

def scale_hdr(hdr, mode='test'):
    intensity_arr = (hdr).flatten()
    intensity_arr.sort()
    if mode == 'train':
        scale = (0.95 - 0.1 * np.random.random()) / np.clip(intensity_arr[int(0.95 * len(intensity_arr))], 0.1, None)
    else:
        scale = (0.95 - 0.05) / np.clip(intensity_arr[int(0.95 * len(intensity_arr))], 0.1, None)

    return scale

def cutout_sphere(depth, *, min_distance, max_distance, fov=60):
    point_cloud = depth_map_to_point_cloud(depth, fov=fov) # TODO: validate FOV
    mask_center = (random.randint(0, depth.shape[0]-1), random.randint(0, depth.shape[1]-1))
    sphere_center = point_cloud[mask_center[0], mask_center[1]]
    sphere_radius = random.uniform(min_distance, max_distance)
    mask_image = np.linalg.norm(point_cloud - sphere_center, axis=2) < sphere_radius
    return mask_image

"""
cutout_sphere_kwargs:
  fov: 60
  min_distance: 0.7
  max_distance: 1.3
  min_minkowski_distance: 0.8
  max_minkowski_distance: 2.5
  rotate_points: true
  anisotropic: true
"""
def cutout_sphere(depth, *, fov, min_distance, max_distance, min_minkowski_distance, max_minkowski_distance, rotate_points, anisotropic, mask_center_border_pixels=0):
    point_cloud = depth_map_to_point_cloud(depth, fov=fov) # TODO: validate FOV
    transformation_matrix = np.eye(3)

    if anisotropic:
        scaling_matrix = np.diag([(1 / random.uniform(min_distance, max_distance)) for _ in range(3)])
    else:
        scaling_matrix = np.eye(3) * (1 / random.uniform(min_distance, max_distance))
    transformation_matrix = scaling_matrix @ transformation_matrix

    if rotate_points:
        rotation_matrix = Rotation.random().as_matrix()
        transformation_matrix = rotation_matrix @ transformation_matrix

    point_cloud_vector = point_cloud.reshape(-1, 3).T
    transformed_point_cloud_vector = transformation_matrix @ point_cloud_vector
    transformed_point_cloud = transformed_point_cloud_vector.T.reshape(point_cloud.shape)

    mask_center = (random.randint(mask_center_border_pixels, depth.shape[0]-1-mask_center_border_pixels), random.randint(mask_center_border_pixels, depth.shape[1]-1-mask_center_border_pixels))
    sphere_center = point_cloud[mask_center[0], mask_center[1]]
    minkowski_distance = random.uniform(min_minkowski_distance, max_minkowski_distance)
    mask_image = np.sum(np.abs(sphere_center - transformed_point_cloud) ** minkowski_distance, axis=2) ** (1/minkowski_distance) < 1.0

    return mask_image

def camera_to_image_space(light_xyz, *, x_fov):
    focal_length = 1/np.tan(np.deg2rad(x_fov)/2)
    light_xyz_image_space = light_xyz * focal_length / (-light_xyz[2])
    return light_xyz_image_space

def camera_to_pixel_space(light_xyz, *, x_fov, image_shape):
    light_xyz_image_space = camera_to_image_space(light_xyz, x_fov=x_fov)
    image_to_pixels = image_shape[1] / 2
    pixel_center = (int(np.clip( light_xyz_image_space[0] * image_to_pixels + image_shape[1] / 2, 0, image_shape[1] - 1)),
                    int(np.clip(-light_xyz_image_space[1] * image_to_pixels + image_shape[0] / 2, 0, image_shape[0] - 1)))
    # converts to imagedraw's coordinates (0,0) at top left, in x,y order.
    return pixel_center

def draw_dot_at_light_position(image, light_xyz_camera_space, color=(255, 0, 0), *, x_fov):
    if isinstance(image, np.ndarray):
        assert len(light_xyz_camera_space) == 3
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        assert image.max() <= 1.0
        # pixel_center = (light_xyz[1] * image.shape[1] / 2 + image.shape[1] / 2, light_xyz[0] * image.shape[0] / 2 + image.shape[0] / 2)
        # clamped
        pixel_center = camera_to_pixel_space(light_xyz_camera_space, x_fov=x_fov, image_shape=image.shape[:2])
        print(light_xyz_camera_space)
        print(x_fov)
        print(image.shape[:2])
        print(pixel_center)
        
        # draw circle
        image = PIL.Image.fromarray((image * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.ellipse((pixel_center[0] - 5, pixel_center[1] - 5, pixel_center[0] + 5, pixel_center[1] + 5), fill=color)
        return np.array(image) / 255
    else:
        raise NotImplementedError

def write_text_to_image(image, text, color=(255, 0, 0)):
    if isinstance(image, np.ndarray):
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        assert image.max() <= 1.0
        image = PIL.Image.fromarray((image * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), text, fill=color, font_size=40)
        return np.array(image) / 255
    else:
        raise NotImplementedError


#IMAGE_DIR = 'OpenRooms/Image/main_xml'
#LIGHT_SOURCE_DIR = 'OpenRooms/LightSource/main_xml'
#GEOMETRY_DIR = 'OpenRooms/Geometry/main_xml'
#MATERIAL_DIR = 'OpenRooms/Material/main_xml'

ENV_DIM = (50, 100)
DOWNSAMPLE_FACTOR = 2.5



if __name__ == '__main__':
    main()
