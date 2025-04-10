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

def skylibs_to_blender_xyz(skylibs_xyz):
    return np.array([skylibs_xyz[0], -skylibs_xyz[2], skylibs_xyz[1]])

def blender_to_skylibs_xyz(blender_xyz):
    return np.array([blender_xyz[0], blender_xyz[2], -blender_xyz[1]])

