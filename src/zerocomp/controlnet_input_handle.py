import random
from typing import Any, Tuple, Union
from PIL import Image
from omegaconf import OmegaConf
# from albumentations.augmentations.dropout import Cutout, CoarseDropout
from coarse_dropout import CoarseDropout
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as tf
import sdi_utils
from safetensors.torch import load_model
import hydra

# from predictors.material_diffusion.iid.material_diffusion.iid import IntrinsicImageDiffusion

import os
import sys
import inspect
pwdpath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(-1, os.path.join(pwdpath, 'predictors'))

EPS = 1e-5
torch.hub.set_dir('.cache')


def match_depth_from_footprint(background_depth, object_depth, object_footprint_depth, *, adjust_background=False, match_disparity=False):
    # TODO: option to match disparity
    assert type(background_depth) == type(object_depth) == type(object_footprint_depth) == torch.Tensor
    assert len(background_depth.shape) == 4
    assert background_depth.shape[1] == 1
    assert object_depth.shape[1] == 2 # assert has alpha channel

    if match_disparity:
        # background_values = 1 / background_depth
        background_values = background_depth # we assume that the background depth is already in disparity
        object_values = 1 / object_depth
        object_footprint_values = 1 / object_footprint_depth
    else:
        background_values = background_depth
        object_values = object_depth
        object_footprint_values = object_footprint_depth

    adjusted_object_values = torch.empty_like(object_values)

    # keep alpha channel same as before
    adjusted_object_values[:, 1, :, :] = object_depth[:, 1, :, :]

    adjusted_background_values = torch.empty_like(background_values)
    for batch_idx in range(background_values.shape[0]):
        # we have to loop this, since the number of masked elements differs between batch elements

        footprint_mask = object_footprint_depth[batch_idx, 1, :, :] == 1.0
        flattened_background_values = background_values[batch_idx, 0, footprint_mask]
        flattened_footprint_values = object_footprint_values[batch_idx, 0, footprint_mask]

        # solve using least squares
        A = torch.vstack([flattened_footprint_values, torch.ones(len(flattened_footprint_values), device=flattened_footprint_values.device)]).T
        y = flattened_background_values
        m, c = torch.linalg.lstsq(A, y).solution
        adjusted_object_values[batch_idx, 0, :, :] = object_values[batch_idx, 0, :, :] * m + c * object_values[batch_idx, 1, :, :]
        adjusted_background_values[batch_idx] = (background_values[batch_idx] - c) / m

    if match_disparity:
        adjusted_object_depth = torch.clone(adjusted_object_values)
        adjusted_object_depth[:, 0, :, :] = 1 / adjusted_object_values[:, 0, :, :]
        adjusted_object_depth[:, 1, :, :] = object_depth[:, 1, :, :]
        adjusted_background_depth = 1 / adjusted_background_values
    else:
        adjusted_object_depth = adjusted_object_values
        adjusted_background_depth = adjusted_background_values
    
    if adjust_background:
        return adjusted_background_depth
    else:
        return adjusted_object_depth


def handle_zoedepth():
    from zoedepth.utils.config import get_config
    from zoedepth.models.builder import build_model

    # ZoeD_NK
    torch.hub.set_dir('.cache')
    conf = get_config("zoedepth_nk", "infer")
    model_zoe_nk = build_model(conf)

    for b in model_zoe_nk.core.core.pretrained.model.blocks:
        b.drop_path = torch.nn.Identity()

    model_zoe_nk.eval()
    model_zoe_nk.requires_grad_(requires_grad=False)
    return model_zoe_nk


def handle_omnidata_depth():
    from OmniData.modules.midas.dpt_depth import DPTDepthModel

    image_size = 384
    pretrained_weights_path = '.cache/checkpoints/omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
    model = DPTDepthModel(backbone='vitb_rn50_384')  # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(requires_grad=False)
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        transforms.Normalize(mean=0.5, std=0.5)])
    return model, trans_totensor


def handle_omnidata_normal():
    from OmniData.modules.midas.dpt_depth import DPTDepthModel

    image_size = 384

    pretrained_weights_path = '.cache/checkpoints/omnidata_dpt_normal_v2.ckpt'
    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)  # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(requires_grad=False)
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                                        transforms.CenterCrop(image_size)])
    return model, trans_totensor


def handle_gmnet():
    from gmnet.gmnet_v1_5 import GMNet
    gmnet = GMNet()
    gmnet = sdi_utils.load_model_from_checkpoint(gmnet, '.cache/checkpoints/gmnet_100.pt')
    gmnet.eval()
    gmnet.requires_grad_(requires_grad=False)
    return gmnet


def handle_dfnet():
    from OmniData.modules.midas.dpt_depth import DPTDepthModel
    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)  # DPT Hybrid
    # Load pretrained backbone
    model.load_state_dict(torch.load('.cache/checkpoints/dfnet.bin'))
    model.eval()
    model.requires_grad_(requires_grad=False)
    return model


def handle_dfnet_w_depth_normal():
    from OmniData.modules.midas.dpt_depth import DPTDepthModel
    from timm.layers.std_conv import StdConv2dSame
    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)  # DPT Hybrid

    input_channels = 3 + 3 + 1  # rgb + normal + depth
    old_input_conv = model.pretrained.model.patch_embed.backbone.stem.conv
    new_input_conv = StdConv2dSame(input_channels, old_input_conv.out_channels, old_input_conv.kernel_size, stride=old_input_conv.stride,
                                   padding='SAME', dilation=old_input_conv.dilation, groups=old_input_conv.groups, bias=old_input_conv.bias)
    model.pretrained.model.patch_embed.backbone.stem.conv = new_input_conv
    # Load pretrained backbone
    # load_model(model, '.cache/checkpoints/dfnet_w_dfnet.safetensors')
    load_model(model, '.cache/checkpoints/dfnet_w_hypersim2.safetensors')
    # load_model(model, '.cache/checkpoints/dfnet_w_openrooms2.safetensors')
    model.eval()
    model.requires_grad_(requires_grad=False)
    return model


def handle_depth_anything():
    from DepthAnything import get_config, build_model

    overwrite = {"pretrained_resource": "local::./.cache/checkpoints/depth_anything_metric_depth_indoor.pt"}
    config = get_config('zoedepth', "eval", 'nyu', **overwrite)
    config = get_config('zoedepth', "eval", 'diode_indoor', **overwrite)
    model = build_model(config)
    model.requires_grad_(requires_grad=False)
    model.eval()

    return model


def handle_depth_anything_v2_relative():
    from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl'  # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'.cache/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model.requires_grad_(requires_grad=False)
    model.eval()

    return model


def handle_depth_anything_v2_metric():
    from DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    encoder = 'vitl'  # or 'vits', 'vitb'
    dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20  # 20 for indoor model, 80 for outdoor model

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(f'.cache/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    model.requires_grad_(requires_grad=False)
    model.eval()

    return model


def handle_stable_normal(device):
    from stablenormal.scheduler.heuristics_ddimsampler import HEURI_DDIMScheduler
    from stablenormal.pipeline_stablenormal import StableNormalPipeline
    from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline

    default_seed = 2024
    default_batch_size = 1

    default_image_processing_resolution = 768

    default_video_num_inference_steps = 10
    default_video_processing_resolution = 768
    default_video_out_max_frames = 60

    x_start_pipeline = YOSONormalsPipeline.from_pretrained(
        '.cache/yoso-normal-v0-2', trust_remote_code=True, variant="fp16", torch_dtype=torch.float16).to(device)
    pipe = StableNormalPipeline.from_pretrained('.cache/stable-normal-v0-1', trust_remote_code=True,
                                                variant="fp16", torch_dtype=torch.float16,
                                                scheduler=HEURI_DDIMScheduler(prediction_type='sample',
                                                                              beta_start=0.00085, beta_end=0.0120,
                                                                              beta_schedule="scaled_linear"))
    pipe.x_start_pipeline = x_start_pipeline
    pipe.to(device, dtype=torch.float16)
    pipe.prior.to(device, torch.float16)

    try:
        import xformers
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers

    return pipe


def compute_shading(img, diffuse):
    shading = torch.ones_like(diffuse) * EPS
    # shading = torch.ones_like(diffuse) * -1
    diffuse_nozero_mask = diffuse > EPS
    shading[diffuse_nozero_mask] = img[diffuse_nozero_mask] / diffuse[diffuse_nozero_mask]

    shading = shading.clamp(EPS, 1e3)

    # shading = img / diffuse # It will lead to super large values
    return shading


def tensor_image_to_numpy(img, initial_range=(0, 1)):
    # scale to [0, 1]
    img = img - initial_range[0]
    img = img / (initial_range[1] - initial_range[0])
    return np.clip(img.permute(1, 2, 0).cpu().numpy(), 0, 1)

def approx_grayscale_lambertian_shading(image, diffuse, *, handle_missing_diffuse_channels=True):
    # luminance = np.dot(image, [0.2126, 0.7152, 0.0722])
    luminance_constants = np.array([0.2126, 0.7152, 0.0722])[np.newaxis, np.newaxis, :]
    diffuse_valid_mask = diffuse > 1e-3
    print('diffuse_valid_mask', diffuse_valid_mask.shape, diffuse_valid_mask.dtype, diffuse_valid_mask.min(), diffuse_valid_mask.max())
    any_valid_mask = np.all(diffuse_valid_mask, axis=2)

    if handle_missing_diffuse_channels:
        approx_grayscale_shading = image / np.clip(diffuse, 1e-5, None) # it should return nans here, which is okay
        approx_grayscale_shading = np.sum(approx_grayscale_shading * diffuse_valid_mask * luminance_constants, axis=2, keepdims=True) / np.clip(
            np.sum(luminance_constants * diffuse_valid_mask, axis=2, keepdims=True),
            1e-5, None)
    else:
        approx_grayscale_shading = np.sum((image / np.clip(diffuse, 1e-5, None)) * luminance_constants, axis=2, keepdims=True)

    return approx_grayscale_shading, any_valid_mask

# def get_dominant_light_from_sh(*, image, diffuse, normals, mask):
#     assert len(image.shape) == 3
#     assert len(diffuse.shape) == 3
#     assert len(normals.shape) == 3
#     assert len(mask.shape) == 3
#     assert mask.shape[:2] == image.shape[:2]
# 
#     mask = mask[..., 0]
#     ENV_DIM = (50, 100)
# 
#     normals = normals * 2 - 1
#     print('normals', normals.shape, normals.dtype, normals.min(), normals.max())
# 
#     estimated_lambertian_shading_gray_smart, any_valid_mask = approx_grayscale_lambertian_shading(image, diffuse, handle_missing_diffuse_channels=True)
# 
#     any_valid_mask = skimage.morphology.binary_erosion(any_valid_mask)
#     estimated_lambertian_shading_gray_smart = estimated_lambertian_shading_gray_smart * any_valid_mask[..., None]
# 
#    # print('estimated_lambertian_shading_gray_smart', estimated_lambertian_shading_gray_smart.shape, estimated_lambertian_shading_gray_smart.dtype, estimated_lambertian_shading_gray_smart.min(), estimated_lambertian_shading_gray_smart.max())
#    # print('any_valid_mask', any_valid_mask.shape, any_valid_mask.dtype, any_valid_mask.min(), any_valid_mask.max())
#    
#     final_mask = mask & any_valid_mask
#     
#     curr_mask_indices = np.where(final_mask)
#     curr_values = estimated_lambertian_shading_gray_smart[curr_mask_indices] # TODO: use smart shading
#    # print('curr_values', curr_values.shape, curr_values.min(), curr_values.max())
#     curr_normals = normals[curr_mask_indices]
#     # 
#     # print('curr_normals[0]', curr_normals[:, 0].min(), curr_normals[:, 0].max(), curr_normals[:, 0].mean())
#     # print('curr_normals[1]', curr_normals[:, 1].min(), curr_normals[:, 1].max(), curr_normals[:, 1].mean())
#     # print('curr_normals[2]', curr_normals[:, 2].min(), curr_normals[:, 2].max(), curr_normals[:, 2].mean())
#     # print('curr_normals', curr_normals.shape)
#     curr_lat_deg = np.rad2deg(np.arcsin(curr_normals[:, 1]))
#     curr_long_deg = np.rad2deg(np.arctan2(curr_normals[:, 0], curr_normals[:, 2]))
#    #print('curr_lat_deg', curr_lat_deg.shape, curr_lat_deg.min(), curr_lat_deg.max())
#    #print('curr_long_deg', curr_long_deg.shape, curr_long_deg.min(), curr_long_deg.max())
#     max_l = 1
#     out = _SHTOOLS.SHExpandLSQ(
#         d=curr_values, lat=curr_lat_deg, lon=curr_long_deg, lmax=max_l, norm=4
#     )
# 
#     reflected_spharm = spharm.SphericalHarmonic(envmap.EnvironmentMap(np.zeros((*ENV_DIM, estimated_lambertian_shading_gray_smart.shape[2])), 'latlong').data, norm=4, max_l=max_l)
#     reflected_spharm.coeffs = [out[1]]
#     
#     skylibs_pos = reflected_spharm.coeffs[0]
#     skylibs_pos = (
#         -coeffs[1, 1, 1], # x
#         coeffs[0, 1, 0], # y
#         coeffs[0, 1, 1] # z
#     )
# 
#     out = torch.tensor([skylibs_pos[0], 0.0, skylibs_pos[2]], dtype=torch.float32)
#     out = out / torch.norm(out)
#     return out

def copy_make_border(input, top, bottom, left, right, value):
    """
    Pad a tensor similar to cv2.copyMakeBorder in OpenCV.

    Args:
        input (torch.Tensor): Input tensor.
        top (int): Number of rows of padding to add on the top.
        bottom (int): Number of rows of padding to add on the bottom.
        left (int): Number of columns of padding to add on the left.
        right (int): Number of columns of padding to add on the right.
        value (float, optional): Value to fill the padding with. Default is 0.

    Returns:
        torch.Tensor: Padded tensor.
    """
    output_r = torch.full((input.shape[0], 1, input.shape[2] + top + bottom, input.shape[3] + left + right), value[0], device=input.device, dtype=input.dtype)
    output_g = torch.full((input.shape[0], 1, input.shape[2] + top + bottom, input.shape[3] + left + right), value[1], device=input.device, dtype=input.dtype)
    output_b = torch.full((input.shape[0], 1, input.shape[2] + top + bottom, input.shape[3] + left + right), value[2], device=input.device, dtype=input.dtype)
    output = torch.cat([output_r, output_g, output_b], dim=1)

    output[:, :, top:top + input.shape[2], left:left + input.shape[3]] = input
    return output


class Metric3D:
    def __init__(self, device) -> None:
        model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)
        model.eval()
        model.requires_grad_(requires_grad=False)
        self.model = model.to(device)

    def __call__(self, img: torch.Tensor):
        # Rescale to 0-255
        rgb_origin = img * 255.0

        # prepare data
        intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]
        input_size = (616, 1064)  # for vit model

        h, w = rgb_origin.shape[2:]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = tf.resize(rgb_origin, (int(h * scale), int(w * scale)), interpolation=tf.InterpolationMode.BILINEAR, antialias=True)
        # remember to scale intrinsic, hold depth
        intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[2:]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2

        rgb = copy_make_border(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        # normalize
        mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32, device=rgb.device)[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32, device=rgb.device)[:, None, None]
        rgb = torch.div((rgb - mean), std)

        # inference
        with torch.no_grad():
            pred_depth, confidence, output_dict = self.model.inference({'input': rgb})

        # process depth: unpad, upsample to original size
        pred_depth = pred_depth[:, :, pad_info[0]: pred_depth.shape[2] - pad_info[1], pad_info[2]: pred_depth.shape[3] - pad_info[3]]
        pred_depth = F.interpolate(pred_depth, rgb_origin.shape[2:], mode='bilinear')

        # process normal
        pred_normal = output_dict['normal_out_list'][0][:, :3, :, :]
        pred_normal = pred_normal[:, :, pad_info[0]: pred_normal.shape[2] - pad_info[1], pad_info[2]: pred_normal.shape[3] - pad_info[3]]
        pred_normal = F.interpolate(pred_normal, rgb_origin.shape[2:], mode='bilinear')
        pred_normal = sdi_utils.omnidata_normal_to_openrooms_normal(pred_normal)

        return pred_depth, pred_normal


class MaterialDiffusion:
    def __init__(self, device, dtype=torch.float16) -> None:
        with hydra.initialize(config_path="configs"):
            cfg = hydra.compose(config_name="stage/material_diffusion", return_hydra_config=True)
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.model = self._load_model(cfg.model)

    def _get_device(self, device):
        if device == "auto":
            return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return device

    def _load_model(self, model_cfg):
        config = OmegaConf.load(model_cfg.config_path)
        model = IntrinsicImageDiffusion(unet_config=config.unet_config,
                                        diffusion_config=config.diffusion_config,
                                        ddim_config=config.ddim_config)

        ckpt = torch.load(model_cfg.ckpt_path)
        model.load_state_dict(ckpt)
        model = model.to(self.device, dtype=self.dtype)
        model.eval()
        return model

    @torch.inference_mode()
    def predict_materials(self, img, num_samples, sampling_batch_size=1, original_size=None):
        # Run model
        preds = []
        for _ in range(num_samples // sampling_batch_size):
            preds.append(
                self.model.sample(batch_size=sampling_batch_size,  # If more VRAM is available, can increase this number
                                  conditioning_img=img.to(self.model.device)))
        assert len(preds) > 0, "No samples were generated"
        preds = torch.cat(preds, dim=0)

        # Resize the output to the original size
        if original_size is not None:
            preds = tf.resize(preds, original_size, interpolation=tf.InterpolationMode.BILINEAR)
        preds = preds.mean(0)

        return preds

    @torch.inference_mode()
    def __call__(self, img: torch.Tensor):
        original_size = img.shape[-2:]

        with torch.autocast(device_type='cuda', dtype=self.dtype):
            preds = self.predict_materials(img,
                                           num_samples=self.cfg.model.num_samples,
                                           sampling_batch_size=self.cfg.model.sampling_batch_size,
                                           original_size=original_size)

        return preds


class ToControlNetInput(object):
    def __init__(self, *,
                 device,
                 tokenizer,
                 feed_empty_prompt=True,
                 blip_processor=None, blip_model=None,
                 for_sdxl=False, sdxl_tokenizers=None, sdxl_encoders=None) -> None:
        self.device = device
        self.feed_empty_prompt = feed_empty_prompt

        self.tokenizer = tokenizer
        self.blip_processor = blip_processor
        self.blip_model = blip_model

        self.for_sdxl = for_sdxl
        # self.sdxl_tokenizers = sdxl_tokenizers
        # self.sdxl_encoders = sdxl_encoders

    def _compute_caption(self, image):
        pil_img = Image.fromarray(np.uint8(tensor_image_to_numpy(image) * 255)).convert('RGB')
        blip_inputs = self.blip_processor(pil_img, return_tensors="pt").to(self.device)
        caption = self.blip_processor.decode(self.blip_model.generate(**blip_inputs)[0], skip_special_tokens=True)
        return caption

    def _tokenize_captions(self, captions):
        assert not self.for_sdxl
        return self.tokenizer(captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids

    def __call__(self, sample):
        img = sample['pixel_values']

        # make sure all inputs are in [0, 1]
        # assert img.min() >= 0 and img.max() <= 1

        # compute BLIP caption on [0, 1] scaled image
        caption = self._compute_caption(img) if not self.feed_empty_prompt else ""

        # destination_composite is the represents the image that CN is trying to denoise
        # SD's VAE works on images of range [-1, 1]
        # input_ids = self._tokenize_captions(caption)[0]
        # input_ids = input_ids.cpu()
        # torch.save(input_ids, 'data/input_ids.pt')
        sample.update({
            'input_ids': self._tokenize_captions(caption)[0],
            'caption': caption
        })

        return sample


# Predict intrinsics from the background image
class ToPredictors(object):
    def __init__(self,
                 device,
                 scale_dst_to_minus_one_to_one,
                 cutout=None,
                 fill_value=-1,
                 conditioning_maps=['depth', 'normal', 'diffuse', 'shading', 'mask'],
                 predictor_names=['depthanythingv2_metric', 'omnidata', 'dfnet'],
                 *, light_parametrization=None) -> None:
        self.device = device
        self.scale_dst_to_minus_one_to_one = scale_dst_to_minus_one_to_one

        # Rebuttal
        # Select from: zoedepth, depthanything, metric3d
        self.depth_model = predictor_names[0]

        # Select from omnidata, metric3d
        self.normal_model = predictor_names[1]

        # Select from dfnet, material_diffusion, precompute
        self.diffuse_model = predictor_names[2]

        if self.depth_model == 'metric3d' or self.normal_model == 'metric3d':
            self.dpnm_model = Metric3D(self.device)
        if self.depth_model == 'zoedepth':
            self.dp_model = handle_zoedepth().to(self.device)
        if self.depth_model == 'depthanything':
            self.dp_model = handle_depth_anything().to(self.device)
        if self.depth_model == 'depthanythingv2_relative':
            self.dp_model = handle_depth_anything_v2_relative().to(self.device)
        if self.depth_model == 'depthanythingv2_metric':
            self.dp_model = handle_depth_anything_v2_metric().to(self.device)

        if self.normal_model == 'omnidata':
            self.nm_model, self.nm_input_transform = handle_omnidata_normal()
            self.nm_model = self.nm_model.to(self.device)
        if self.normal_model == 'stablenormal':
            self.nm_model = handle_stable_normal(self.device)

        if self.diffuse_model == 'dfnet':
            self.df_model = handle_dfnet_w_depth_normal().to(self.device)
        elif self.diffuse_model == 'material_diffusion':
            self.df_model = handle_material_diffusion().to(self.device) # TODO: fix this?
        elif self.diffuse_model == 'precompute':
            self.df_model = None

        self.cutout = cutout
        self.fill_value = fill_value

        self.conditioning_maps = conditioning_maps

        self.light_parametrization = light_parametrization

    def __call__(self, sample, *, override_image_for_shading_computation=None):
        img = sample['pixel_values']

        # Rebuttal
        if self.depth_model == 'precompute':
            if 'bg_depth' in sample and sample['bg_depth'] is not None:
                depth = sample['bg_depth']
            elif 'depth' in sample and sample['depth'] is not None:
                depth = sample['depth']
            else:
                raise ValueError('bg_depth and depth is not in sample')
        elif self.depth_model == 'metric3d':
            depth = self.dpnm_model(img)[0]
        elif self.depth_model == 'zoedepth':
            depth = self.dp_model(img)['metric_depth']
            depth = F.interpolate(depth, size=img.shape[2:], mode='bilinear', align_corners=True)
        elif self.depth_model == 'depthanything':
            depth = self.dp_model.infer(img, pad_input=True, with_flip_aug=True)
        elif self.depth_model == 'depthanythingv2_relative':
            depth = self.dp_model.infer_tensor(img)
        elif self.depth_model == 'depthanythingv2_metric':
            depth = self.dp_model.infer_tensor(img)

        if self.normal_model == 'precompute_stablenormal':
            if 'bg_normal' in sample and sample['bg_normal'] is not None:
                normal = sample['bg_normal']
                normal = sdi_utils.stablenormal_normal_to_openrooms_normal(normal)
            elif 'normal' in sample and sample['normal'] is not None:
                normal = sample['normal']
            else:
                raise ValueError('bg_normal is not in sample')
        elif self.normal_model == 'metric3d':
            normal = self.dpnm_model(img)[1]
        elif self.normal_model == 'omnidata':
            normal = self.nm_model(self.nm_input_transform(img)).clamp(0, 1)
            normal = F.interpolate(normal, size=img.shape[2:], mode='bilinear', align_corners=True)
            normal = sdi_utils.omnidata_normal_to_openrooms_normal(normal)
        elif self.normal_model == 'stablenormal':
            default_stablenormal_resolution = 768
            input_image = F.interpolate(img, (default_stablenormal_resolution, default_stablenormal_resolution), mode='bilinear', align_corners=True)
            normal = self.nm_model(
                input_image,
                match_input_resolution=False,
                processing_resolution=default_stablenormal_resolution,
                output_type='pt'
            ).prediction

            normal = (normal + 1) / 2
            normal = F.interpolate(normal, size=img.shape[2:], mode='bilinear', align_corners=True)
            normal = sdi_utils.stablenormal_normal_to_openrooms_normal(normal)
            normal = normal.to(dtype=img.dtype)

        if self.diffuse_model == 'precompute':
            if 'bg_diffuse' in sample and sample['bg_diffuse'] is not None:
                diffuse = sample['bg_diffuse']
            elif 'diffuse' in sample and sample['diffuse'] is not None:
                diffuse = sample['diffuse']
            else:
                raise ValueError('bg_diffuse and diffuse is not in sample')
        else:
            if self.diffuse_model == 'dfnet':
                diffuse = self.df_model(torch.cat([img, depth, normal], dim=1))
                diffuse = torch.clamp(diffuse, 1e-5, 1)
                # diffuse_zero_mask = diffuse < 1e-3
                # diffuse[diffuse_zero_mask] = 1e-3
            elif self.diffuse_model == 'material_diffusion':
                raise NotImplementedError('Material Diffusion is not implemented yet')

        # Use GMNet to predict diffuse and shading
        # _, _, diffuse, _ = self.df_model(img)

        shading = compute_shading(override_image_for_shading_computation if override_image_for_shading_computation is not None else img, diffuse)

        mask = torch.ones((img.shape[0], 1, img.shape[2], img.shape[3]), dtype=torch.float32, device=self.device)
        if self.cutout is not None:
            shading = self.cutout(shading)
            mask[shading[:, 0:1, :, :] == self.fill_value] = self.fill_value

        if self.light_parametrization == 'light_bounding_box_2d' or self.light_parametrization == 'light_bounding_box_3d':
            dominant_light_center = sample['dominant_light_center']
        elif self.light_parametrization == 'spherical_harmonics_2d' or self.light_parametrization == 'spherical_harmonics_3d':
            dominant_light_center = sample['dominant_light_center']
            assert dominant_light_center.shape == (img.shape[0], 3)
            
        else:
            dominant_light_center = None

        if self.scale_dst_to_minus_one_to_one:
            img = (img - 0.5) * 2

        all_controlnet_inputs = {
            'depth': depth,
            'normal': normal,
            'diffuse': diffuse,
            'shading': shading,
            'mask': mask
        }
        if 'roughness' in sample:
            all_controlnet_inputs['roughness'] = sample['roughness']
        elif 'bg_roughness' in sample:
            all_controlnet_inputs['roughness'] = sample['bg_roughness']
        
        if 'metallic' in sample:
            all_controlnet_inputs['metallic'] = sample['metallic']
        elif 'bg_metallic' in sample:
            all_controlnet_inputs['metallic'] = sample['bg_metallic']
        
        controlnet_inputs = {k: v for k, v in all_controlnet_inputs.items() if k in self.conditioning_maps}

        conditioning_pixel_values = torch.cat([v for _, v in controlnet_inputs.items()], dim=1)

        if 'input_ids' not in sample:
            sample['input_ids'] = torch.load('data/input_ids.pt').to(self.device).expand(img.shape[0], -1)
            sample['caption'] = ["" for _ in range(img.shape[0])]
        out = {
            'conditioning_pixel_values': conditioning_pixel_values,
            'controlnet_inputs': controlnet_inputs,
            'pixel_values': img,
            'input_ids': sample['input_ids'],
            'caption': sample['caption'],
            'name': sample['name'] if 'name' in sample else False,
            'dominant_light_center': dominant_light_center,
        }

        return out


class ToPredictorsWithoutEstim(object):
    def __init__(self,
                 device,
                 scale_dst_to_minus_one_to_one,
                 cutout=None,
                 cutout_diffuse=None,
                 fill_value=-1,
                 conditioning_maps=['depth', 'normal', 'diffuse', 'shading', 'mask'],
                 *, light_parametrization=None) -> None:
        self.device = device
        self.scale_dst_to_minus_one_to_one = scale_dst_to_minus_one_to_one

        self.cutout = cutout
        self.cutout_diffuse = cutout_diffuse
        self.fill_value = fill_value

        self.conditioning_maps = conditioning_maps

        self.light_parametrization = light_parametrization


    def __call__(self, sample):
        img = sample['pixel_values']
        # make sure all inputs are in [0, 1]
        assert img.min() >= 0 and img.max() <= 1

        depth = sample['depth']
        normal = sample['normal']
        diffuse = sample['diffuse']

        if 'shading' in sample:
            shading = sample['shading']
        else:
            shading = compute_shading(img, diffuse)

        # Coarse dropout shading
        mask = torch.ones((img.shape[0], 1, img.shape[2], img.shape[3]), dtype=torch.float32, device=self.device)

        
        if self.light_parametrization == 'light_bounding_box_2d' or self.light_parametrization == 'light_bounding_box_3d':
            dominant_light_center = sample['dominant_light_center']
        elif self.light_parametrization == 'spherical_harmonics_2d' or self.light_parametrization == 'spherical_harmonics_3d':
            dominant_light_center = sample['dominant_light_center']
            assert dominant_light_center.shape == (img.shape[0], 3)
        else:
            dominant_light_center = None
            
        if self.cutout is not None:
            shading = self.cutout(shading)
            mask[shading[:, 0:1, :, :] == self.fill_value] = self.fill_value

        # Randomly replace diffuse with rgb
        if self.cutout_diffuse is not None:
            diffuse = self.cutout_diffuse(diffuse)
            diffuse[diffuse == self.fill_value] = img[diffuse == self.fill_value]

        # Also add masked background as input
        masked_bg = img.clone()
        masked_bg[mask.expand_as(masked_bg) == self.fill_value] = self.fill_value

        if self.scale_dst_to_minus_one_to_one:
            img = (img - 0.5) * 2

        controlnet_inputs = {
            'depth': depth,
            'normal': normal,
            'diffuse': diffuse,
            'shading': shading,
            'mask': mask,
        }
        if 'roughness' in self.conditioning_maps:
            controlnet_inputs['roughness'] = sample['roughness']
        if 'metallic' in self.conditioning_maps:
            controlnet_inputs['metallic'] = sample['metallic']

        conditioning_pixel_values = torch.cat([v for _, v in controlnet_inputs.items()], dim=1)

        # Valid mask
        if 'depth_valid_mask' in sample:
            controlnet_inputs['depth_valid_mask'] = sample['depth_valid_mask']
        if 'shading_valid_mask' in sample:
            controlnet_inputs['shading_valid_mask'] = sample['shading_valid_mask']

        out = {
            'conditioning_pixel_values': conditioning_pixel_values,
            'controlnet_inputs': controlnet_inputs,
            #'dominant_light_info': sample['dominant_light_info'],
            'dominant_light_center': dominant_light_center,
            'pixel_values': img,
            'input_ids': sample['input_ids'],
            'caption': sample['caption'],
            'name': sample['name'] if 'name' in sample else False
        }

        return out


def collate_fn(examples):
    stacked_examples = {}
    for k, _ in examples[0].items():
        if k == 'input_ids':
            stacked_examples[k] = torch.stack([example[k] for example in examples])
        elif k == 'caption' or k == 'name':
            stacked_examples[k] = [example[k] for example in examples]
        elif k == 'dominant_light_info':
            stacked_examples[k] = [example[k] for example in examples]
        elif k == 'dominant_light_center':
            stacked_examples[k] = torch.stack([example[k] for example in examples])
        elif k == 'dominant_light_center':
            stacked_examples[k] = torch.stack([example[k] for example in examples])
        else:
            batched = torch.stack([example[k] for example in examples])
            stacked_examples[k] = batched.to(memory_format=torch.contiguous_format)

    return stacked_examples


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'D:/repos/sd_intrinsics/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6',
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )

    cutout = CoarseDropout(max_holes=3, max_height=0.7, max_width=0.7,
                           min_holes=1, min_height=0.05, min_width=0.05,
                           fill_value=-1, p=0.7, fully_drop_p=0.3,
                           max_circles=3, min_circles=1, max_radius=0.3, min_radius=0.05, p_circle=0.7)

    to_predictors = ToPredictors('cuda:0', True, cutout=cutout, predictor_names=['depthanything_v2_metric', 'omnidata', 'dfnet'])
    # to_predictors = ToPredictorsOpenrooms('cuda:0', True, cutout=cutout)
    # to_predictors = ToPredictorsWithoutEstimCutoutIntrinsics('cuda:0', True, cutout=cutout)

    img1 = torch.from_numpy(np.array(Image.open("test/composite0001.jpg"), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to('cuda:0')
    img2 = torch.from_numpy(np.array(Image.open("test/composite0002.jpg"), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to('cuda:0')
    input = {'pixel_values': torch.cat([img1, img2], dim=0),
             'input_ids': '',
             'caption': ''}
    out = to_predictors(input)
    controlnet_inputs = out['controlnet_inputs']
    for k, v in controlnet_inputs.items():
        if k == 'depth':
            v = sdi_utils.tensor_to_pil_list(v, [v.min(), v.max()])
        else:
            v = sdi_utils.tensor_to_pil_list(v)
        for i, img in enumerate(v):
            img.save(f'test/{k}_{i}.png')