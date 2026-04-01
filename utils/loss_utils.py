#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import cv2
from torch.autograd import Variable
from math import exp
from utils.sh_utils import eval_sh
import numpy as np
from utils.general_utils import rand_hemisphere_dir
import random
import torch.nn.functional as F


TINY_NUMBER = 1e-6


def l1_loss(network_output, gt, mask=None):
    if mask is not None:
        num_pixels = mask.sum()
        if num_pixels == 0:
            return torch.tensor(0.0, device=network_output.device)
        assert mask.shape[0] == network_output.shape[0], "The mask must be expanded as the input images"
        return (torch.abs((network_output - gt) * mask).sum()) / num_pixels
    else:
        return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1_, img2_, window_size=11, size_average=True, mask=None):
    if mask is not None and torch.sum(mask) == 0:
        return 1
    img1 = img1_.squeeze(0)
    img2 = img2_.squeeze(0)
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask)


def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    img1 = img1[None,...]
    img2 = img2[None,...]
    mask = mask[None,...]
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        if mask is not None:
            if(len(mask.shape) > 4):
                mask = mask.squeeze(0)
            assert mask.shape[1] == img1.shape[1], "the mask must be expanded as the input images"
            return ((ssim_map * mask).sum()) / (torch.sum(mask == 1))
        return ssim_map.mean()
    else:
        if mask is not None:
            raise NotImplementedError
        return ssim_map.mean(1).mean(1).mean(1)


def normal_consistency_loss(rendered_normal: torch.Tensor,
                            normal_ref: torch.Tensor,
                            mask: torch.Tensor = None
                            ) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(rendered_normal)
   
    dot_product = (rendered_normal * normal_ref * mask).sum(dim=0)
    normal_consistency_loss = ((1 - dot_product)[None]).mean()

    return normal_consistency_loss


def tv_loss(gt_image: torch.Tensor,
            rendered_image: torch.Tensor,
            mask: torch.Tensor,
            ) -> torch.Tensor:
    """
    Masked total variation loss for rendered_image of shape CxHxW given the reference image gt_image of same shape.
    The input mask can have both shape CxHxW or 1xHxW.
    """
    if mask.shape[0] == gt_image.shape[0]:
        mask = mask.mean(dim=0, keepdim=True) #1xHxW
    rgb_grad_h = torch.exp(
        -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    rgb_grad_w = torch.exp(
        -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    mask_h = mask[:, 1:, :] * mask[:, :-1, :]
    mask_w = mask[:, :, 1:] * mask[:, :, :-1]
    tv_h = torch.pow(rendered_image[:, 1:, :] - rendered_image[:, :-1, :], 2)  # [C, H-1, W]
    tv_w = torch.pow(rendered_image[:, :, 1:] - rendered_image[:, :, :-1], 2)  # [C, H, W-1]
    tv_loss = (tv_h * rgb_grad_h * mask_h).sum() / (mask_h.sum() + TINY_NUMBER) \
        + (tv_w * rgb_grad_w * mask_w).sum() / (mask_w.sum() + TINY_NUMBER)

    return tv_loss


def edge_aware_tv_loss(gt_image: torch.Tensor,
                       rendered_image: torch.Tensor,
                       mask: torch.Tensor
                       ) -> torch.Tensor:
    """
    Masked total variation loss between rendered_image of shape CxHxW
    and reference image, gt_image, of same shape.
    The input mask can have both shape CxHxW or 1xHxW.
    """
    with torch.no_grad():
        img = gt_image.permute(1, 2, 0).data.clone().cpu().numpy()
        img = (img * 255.0).astype(np.uint8)
        edges_map = cv2.Canny(img, 20, 200)
        edges_map = 1 - torch.from_numpy(edges_map).float().to(gt_image.device)/255.0

    if mask.shape[0] == gt_image.shape[0]:
        mask = mask.mean(dim=0, keepdim=True) #1xHxW

    mask = mask * edges_map.unsqueeze(0)

    rgb_grad_h = torch.exp(
        -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    rgb_grad_w = torch.exp(
        -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    mask_h = mask[:, 1:, :] * mask[:, :-1, :]
    mask_w = mask[:, :, 1:] * mask[:, :, :-1]
    tv_h = torch.pow(rendered_image[:, 1:, :] - rendered_image[:, :-1, :], 2)  # [C, H-1, W]
    tv_w = torch.pow(rendered_image[:, :, 1:] - rendered_image[:, :, :-1], 2)  # [C, H, W-1]
    tv_loss = (tv_h * rgb_grad_h * mask_h).sum() / (mask_h.sum() + TINY_NUMBER) \
        + (tv_w * rgb_grad_w * mask_w).sum() / (mask_w.sum() + TINY_NUMBER)

    return tv_loss


def edge_aware_smoothing_depth_loss(gt_image: torch.Tensor,
                                    depth_map: torch.Tensor
                                    ) -> torch.Tensor:
    with torch.no_grad():
        img = gt_image.permute(1, 2, 0).data.clone().cpu().numpy()
        img = (img * 255.0).astype(np.uint8)
        edges_map = cv2.Canny(img, 20, 200)
        edges_map = 1 - torch.from_numpy(edges_map).float().to(gt_image.device) / 255.0 # the mask is 0 on th edges
        kernel = torch.ones((1,1,3,3), device=gt_image.device)
        # Exclude edges pixels
        neighbors_depth = F.conv2d(depth_map.unsqueeze(0).unsqueeze(0) * edges_map, kernel, padding=1)
        tot_pixels_per_neighbor = F.conv2d(edges_map.unsqueeze(0).unsqueeze(0), kernel, padding=1)
        avg_depth_per_neighbor = neighbors_depth / (tot_pixels_per_neighbor + 1e-8)
    tot_pixels = torch.sum(edges_map)
    return (((depth_map - avg_depth_per_neighbor.squeeze()) ** 2) * edges_map).sum() / (tot_pixels + 1e-8)


def sky_depth_loss(depth_map: torch.Tensor,
                   sky_mask: torch.Tensor,
                   gamma: float = 0.02) -> torch.Tensor:
    """
    The function computes the mean depth in no-sky region and sky region and compares the difference.
    The function is based on the rendered depth map.
    """
    nosky_mask = 1 - sky_mask
    n_sky_pixels = torch.sum(nosky_mask == 1)
    if n_sky_pixels == 0:
        return 0
    n_no_sky_pixels = torch.sum(sky_mask == 1)
    with torch.no_grad():
        mean_depth_no_sky = (depth_map[0] * sky_mask).sum() / n_no_sky_pixels
    sky_depth = depth_map[0] * nosky_mask
    mean_depth_sky = (sky_depth).sum() / n_sky_pixels
    loss = torch.exp(-gamma * (mean_depth_sky-mean_depth_no_sky))
    return mean_depth_sky.detach(), loss


def depth_loss_gaussians(gaussians,
                         camera,
                         visibility_filter,
                         gamma = 0.02
                         ):
    """
    The function compares the difference of the average depth of sky and non sky gaussians using an exponential function.
    """
    sky_gaussians_mask = gaussians.get_is_sky.squeeze()
    gaussians_depth = gaussians.get_depth(camera)
    sky_gaussians_depth = gaussians_depth[(sky_gaussians_mask) & (visibility_filter)]
    avg_depth_sky_gauss = torch.mean(sky_gaussians_depth)
    avg_depth_non_sky_gauss = torch.mean(gaussians_depth[(~sky_gaussians_mask) & (visibility_filter)]).detach()
    loss = torch.exp(-gamma * (avg_depth_sky_gauss - avg_depth_non_sky_gauss))
    return loss


def envlight_loss(envlight_sh: torch.Tensor,
                  sh_degree: int,
                  normals: torch.Tensor,
                  N_dirs: int = 1000,
                  normals_subset_size: int = 100
                  ) -> torch.Tensor:
    """
    Regularization on environment lighting coefficients: incoming light should belong to R+.
    The loss is computed on a random subset of the input normals. For each normal N random directions
    in the hemisphere around it are sampled, then the irradiance corresponding to such direction is computed
    and the minimum between its values and 0 is taken.
    Args:
        envlight: environment lighting SH coefficients,
        normals: normal vectors of shape [..., 3],
        N_dirs: number of viewing directions samples,
        normals_subset_size: number of normal samples
    """
    assert normals.shape[-1] == 3 , "error: normals must have size  [...,3]"

    if normals.shape[0] > normals_subset_size:
        normals_rand_subset = random.sample(range(0, normals_subset_size), normals_subset_size)
        normals = normals[normals_rand_subset]

    # generate N_dirs random viewing directions in the hemisphere centered in n for each n in normals
    rand_hemisphere_dirs = rand_hemisphere_dir(N_dirs, normals) # (..., N, 3)
    # evaluate SH coefficients of env light
    light = eval_sh(sh_degree, envlight_sh.transpose(0,1), rand_hemisphere_dirs)
    # extract negative values
    light = torch.minimum(light, torch.zeros_like(light))
    # average negative light values over number of viewing direction samples
    avg_light_per_normal = torch.mean(light, dim = 1)
    # average over normals
    avg_light = torch.mean(avg_light_per_normal, dim = 0)
    # take squared 2 norm
    envlight_loss = torch.mean((avg_light) ** 2)

    return envlight_loss


def envl_sh_loss(sh_env: torch.Tensor,
                 sh_degree: int,
                 N_samples: int=10
                 ) -> torch.Tensor:
    """
    Environment light regularization from LumiGauss https://github.com/joaxkal/lumigauss.
    """
    shs_view = sh_env.repeat(N_samples, 1, 1)
    view_dir_unnorm =torch.empty(shs_view.shape[0], 3, device=shs_view.device).uniform_(-1,1)
    view_dir = view_dir_unnorm / view_dir_unnorm.norm(dim=1, keepdim=True)
    envl_val = eval_sh(sh_degree, shs_view.transpose(1,2), view_dir)

    #Constrain illumination values to R+
    sh_envl_loss = penalize_outside_range(envl_val.view(-1), 0.0,torch.inf)
    
    return sh_envl_loss


def penalize_outside_range(tensor, lower_bound=0.0, upper_bound=1.0):
    """
    The function is taken from LumiGauss https://github.com/joaxkal/lumigauss.
    """
    error = 0
    below_lower_bound = tensor[tensor < lower_bound]
    above_upper_bound = tensor[tensor > upper_bound]
    if below_lower_bound.numel():
        error += torch.mean((below_lower_bound - lower_bound) ** 2)
    if above_upper_bound.numel():
        error += torch.mean((above_upper_bound - upper_bound) ** 2)
    return error


def min_scale_loss(radii, gaussians):
    visibility_filter = radii > 0
    try:
        if visibility_filter.sum() > 0: # consider just visible fg gaussians
            scale = gaussians.get_scaling[visibility_filter & ~gaussians.get_is_sky.squeeze()]
            if scale.numel() == 0:
                return torch.tensor(0.0, device=radii.device)
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0] # take minimum scales
            return min_scale_loss.mean()
    except Exception as e:
        raise RuntimeError(f"Failed to compute min_scale_loss: {e}")


def cam_depth2world_point(cam_z, pixel_idx, intrinsic, extrinsic):
    '''
    cam_z: (1, N)
    pixel_idx: (1, N, 2)
    intrinsic: (3, 3)
    extrinsic: (4, 4)
    world_xyz: (1, N, 3)
    '''
    valid_x = (pixel_idx[..., 0] + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0]
    valid_y = (pixel_idx[..., 1] + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1]
    ndc_xy = torch.stack([valid_x, valid_y], dim=-1)
    # inv_scale = torch.tensor([[W - 1, H - 1]], device=cam_z.device)
    # cam_xy = ndc_xy * inv_scale * cam_z[...,None]
    cam_xy = ndc_xy * cam_z[...,None]
    cam_xyz = torch.cat([cam_xy, cam_z[...,None]], dim=-1)
    world_xyz = torch.cat([cam_xyz, torch.ones_like(cam_xyz[...,0:1])], axis=-1) @ torch.inverse(extrinsic).transpose(0,1)
    world_xyz = world_xyz[...,:3]
    return world_xyz, cam_xyz

# From LumiGauss https://github.com/joaxkal/lumigauss

def img2mse(x, y, mask=None):
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        if mask.shape[0] == x.shape[0]:
            return torch.sum((x - y) * (x - y) * mask.unsqueeze(0)) / (torch.sum(mask) + TINY_NUMBER)
        else:
            return torch.sum((x - y) * (x - y) * mask.unsqueeze(0)) / (torch.sum(mask)*x.shape[0] + TINY_NUMBER)


def img2mae(x, y, mask=None):
    if mask is None:
        return torch.mean(torch.abs(x - y))
    else:
        if mask.shape[0] == x.shape[0]:
            return torch.sum(torch.abs(x - y) * mask.unsqueeze(0)) / (torch.sum(mask) + TINY_NUMBER)
        else:
            return torch.sum(torch.abs(x - y) * mask.unsqueeze(0)) / (torch.sum(mask) * x.shape[0] + TINY_NUMBER)


def mse2psnr(x):
    return -10. * torch.log(torch.tensor(x)+TINY_NUMBER) / torch.log(torch.tensor(10))


def img2mse_image(x, y, mask=None):
    
    # Compute squared difference per pixel per channel
    mse_image = (x - y) ** 2
    
    # If a mask is provided, apply the mask
    if mask is not None:
        # Ensure the mask is expanded to match the shape (3, W, H)
        mask = mask.unsqueeze(0)  # Add channel dimension (1, W, H) -> (3, W, H)
        mse_image = mse_image * mask

    return mse_image
