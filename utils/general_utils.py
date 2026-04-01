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
import sys
from datetime import datetime
import numpy as np
import random
import os
import glob
from PIL import Image
from pathlib import Path
import math
from torch.utils.data import TensorDataset


def grad_thr_exp_scheduling(iter, max_iter, grad_thr_start, grad_thr_end=0.0004):
    return np.exp(np.log(grad_thr_start)*(1-iter/max_iter)+np.log(grad_thr_end)*(iter/max_iter))


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def PILtoTorch(pil_image, resolution, is_mask = False):
    if is_mask:
        vals = np.unique(np.array(pil_image))
        resized_image_PIL = pil_image.resize(resolution, resample=Image.Resampling.NEAREST)
        vals = np.unique(np.array(resized_image_PIL))
    else:
        resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_const_lr_func(const):
    def helper(step):
        return const

def get_lambda_lr_func(
    lr_init, lr_final, max_steps=1000000
):
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if step < max_steps:
            return lr_init
        else:
            return lr_final

    return helper


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """


    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def get_minimum_axis(scales, R):
    smallest_axis_idx = scales.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
    smallest_axis = R.gather(2, smallest_axis_idx)
    return smallest_axis.squeeze(dim=2)


def flip_align_view(normal, viewdir):
    # normal: (N, 3), viewdir: (N, 3)
    dotprod = torch.sum(
        normal * -viewdir, dim=-1, keepdims=True) # (N, 1)
    non_flip = dotprod >= 0 # (N, 1)
    normal_flipped = normal * torch.where(non_flip, 1, -1) # (N, 3)
    return normal_flipped, non_flip


def normalize_normal_inplace(normal, alpha):
    # normal: (3, H, W), alpha: (H, W)
    fg_mask = (alpha[None,...]>0.).repeat(3, 1, 1)
    normal = torch.where(fg_mask, torch.nn.functional.normalize(normal, p=2, dim=0), normal)

def get_homogeneous(points): # get 4D representation with 1 as w coord for set of points.
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def rand_hemisphere_dir(N: torch.Tensor, n: torch.Tensor):
    """
    Sample a cosine-weighted random direction on the unit hemisphere oriented around vector n.
    In case of multiple normal vectors stored in n, N samples are generated for each. The tensor storing 
    n is expected to have len(n.shape) == 2, where the first dimension refers to the number of input normal 
    vectors.
    Args:
        N: number of samples,
        n: normal vector of shape L x 3
    Returns:
        d (torch.Tensor): sampled cosine weighted direction of shape L x N x 3.
    """
    assert len(n.shape) == 2 and n.shape[-1] == 3 , "error: n must have size  L X 3"

    # sample N points on the unit sphere
    rand = torch.rand(n.shape[0], N, 3).cuda()
    normals = n.repeat(N, 1 , 1).transpose(0,1)
    phi = 2*np.pi*rand[...,1] # phi in [0, 2pi), shape N x n.shape[0]
    d = torch.zeros_like(normals)
    d[..., 0] = torch.cos(phi) * torch.sqrt(rand[...,0])
    d[..., 1] = torch.sin(phi) * torch.sqrt(rand[...,0])
    d[...,2] = torch.sqrt(1- torch.linalg.vector_norm(d, dim = -1)**2)
    # orient points around corresponding normal vectorſ
    tangent = torch.nn.functional.normalize(rand, dim=-1)
    bitangent = torch.linalg.cross(tangent, normals) # cross product along dim=-1
    d = tangent * d[...,0].unsqueeze(-1) + bitangent * d[...,1].unsqueeze(-1) + normals * d[...,2].unsqueeze(-1) 

    return d


def get_uniform_points_on_sphere_fibonacci(num_points, *, dtype=None, xnp=torch):
    # https://arxiv.org/pdf/0912.4540.pdf
    # Golden angle in radians
    if dtype is None:
        dtype = xnp.float32
    phi = math.pi * (3. - math.sqrt(5.))
    N = (num_points - 1) / 2
    i = xnp.linspace(-N, N, num_points, dtype=dtype)
    lat = xnp.arcsin(2.0 * i / (2*N+1))
    lon = phi * i

    # Spherical to cartesian
    x = xnp.cos(lon) * xnp.cos(lat)
    y = xnp.sin(lon) * xnp.cos(lat)
    z = xnp.sin(lat)
    return xnp.stack([x, y, z], -1)


def sample_points_on_unit_hemisphere(num_points, *, dtype=None, xnp=torch):
    if dtype is None:
        dtype = xnp.float32
    # Sample points on a portion of the unit hemisphere (COLMAP coordinates system)
    torch.manual_seed(0)
    y = - 0.5 * xnp.rand(num_points)
    theta = torch.acos(y)
    phi = xnp.pi*(1/2) * xnp.rand(num_points) - xnp.pi/4 # phi in [-pi/4, pi3/4]
    # Spherical to cartesian
    x = xnp.sin(phi) * xnp.sin(theta)
    z = xnp.sin(theta) * xnp.cos(phi)
    return xnp.stack([x, y, z], -1)


def load_npy_tensors(path: Path):
    """The function loads all npy tensors in path."""
    npy_tensors = {}
    npy_tensors_fnames = path.glob("*.npy")

    for npy_tensor_fname in npy_tensors_fnames:
        npy_tensor = np.load(npy_tensor_fname)
        npy_tensors[str(npy_tensor_fname)] = npy_tensor

    return npy_tensors

def get_half_images(img: torch.Tensor, left: bool = True):
    """The function return a TesnorDatset containing the specified vertical halfs of the input.
    Args:
        img : batch of images of shape (..., 3, H, W)
        left(bool): if True returns left vertical half
    Returns:
        left/right_img_half : imgs half cropped
    """
    if left: 
        left_half_images = img[..., :, :img.shape[2] // 2]
        return left_half_images
    else:
        right_half_images = img[..., :, img.shape[2] // 2:]
        return right_half_images


def insert_zeros(batch: torch.Tensor, zeros_idxs: torch.Tensor):
    """The function increases the dimension of the tensors contained in the input batch
    from n to n+1 by inserting a 0 in a specified position.
    Args:
        batch : batch of N tensors of shape n,
        zeros_idxs : positions, in {0,...,n}, where to add the 0 for each tensor.
    Returns:
        batch_out : augmented tensor. """
    batch_size = batch.shape[0]
    tensors_dim = batch.shape[1]
    batch_out = torch.zeros((batch_size, tensors_dim+1),dtype=batch.dtype, device="cuda")
    # Get indices from 0 to tensors_dim -1
    tensors_idxs = torch.arange(tensors_dim).expand(batch_size, -1).to("cuda")
    # Given the indices for tensors dimension establish whether the index where to insert the 0 comes before (True) or after (False) to handle shifting
    shift_idxs_mask =  tensors_idxs >= zeros_idxs.unsqueeze(1)
    shifted_idxs = tensors_idxs + shift_idxs_mask.int()
    batch_out[torch.arange(batch_size).unsqueeze(1).expand(batch_size, tensors_dim).reshape(-1), shifted_idxs.flatten()] = batch.flatten()
    return batch_out


def pinverse(A: torch.Tensor):
    # The function computes the pseudo-inverse of the input matrix A.
    return torch.linalg.pinv(A, rtol=1e-6)


def cartesian_to_polar_coord(
    xyz: torch.Tensor,
    center: torch.Tensor = torch.zeros(3, dtype=torch.float32, device="cuda"),
    radius: torch.Tensor | float = 1.0
) -> torch.Tensor:
    diff = xyz - center
    r = radius if torch.is_tensor(radius) else torch.tensor(radius, device=xyz.device, dtype=xyz.dtype)
    cos_theta = torch.clamp(-diff[..., 1] / r, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    phi = torch.atan2(diff[..., 0], diff[..., 2])
    return torch.stack((theta, phi), dim=-1)