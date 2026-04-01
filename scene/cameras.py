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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, get_rays, getView2World

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda",
                 sky_mask=None, occluders_mask=None, normal_prior=None,
                 normal_image=None, cx=None, cy=None, image_w=None, image_h=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.cx = cx 
        self.cy = cy
        self.image_name = image_name
        self.image_width = image_w
        self.image_height = image_h

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        if normal_image is not None:
            self.original_normal_image = normal_image.clamp(0.0, 1.0).to(self.data_device)
            original_normal_image = self.original_normal_image * 2 - 1. # (0, 1) -> (-1, 1)
            original_normal_image = torch.nn.functional.normalize(original_normal_image, p=2, dim=0)
            fg_mask = torch.where((self.original_normal_image==0.).sum(0)<3, True, False)[None,...].repeat(3, 1, 1)
            original_normal_image = torch.where(fg_mask, original_normal_image, torch.ones_like(original_normal_image))            
            self.original_normal_image = original_normal_image
        else:
            self.original_normal_image = None
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.gt_alpha_mask = gt_alpha_mask
        self.sky_mask = sky_mask
        self.occluders_mask = occluders_mask
        self.normal_prior = normal_prior

        # if gt_alpha_mask is not None:
        if False:
            self.original_image *= gt_alpha_mask.to(self.data_device)
            if self.original_normal_image is not None:
                self.original_normal_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            if self.original_normal_image is not None:
                    self.original_normal_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.view_world_transform = torch.tensor(getView2World(R, T)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


    def get_calib_matrix_nerf(self, scale=1.0):
        Fx = fov2focal(self.FoVx, self.image_width)
        Fy = fov2focal(self.FoVy, self.image_height)
        Cx = 0.5 * self.image_width
        Cy = 0.5 * self.image_height
        intrinsic_matrix = torch.tensor([[Fx/scale, 0, Cx/scale], [0, Fy/scale, Cy/scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix
    

    def get_rays(self):
        intrinsic_matrix, extrinsic_matrix = self.get_calib_matrix_nerf()

        viewdirs = get_rays(self.image_width, self.image_height, intrinsic_matrix, extrinsic_matrix[:3,:3])
        return viewdirs

    def project(self, xyz: torch.Tensor):
        """Project point in world coordinate system onto camera coordinates"""
        eps = torch.finfo(xyz.dtype).eps
        assert xyz.shape[-1] == 3
        # Translate and rotate
        T = torch.tensor(self.T, dtype=torch.float32, device='cuda')
        R = torch.tensor(self.R, dtype=torch.float32, device='cuda')
        uvw = xyz - T
        uvw = (R*uvw[...,:,None]).sum(-2)
        # Camera -> Camera distorted
        uv = torch.where(uvw[..., 2:] > eps, uvw[..., :2] / uvw[..., 2:], torch.zeros_like(uvw[..., :2]))
        x, y = torch.moveaxis(uv, -1, 0)
        # Camera distorted --> image
        x = self.FoVx*x + self.cx
        y = self.FoVy*y + self.cy
        return torch.stack((x,y), -1)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
