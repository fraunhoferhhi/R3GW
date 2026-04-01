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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import depth_to_normal
from scene.NVDIFFREC.util import safe_normalize
from scene.NVDIFFREC.light import EnvironmentLight
from typing import Optional, Tuple, Dict


def get_shaded_colors(envlight: EnvironmentLight,
                      pos: torch.Tensor,
                      view_pos: torch.Tensor,
                      normal: Optional[torch.Tensor],
                      albedo: Optional[torch.Tensor],
                      roughness: Optional[torch.Tensor],
                      metalness: Optional[torch.Tensor],
                      specular: bool = True
                      ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    colors_precomp, brdf_pkg = envlight.shade(gb_pos=pos[None, None, ...],
                                              gb_normal=normal[None, None, ...],
                                              albedo=albedo[None, None, ...],
                                              view_pos=view_pos[None, None, ...],
                                              kr=roughness[None, None, ...],
                                              km=None if metalness is None else metalness[None, None, ...],
                                              specular=specular
                                             )

    return colors_precomp, brdf_pkg


def normalize_normal(normal: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    # normal: (3, H, W), alpha: (H, W)
    fg_mask = (alpha[None,...] > 0.).repeat(3, 1, 1) & (normal > 1e-6)
    normal = torch.where(fg_mask, torch.nn.functional.normalize(normal, p=2, dim=0), normal)

    return normal


def render(viewpoint_camera,
           pc: GaussianModel,
           envlight: EnvironmentLight,
           sky_sh: torch.Tensor,
           sky_sh_degree: int,
           bg_color: torch.Tensor,
           debug: bool = True,
           specular: bool = True,
           fix_sky: bool = False,
           normal_view: bool = False
           ) -> Dict[str, torch.Tensor]:
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=-1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation

    sky_mask = viewpoint_camera.sky_mask.expand_as(viewpoint_camera.original_image).cuda()
    sky_gaussians_mask = pc.get_is_sky.squeeze() # (N)
    positions = pc.get_xyz # (N, 3)
    albedo = pc.get_albedo # (N, 3)
    roughness = pc.get_roughness # (N, 1)
    metalness = pc.get_metalness # (N,1) (if not None)

    # Render under camera viewpoint
    view_pos = viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1) # (N, 3) 
    dir_pp = positions - view_pos
    dir_pp_normalized = safe_normalize(dir_pp) # (N, 3)
    normal = pc.get_normal(dir_pp_normalized=dir_pp_normalized) # (N, 3)
    normal = normal[~sky_gaussians_mask]

    colors_precomp, fg_color, sky_color =  (torch.zeros(positions.shape[0], 3, dtype=torch.float32, device="cuda") for _ in range(3))

    # Compute color for the foreground Gaussians
    color_fg_gaussians, brdf_pkg = get_shaded_colors(envlight=envlight,
                                                     pos=positions[~sky_gaussians_mask],
                                                     view_pos=view_pos[~sky_gaussians_mask],
                                                     normal=normal,
                                                     albedo=albedo,
                                                     roughness=roughness,
                                                     metalness=metalness,
                                                     specular=specular
                                                    )
    colors_precomp[~sky_gaussians_mask] = color_fg_gaussians.squeeze()
    fg_color[~sky_gaussians_mask] = color_fg_gaussians.squeeze()

    # Compute color for the sky (background) Gaussians
    if fix_sky:
        colors_precomp[sky_gaussians_mask] = torch.ones_like(positions[sky_gaussians_mask])
    else:
        sky_sh2rgb = eval_sh(sky_sh_degree, sky_sh.transpose(1,2), dir_pp_normalized[sky_gaussians_mask])
        color_sky_gaussians = torch.clamp_min(sky_sh2rgb + 0.5, 0.0)
        colors_precomp[sky_gaussians_mask] = color_sky_gaussians
        sky_color[sky_gaussians_mask] = color_sky_gaussians

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(means3D = means3D,
                                       means2D = means2D,
                                       shs = None,
                                       colors_precomp = colors_precomp,
                                       opacities = opacity,
                                       scales = scales,
                                       rotations = rotations,
                                       cov3D_precomp = None
                                       )
    
    torch.cuda.empty_cache()

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    out = {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii, 
            }

    render_extras = {
                    "fg_color": fg_color,
                    "sky_color": sky_color,
                    "alpha": torch.ones_like(means3D) 
                    }

    # Render depth and normals
    # Get Gaussians depth as z coordinate of their position in camera space
    depth = pc.get_depth(viewpoint_camera)
    depth = depth.repeat(1,3)
    render_extras.update({"depth": depth})

    normal = 0.5 * normal + 0.5 # range (-1, 1) -> (0, 1)
    render_extras.update({"normal": normal})

    roughness_all = torch.zeros((positions.shape[0], 1), device="cuda", dtype=torch.float32)
    roughness_all[~sky_gaussians_mask] = pc.get_roughness
    albedo_all = torch.ones_like(positions)
    albedo_all[~sky_gaussians_mask] = pc.get_albedo
    render_extras.update({
                         "roughness": roughness_all.repeat(1, 3),
                         "albedo": albedo_all
                         })

    if debug:
        diffuse_color, specular_color = (torch.zeros(positions.shape[0], 3, dtype=torch.float32, device="cuda") for _ in range(2))
        diffuse_color[~sky_gaussians_mask] = brdf_pkg['diffuse'].squeeze()
        specular_color[~sky_gaussians_mask] = brdf_pkg['specular'].squeeze()

        render_extras.update({
                            "diffuse_color": diffuse_color,
                            "specular_color": specular_color
                            })

        if metalness is not None:
            metalness_all = torch.zeros((positions.shape[0], 1), device="cuda", dtype=torch.float32)
            metalness_all[~sky_gaussians_mask] = metalness
            render_extras.update({"metalness": metalness_all.repeat(1, 3)})

    out_extras = {}
    for k, extra in render_extras.items():
        if extra is None: 
            continue

        if k == "normal":
            image = rasterizer(means3D = means3D[~sky_gaussians_mask],
                               means2D = means2D[~sky_gaussians_mask],
                               shs = None,
                               colors_precomp = extra,
                               opacities = opacity[~sky_gaussians_mask],
                               scales = scales[~sky_gaussians_mask],
                               rotations = rotations[~sky_gaussians_mask],
                               cov3D_precomp = None
                               )[0]
            image = (image - 0.5) * 2. # range (0, 1) -> (-1, 1)
            if normal_view:
                out_extras["normal_view"] = (image.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3])).permute(2,0,1)
                out_extras["normal_view"] = - out_extras["normal_view"].clone()
                out_extras["normal_view"] = out_extras["normal_view"] * sky_mask + torch.ones_like(image) * (1 - sky_mask)
        else:
            colors_extra= extra.detach() if k in ("fg_color", "sky_color") else extra
            image = rasterizer(means3D = means3D,
                              means2D = means2D,
                              shs = None,
                              colors_precomp = colors_extra,
                              opacities = opacity,
                              scales = scales,
                              rotations = rotations,
                              cov3D_precomp = None
                              )[0]
        out_extras[k] = image

        torch.cuda.empty_cache()

    depth_mask = sky_mask > 0
    out_extras["depth"][depth_mask] = out_extras["depth"][depth_mask] / out_extras["alpha"][depth_mask].clamp_min(1e-6)

    # Get surface normal from depth map.
    out_extras["normal_ref"]  = depth_to_normal(viewpoint_camera, ((out_extras["depth"] * depth_mask)[0]).unsqueeze(0))
    out_extras["normal_ref"] = out_extras["normal_ref"].permute(2,0,1)
    out_extras["normal_ref"] = (out_extras["normal_ref"] * (out_extras["alpha"]).detach() * depth_mask)
    out.update(out_extras)

    return out
