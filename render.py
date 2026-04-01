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
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.image_utils import apply_depth_colormap
import numpy as np
from omegaconf import DictConfig
from scene.R3GW_model import R3GW
import hydra
from eval_with_gt_envmaps import process_environment_map_image
import glob
import spaudiopy

@torch.no_grad()
def render_test_with_gt_envmaps(source_path, model_path, iteration, views, model, background, sky_sh_degree, specular):
    render_path = os.path.join(model_path, "test", "iteration_{}".format(iteration), "renders_with_gt_envmaps")
    gt_path = os.path.join(model_path, "test", "iteration_{}".format(iteration), "gts")

    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)
 
    # Envmaps params
    if "st" in source_path:
        scale = 30
    elif "lwp" in source_path:
        scale = 20
    else:
        scale = 10
    threshold = 0.99

    for _, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()

        if "_DSC" in view.image_name:
                lighting_condition = view.image_name.split("_DSC")[0]
        else:
                lighting_condition = view.image_name.split("_IMG")[0]
        envmap_folder_path = os.path.join(source_path, "eval_files", "ENV_MAP_CC", lighting_condition)
        envmap_img_path =  glob.glob(os.path.join(envmap_folder_path, '*.jpg'))
        if len(envmap_img_path) == 0:
            continue
        #TODO: update below
        envmap_img_path = [fname for fname in envmap_img_path if "SH" not in fname and "rotated" not in fname][0]
        envlight_sh = process_environment_map_image(envmap_img_path, scale, threshold)
        # Rotate envmap SH around x axis, sun direction is not adjusted here !!!! (see evaluation code for that)
        envlight_sh = spaudiopy.sph.rotate_sh(envlight_sh.T, 0, -np.pi/2, 0, 'real')
        envlight_sh = torch.tensor(envlight_sh.T, dtype=torch.float32, device="cuda")
        gt = view.original_image.cuda()
        model.envlight.set_base(envlight_sh)
        # Fix sky color
        sky_sh = torch.zeros((9,3), dtype=torch.float32, device="cuda")

        render_pkg = render(view, model.gaussians, model.envlight, sky_sh, sky_sh_degree,
                            background, debug=False, fix_sky=True, specular=specular, normal_view=True)
        render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)

        for k in render_pkg.keys():
            if render_pkg[k].dim() < 3 or k=="render" or k=="delta_normal_norm" or k == "normal_ref" or k == "alpha":
                continue
            save_path = os.path.join(model_path, "test", "iteration_{}".format(iteration), k)
            makedirs(save_path, exist_ok=True)
            if k == "albedo":
                render_pkg[k] = torch.clamp(render_pkg[k], 0.0, 1.0)
            if k == "depth":
                render_pkg[k] = apply_depth_colormap(-render_pkg["depth"][0][...,None]).permute(2,0,1)
            elif "normal" in k:
                render_pkg[k] = 0.5 + 0.5 * render_pkg[k]
            torchvision.utils.save_image(render_pkg[k], os.path.join(save_path, view.image_name + ".png"))

        torch.cuda.synchronize()

        gt = gt[0:3, :, :]
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gt_path, view.image_name + ".png"))

@torch.no_grad()
def render_train(model_path, iteration, views, model, background, sky_sh_degree, fix_sky, specular):
    name = "train"
    render_path = os.path.join(model_path, name, "iteration_{}".format(iteration), "renders")
    gt_path = os.path.join(model_path, name, "iteration_{}".format(iteration), "gts")
    lighting_path = os.path.join(model_path, name, "iteration_{}".format(iteration), "rendered_envlights")
    sky_map_path = os.path.join(model_path, name, "iteration_{}".format(iteration), "rendered_sky_maps")

    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)
    makedirs(lighting_path, exist_ok=True)
    makedirs(sky_map_path, exist_ok=True)

    for view in tqdm(views, desc="Rendering progress"):
        torch.cuda.synchronize()

        view_id = torch.tensor([view.uid], device ='cuda')
        gt = view.original_image.cuda()
        embedding_gt = model.embeddings(view_id)
        envlight_sh, sky_sh = model.mlp(embedding_gt)
        model.envlight.set_base(envlight_sh)

        render_pkg = render(view, model.gaussians, model.envlight, sky_sh, sky_sh_degree,
                            background, debug=False, fix_sky=fix_sky, specular=specular, normal_view=True)
        render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)

        torch.cuda.synchronize()

        gt = gt[0:3, :, :]
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gt_path, view.image_name + ".png"))

        for k in render_pkg.keys():
            if render_pkg[k].dim() < 3 or k=="render" or k=="delta_normal_norm" or k == "normal_ref" or k == "alpha":
                continue
            save_path = os.path.join(model_path, name, "iteration_{}".format(iteration), k)
            makedirs(save_path, exist_ok=True)
            if k == "albedo":
                render_pkg[k] = torch.clamp(render_pkg[k], 0.0, 1.0)
            if k == "depth":
                render_pkg[k] = apply_depth_colormap(-render_pkg["depth"][0][...,None]).permute(2,0,1)
            elif "normal" in k:
                render_pkg[k] = 0.5 + 0.5 * render_pkg[k]
            torchvision.utils.save_image(render_pkg[k], os.path.join(save_path, view.image_name + ".png"))

    print(f"{name}- rendering illuminations")
    model.render_envlights_sh_all(save_path=lighting_path, save_sh_coeffs=True)
    model.render_sky_sh_all(save_path=sky_map_path, save_sh_coeffs=True)

@torch.no_grad()
def render_sets(cfg):
    # Load model
    model = R3GW(cfg, load_chkpt_iteration=cfg.render.iteration)
    iteration = model.load_chkpt_iteration

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not cfg.render.skip_train:
            render_train(cfg.dataset.model_path, iteration, model.scene.getTrainCameras(),
                       model, background, cfg.sky_sh_degree, cfg.fix_sky, cfg.specular)

    if not cfg.render.skip_test:
        if cfg.render.render_with_gt_envmaps:
            render_test_with_gt_envmaps(cfg.dataset.source_path,cfg.dataset.model_path, iteration, model.scene.getTestCameras(),
                                        model, background, cfg.sky_sh_degree, cfg.specular)

@hydra.main(version_base=None, config_path="configs", config_name="R3GW")
def main(cfg: DictConfig):

    print("Rendering " + cfg.dataset.model_path)

    # Initialize system state (RNG)
    safe_state(cfg.run.quiet)

    cfg.dataset.eval = True
    # Render
    render_sets(cfg)
    # All done
    print("\nRendering complete.")


if __name__ == "__main__":
    main()
