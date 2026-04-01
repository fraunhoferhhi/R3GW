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
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, depth_loss_gaussians, min_scale_loss, envl_sh_loss, normal_consistency_loss
from gaussian_renderer import render
from utils.general_utils import safe_state, grad_thr_exp_scheduling
from utils.image_utils import apply_depth_colormap
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import Namespace
from scene.R3GW_model import R3GW
from omegaconf import DictConfig
import hydra
from utils.sh_utils import render_sh_map
from eval_with_gt_envmaps import evaluate_test_report


TINY_NUMBER = 1e-6


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(cfg):
    first_iter = 0
    tb_writer = prepare_output_and_logger(cfg.dataset)

    if cfg.train.start_checkpoint:
        checkpoint =  torch.load(cfg.train.start_checkpoint)
        model = R3GW(cfg, chkpt=checkpoint, training=True)
        first_iter = checkpoint["iteration"]
        grad_threshold = checkpoint["grad_threshold"]
    else:
        model = R3GW(cfg, training=True)
        model.training_setup()
    model.save_config()

    eval_mode = cfg.dataset.eval

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, cfg.optimizer.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, cfg.optimizer.iterations + 1): 
        iter_start.record()

        is_warm_up = iteration <= cfg.optimizer.warm_up_iters

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = model.scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        viewpoint_cam_id = torch.tensor([viewpoint_cam.uid], device = 'cuda')
        gt_image = viewpoint_cam.original_image.cuda()
        sky_mask = viewpoint_cam.sky_mask.expand_as(gt_image).cuda()
        occluders_mask = viewpoint_cam.occluders_mask.expand_as(gt_image).cuda()

        # Get SH coefficients of environment lighting for current training image
        embedding_gt_image = model.embeddings(viewpoint_cam_id)
        envlight_sh, sky_sh = model.mlp(embedding_gt_image)
        envlight_sh_rand_noise = torch.randn_like(envlight_sh) * 0.025
        # Get environment lighting sh for the current training image
        model.envlight.set_base(envlight_sh + envlight_sh_rand_noise)

        # For the first N iterations, disable specular color
        render_specular = cfg.specular and (iteration > cfg.optimizer.start_specular_from_iter)
        render_pkg = render(viewpoint_cam, model.gaussians, model.envlight, sky_sh, cfg.sky_sh_degree,
                            background, debug=False, specular=render_specular)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        fg_col_norm = render_pkg["fg_color"]
        sky_col_norm = render_pkg["sky_color"]

        # Loss

        loss = torch.tensor(0.0, device=gt_image.device)
        logs = {}
        
        # Reconstruction loss
        Ll1 = l1_loss(image, gt_image, mask=occluders_mask)
        Ssim = (1.0 - ssim(image, gt_image, mask=occluders_mask))
        rec_loss = Ll1 * (1 - cfg.optimizer.lambda_dssim) + cfg.optimizer.lambda_dssim * Ssim
        if not is_warm_up or cfg.optimizer.warm_up_iters == 0:
            loss += rec_loss
            logs.update({"Reconstruction loss": f"{rec_loss:.{7}f}"})

        # Foreground-sky separation loss
        if cfg.optimizer.lambda_sky_fg > 0:
           loss_sky_fg = l1_loss(fg_col_norm, torch.zeros_like(fg_col_norm), mask=1-sky_mask) \
                        + l1_loss(sky_col_norm, torch.zeros_like(sky_col_norm), mask=sky_mask)
           loss += cfg.optimizer.lambda_sky_fg * loss_sky_fg
           logs.update({"Fg-sky loss": f"{cfg.optimizer.lambda_sky_fg * loss_sky_fg:.{7}f}"})


        # Sky Gaussians depth regularization
        if (
            not is_warm_up
            and iteration > cfg.optimizer.reg_sky_gauss_depth_from_iter
            and cfg.optimizer.lambda_sky_gauss > 0
        ):
            depth_loss_sky_gauss = cfg.optimizer.lambda_sky_gauss * depth_loss_gaussians(
                model.gaussians,
                viewpoint_cam,
                visibility_filter)
            loss += depth_loss_sky_gauss
            logs.update({"Depth loss": f"{depth_loss_sky_gauss:.{5}f}"})

        # Normals and planar regularization
        if (
            not is_warm_up
            and iteration > cfg.optimizer.reg_normal_from_iter
            and iteration < cfg.optimizer.reg_normal_until_iter
            and cfg.optimizer.lambda_normal > 0
        ):
            rendered_normal = render_pkg["normal"]
            rendered_surf_normal = render_pkg["normal_ref"]
            normal_mask = ((rendered_surf_normal != 0).all(0)).unsqueeze(0)

            # Depth-normals supervision
            normal_depth_loss = normal_consistency_loss(
                rendered_normal,
                rendered_surf_normal,
                normal_mask * sky_mask * occluders_mask
                )
            normal_loss = cfg.optimizer.lambda_normal * normal_depth_loss

            # Prior normals supervision
            if iteration < cfg.optimizer.normal_prior_loss_until_iter:
                normal_prior = viewpoint_cam.normal_prior.cuda()
                # Transfrom normal prior to world space
                normal_prior = (normal_prior.permute(1,2,0) @ (viewpoint_cam.view_world_transform[:3,:3].T)).permute(2,0,1)
                normal_prior = - normal_prior
                normal_prior_loss = normal_consistency_loss(
                    rendered_normal,
                    normal_prior,
                    normal_mask * sky_mask * occluders_mask
                    )
                normal_loss +=  cfg.optimizer.lambda_normal_prior * normal_prior_loss
            loss += normal_loss
            logs.update({"Normal loss": f"{normal_loss:.{5}f}"})

            if cfg.optimizer.lambda_scale > 0:
                scale_loss = cfg.optimizer.lambda_scale * min_scale_loss(radii, model.gaussians)
                loss += scale_loss
                logs.update({"Scale loss": f"{scale_loss:.{5}f}"})

        # Envlight regularization
        if not is_warm_up and cfg.optimizer.lambda_envlight > 0:         
            envl_loss = envl_sh_loss(envlight_sh, cfg.envlight_sh_degree)
            loss += cfg.optimizer.lambda_envlight * envl_loss
            logs.update({"Envlight loss": f"{envl_loss:.{5}f}"})

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 100 == 0:
                progress_bar.set_postfix(logs)
                progress_bar.update(100)
            if iteration == cfg.optimizer.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            model.gaussians.max_radii2D[visibility_filter] = torch.max(model.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            # Log and save
            losses_extra = {}
            losses_extra['psnr'] = psnr(image*occluders_mask, gt_image*occluders_mask).mean()
            training_report(tb_writer, iteration, loss, losses_extra, l1_loss,
                            iter_start.elapsed_time(iter_end), cfg.train.test_iterations,
                            model, eval_mode, render, {"sky_sh_degree": cfg.sky_sh_degree,"background": background, "debug": True, "specular": render_specular})
            if iteration in cfg.train.checkpoint_iterations or iteration == cfg.optimizer.iterations:
                print(f" ITER: {iteration} saving checkpoint")
                if iteration > cfg.optimizer.densify_from_iter and iteration < cfg.optimizer.densify_until_iter:
                    # save densification grad threshold as well due to exponential scheduling
                    model.save_checkpoint(iteration, grad_threshold)
                else:
                    model.save_checkpoint(iteration)


            # Densification
            if iteration < cfg.optimizer.densify_until_iter:
                model.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration == cfg.optimizer.densify_from_iter:
                    grad_threshold = cfg.optimizer.densify_grad_threshold

                if iteration > cfg.optimizer.densify_from_iter and iteration % cfg.optimizer.densification_interval == 0:
                    prune_by_size = True if iteration > cfg.optimizer.opacity_reset_interval else False
                    model.gaussians.densify_and_prune(grad_threshold, 0.005, model.scene.cameras_extent, prune_by_size)
                    grad_threshold = grad_thr_exp_scheduling(iteration, cfg.optimizer.densify_until_iter, cfg.optimizer.densify_grad_threshold)

                if iteration % cfg.optimizer.opacity_reset_interval == 0 or (iteration == cfg.optimizer.densify_from_iter):
                    model.gaussians.reset_opacity()

            # Optimizer step
            if iteration < cfg.optimizer.iterations:
                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none = True)
                model.update_learning_rate(iteration)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if not args.logger:
        return tb_writer
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, loss, losses_extra, l1_loss, elapsed, testing_iterations, model: R3GW, eval_mode, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        for k in losses_extra.keys():
            tb_writer.add_scalar(f'train_loss_patches/{k}_loss', losses_extra[k].item(), iteration)

    # Report samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        train_config = [{'name': 'train', 'cameras' : [model.scene.getTrainCameras()[idx % len(model.scene.getTrainCameras())] for idx in range(5, 30, 5)]}]
        with torch.no_grad():
            for config in train_config:
                if config['cameras'] and len(config['cameras']) > 0:
                    images = []
                    gts = []
                    for idx, viewpoint in enumerate(config['cameras']):
                        gt_image = viewpoint.original_image.cuda()
                        viewpoint_cam_id = torch.tensor([viewpoint.uid], device = 'cuda')
                        embedding_gt_image = model.embeddings(viewpoint_cam_id)
                        envlight_sh, sky_sh = model.mlp(embedding_gt_image)
                        model.envlight.set_base(envlight_sh)
                        render_pkg = renderFunc(viewpoint, model.gaussians, model.envlight, sky_sh, renderArgs["sky_sh_degree"],
                                                renderArgs["background"], debug=renderArgs["debug"], specular=renderArgs["specular"])
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        images.append(image)
                        gt_image = torch.clamp(gt_image, 0.0, 1.0)
                        gts.append(gt_image)
                        reconstructed_envlight = model.envlight.render_sh().cuda().permute(2,0,1)
                        reconstructed_sky_map = render_sh_map(sky_sh.squeeze()).cuda().permute(2,0,1)
                        if tb_writer and (idx < 10):
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/reconstructed_envlight".format(viewpoint.image_name), reconstructed_envlight[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/reconstructed_sky_map".format(viewpoint.image_name), reconstructed_sky_map[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            for k in render_pkg.keys():
                                if (render_pkg[k].dim()<3 or k=="render" or k=="viewspace_points") and k != "depth":
                                    continue
                                if "diffuse" in k:
                                    image_k = render_pkg[k]
                                elif k == "depth":
                                    image_k = apply_depth_colormap(-render_pkg[k][0][...,None])
                                    image_k = image_k.permute(2,0,1)
                                elif k == "alpha":
                                    image_k = apply_depth_colormap(render_pkg[k][0][...,None], min=0., max=1.)
                                    image_k = image_k.permute(2,0,1)
                                else:
                                    if "normal" in k:
                                        render_pkg[k] = 0.5 + (0.5 * render_pkg[k]) # (-1, 1) -> (0, 1)
                                    image_k = torch.clamp(render_pkg[k], 0.0, 1.0)
                                tb_writer.add_images(config['name'] + "_view_{}/{}".format(viewpoint.image_name, k), image_k[None], global_step=iteration)

                    l1_losses = [l1_loss(image, gt) for image, gt in zip(images,gts)]
                    l1_train = torch.tensor(l1_losses).mean() 
                    psnrs = [psnr(image, gt) for image, gt in zip(images,gts)]
                    psnr_train = torch.mean(torch.stack(psnrs))
                    print("\n[ITER {}] Evaluating train : L1 {} PSNR {}".format(iteration, l1_train, psnr_train))
                    if tb_writer:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_train, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_train, iteration)

                    if eval_mode and model.config.dataset.test_config_path != "":
                        psnr_test =  evaluate_test_report(model, renderArgs["background"], iteration, tb_writer)
                        print("\n[ITER {}] Evaluating test : PSNR {}".format(iteration, psnr_test))

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram/all_gauss", model.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/opacity_histogram/fg_gauss", model.gaussians.get_opacity[~model.gaussians.get_is_sky.squeeze()], iteration)
            tb_writer.add_histogram("scene/opacity_histogram/sky_gauss", model.gaussians.get_opacity[model.gaussians.get_is_sky.squeeze()], iteration)
            tb_writer.add_histogram("scene/albedo_histogram", model.gaussians.get_albedo, iteration)
            tb_writer.add_histogram("scene/roughness_histogram", model.gaussians.get_roughness, iteration)
            if model.gaussians.get_metalness is not None:
                tb_writer.add_histogram("scene/metalness_histogram", model.gaussians.get_metalness, iteration)
            tb_writer.add_scalar('total_points', model.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

@hydra.main(version_base=None, config_path="configs", config_name="R3GW")
def main(cfg: DictConfig):
    print("Optimizing " + cfg.dataset.source_path)

    # Initialize system state (RNG)
    safe_state(cfg.run.quiet)

    # Debug mode
    torch.autograd.set_detect_anomaly(cfg.train.detect_anomaly)

    training(cfg)
    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
