import cv2
import torch
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
import numpy as np
import sys
import importlib
from skimage.metrics import structural_similarity as ssim_skimage
from utils.loss_utils import mse2psnr, img2mae, img2mse, l1_loss
import torchvision
from omegaconf import DictConfig
from scene.R3GW_model import R3GW
from scene.cameras import Camera
import hydra
import matplotlib.pyplot as plt
import spaudiopy
import utils.sh_additional_utils as sh_utility
from utils.envmap_utils import process_environment_map_image
from typing import Dict, Tuple


SUN_ANGLES_NUM = 51


def find_view_best_eval_config(config_test: Dict[str, Dict], cfg: DictConfig, view: Camera,
                             model: R3GW, background: torch.Tensor, verbose: bool = False) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    # View evaluation parameters and environment map path
    image_config = config_test[view.image_name]
    mask_path = image_config["mask_path"]
    envmap_img_path = image_config["env_map_path"]
    init_rot_x = image_config["initial_env_map_rotation"]["x"]
    init_rot_y = image_config["initial_env_map_rotation"]["y"]
    init_rot_z = image_config["initial_env_map_rotation"]["z"]
    threshold = image_config["env_map_scaling"]["threshold"]
    scale = image_config["env_map_scaling"]["scale"]
    sun_angle_range = image_config["sun_angles"]

    # Load view environment map and extract SH coefficients
    gt_envmap_sh = process_environment_map_image(envmap_img_path, scale, threshold,
                                                sh_degree=cfg.envlight_sh_degree)

    # Get gt image
    gt_image = view.original_image.cuda()

    # Get eval mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (gt_image.shape[2], gt_image.shape[1]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = torch.from_numpy(mask // 255).cuda()

    best_psnr = 0
    best_angle = None
    all_psnrs = []

    # Define sun angles range
    sun_angles_prepare_list = torch.linspace(sun_angle_range[0], sun_angle_range[1], SUN_ANGLES_NUM)
    sun_angles = [torch.tensor([angle,0, 0]) for angle in sun_angles_prepare_list]

    # Find best sun direction: render view with GT envmap rotated with sun directions in the defined range
    # Best sun angle gives highest PSNR
    for angle in tqdm(sun_angles):
        # Rotate envmap
        gt_envmap_sh_rot = spaudiopy.sph.rotate_sh(gt_envmap_sh.T, init_rot_z,
                                                    init_rot_y, init_rot_x, 'real')
        gt_envmap_sh_rot = spaudiopy.sph.rotate_sh(gt_envmap_sh_rot, angle[2],
                                                    angle[0], angle[1], 'real')
        
        # Render
        gt_envmap_sh_rot = torch.tensor(gt_envmap_sh_rot.T, dtype=torch.float32, device="cuda")
        model.envlight.set_base(gt_envmap_sh_rot)
        sky_sh = torch.zeros(((cfg.sky_sh_degree + 1) ** 2, 3), dtype=torch.float32, device="cuda")

        render_pkg = render(view, model.gaussians, model.envlight, sky_sh, cfg.sky_sh_degree,
                            background, debug=False, fix_sky=True, specular=cfg.specular)
        render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)

        # Compute metrics
        current_psnr = mse2psnr(img2mse(render_pkg["render"], gt_image, mask=mask))
        all_psnrs.append(current_psnr.cpu())

        if current_psnr > best_psnr:
            best_angle = angle
            best_psnr = current_psnr
        
    if verbose:
        print(f"Lowest PSNR: {np.array(all_psnrs).min()}\n")
        print(f"Best angle {best_angle}")
        print(f"Best PSNR: {best_psnr}\n")

    # Rotate envmap SH around y axis by identified best sun direction     
    gt_envmap_sh_rot = spaudiopy.sph.rotate_sh(gt_envmap_sh.T, init_rot_z,
                                                init_rot_y, init_rot_x, 'real')
    gt_envmap_sh_rot = spaudiopy.sph.rotate_sh(gt_envmap_sh_rot, best_angle[2],
                                                best_angle[0], best_angle[1], 'real')
    
    return best_angle, gt_envmap_sh_rot, mask

@torch.no_grad()
def evaluate_test_report(model: R3GW, bg_color: torch.Tensor, iteration: int, tb_writer=None) -> torch.Tensor:

    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Read test config
    sys.path.append(model.config.dataset.test_config_path)
    config_test = importlib.import_module("test_config").config
    config_test_names = [key.split(".")[0] for key in config_test.keys()]

    test_cameras = [c for c in model.scene.getTestCameras() if c.image_name in config_test_names]

    psnrs = []

    for view in tqdm(test_cameras):
        print(view.image_name)
        gt_image = view.original_image.cuda()

        # Find best sun angle for the current view's envmap and rotate envmap SH accordingly
        _, gt_envmap_sh_rot, mask = find_view_best_eval_config(config_test, model.config, view, model, background, verbose=False)
   
        gt_envmap_sh_rot = torch.tensor(gt_envmap_sh_rot, dtype=torch.float32, device="cuda")
        model.envlight.set_base(gt_envmap_sh_rot.T)
        sky_sh = torch.zeros(((model.config.sky_sh_degree + 1) ** 2, 3), dtype=torch.float32, device="cuda")

        render_pkg = render(view, model.gaussians, model.envlight, sky_sh, model.config.sky_sh_degree,
                            background, debug=False, fix_sky=True, specular=model.config.specular)
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        torch.cuda.synchronize()
        gt_image = gt_image[0:3, :, :]
        if tb_writer:
            tb_writer.add_images("test" + "_view_{}/{}".format(view.image_name, "render"),
                                 rendering[None], global_step=iteration)
            tb_writer.add_images("test" + "_view_{}/{}".format(view.image_name, "ground_truth"),
                                 gt_image[None], global_step=iteration)
        # Compute metrics
        psnrs.append(mse2psnr(img2mse(rendering, gt_image, mask=mask)))

    return torch.tensor(psnrs).mean()

@torch.no_grad()
def render_and_evaluate_test_scenes(cfg: DictConfig) -> None:
    "The function is adapted from LumiGauss https://lumigauss.github.io/"
    model = R3GW(cfg, load_chkpt_iteration=cfg.eval.iteration)
    iteration = model.load_chkpt_iteration

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Read test config
    sys.path.append(cfg.dataset.test_config_path)
    config_test = importlib.import_module("test_config").config
    config_test_names = [key.split(".")[0] for key in config_test.keys()]

    test_cameras = [c for c in model.scene.getTestCameras() if c.image_name in config_test_names]

    out_dir_name = "eval_gt_envmaps"
    renders_path = os.path.join(cfg.dataset.model_path, out_dir_name, "test",
                                "iteration_{}".format(iteration), "renders")
    renders_unmasked_path = os.path.join(cfg.dataset.model_path, out_dir_name, "test",
                                         "iteration_{}".format(iteration), "renders_unmasked")
    gts_path = os.path.join(cfg.dataset.model_path, out_dir_name, "test",
                            "iteration_{}".format(iteration), "gt")
    makedirs(renders_path, exist_ok=True)
    makedirs(renders_unmasked_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    ssims, psnrs, mses, maes, rec_losses = [], [], [], [], []
    img_names, used_angles =[], []

    for view in tqdm(test_cameras):
        print(view.image_name)
        sky_mask = view.sky_mask.cuda()
        gt_image = view.original_image.cuda()

        # Find best sun angle for the current view's envmap and rotate envmap SH accordingly
        best_sun_angle, gt_envmap_sh_rot, mask = find_view_best_eval_config(config_test, cfg, view, model, background, verbose=True)

        # Save best_envmap:
        np.save(os.path.join(renders_path, "best_envmap" + view.image_name+".npy"), gt_envmap_sh_rot)      
        render_best_angle_envmap = sh_utility.sh_render(gt_envmap_sh_rot.T, width = 360)
        render_best_angle_envmap = torch.tensor(render_best_angle_envmap ** (1/ 2.2))
        render_best_angle_envmap =  np.array(render_best_angle_envmap * 255).clip(0,255).astype(np.uint8)
        render_best_angle_envmap = (render_best_angle_envmap -  render_best_angle_envmap.min()) / (render_best_angle_envmap.max() -  render_best_angle_envmap.min())
        plt.imsave(os.path.join(renders_path, "best_angle_rot_envmap" + view.image_name + ".jpg"), render_best_angle_envmap)
        
        # Render with GT envmap rotated according to best sun angle 
        gt_envmap_sh_rot = torch.tensor(gt_envmap_sh_rot.T, dtype=torch.float32, device="cuda")
        model.envlight.set_base(gt_envmap_sh_rot)
        sky_sh = torch.zeros(((cfg.sky_sh_degree + 1) ** 2, 3), dtype=torch.float32, device="cuda")
        render_pkg = render(view, model.gaussians, model.envlight, sky_sh, cfg.sky_sh_degree,
                            background, debug=False, fix_sky=True, specular=cfg.specular)
        
        # Save
        rendering_masked = torch.clamp(render_pkg["render"] * mask, 0.0, 1.0)
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        torch.cuda.synchronize()
        gt_image = gt_image[0:3, :, :]
        torchvision.utils.save_image(rendering_masked, os.path.join(renders_path, view.image_name + "_masked.png"))
        torchvision.utils.save_image(rendering, os.path.join(renders_unmasked_path, view.image_name + ".png"))
        torchvision.utils.save_image(rendering * sky_mask + torch.ones_like(rendering) * (1 - sky_mask),
                                     os.path.join(renders_unmasked_path, view.image_name + "_masked_sky.png"))
        torchvision.utils.save_image(gt_image * mask, os.path.join(gts_path, view.image_name + ".png"))
        
        used_angles.append(best_sun_angle)
        img_names.append(view.image_name)
        
        # Compute metrics
        psnrs.append(mse2psnr(img2mse(rendering, gt_image, mask=mask)))
        maes.append(img2mae(rendering, gt_image, mask=mask))
        mses.append(img2mse(rendering, gt_image, mask=mask))

        rendered_np= rendering.cpu().detach().numpy().transpose(1, 2, 0)
        gt_image_np = gt_image.cpu().detach().numpy().transpose(1, 2, 0)
        
        _, full = ssim_skimage(rendered_np, gt_image_np, win_size=5, channel_axis=2, full=True, data_range=1.0)
        mssim_over_mask = (torch.tensor(full).cuda() * mask.unsqueeze(-1)).sum() / (3 * mask.sum())
        ssims.append(mssim_over_mask)
        Ll1 = l1_loss(rendering, gt_image, mask=mask.expand_as(gt_image))
        Ssim = (1.0 - mssim_over_mask)
        rec_loss = Ll1 * (1-cfg.optimizer.lambda_dssim) + cfg.optimizer.lambda_dssim * Ssim
        rec_losses.append(rec_loss)

    psnrs_dict = {img_name: psnr.item() for img_name, psnr in zip(img_names, psnrs)}
    mses_dict = {img_name: mse.item() for img_name, mse in zip(img_names, mses)}
    maes_dict = {img_name: mae.item() for img_name, mae in zip(img_names, maes)}
    ssims_dict = {img_name: ssim.item() for img_name, ssim in zip(img_names, ssims)}
    # Print metrics
    print("  PSNR: {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  MSE: {:>12.7f}".format(torch.tensor(mses).mean(), ".5"))
    print("  MAE: {:>12.7f}".format(torch.tensor(maes).mean(), ".5"))
    print("  SSIM: {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    # Save metrics 
    with open(os.path.join(renders_path, "metrics.txt"), 'w') as f:
        print("  PSNR: {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"), file=f)
        print("  MSE: {:>12.7f}".format(torch.tensor(mses).mean(), ".5"), file=f)
        print("  MAE: {:>12.7f}".format(torch.tensor(maes).mean(), ".5"), file=f)
        print("  SSIM: {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"), file=f)
        print("  REC LOSS: {:>12.7f}".format(torch.tensor(rec_losses).mean(), ".5"), file=f)
        print(f"  best PSNRs: {psnrs_dict}", file=f)
        print(f"  best MSEs: {mses_dict}", file=f)
        print(f"  best MAEs: {maes_dict}", file=f)
        print(f"  best SSIMs: {ssims_dict}", file=f)


@hydra.main(version_base=None, config_path="configs", config_name="R3GW")
def main(cfg: DictConfig):

    print("Rendering and evaluating with GT environment maps " + cfg.dataset.model_path)

    safe_state(cfg.run.quiet)

    cfg.dataset.eval = True
    render_and_evaluate_test_scenes(cfg)

    print("\nEvaluation with GT environment maps complete.")


if __name__ == "__main__":
    main()
