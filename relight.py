import torch
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import numpy as np
import torchvision
from utils.general_utils import safe_state
from omegaconf import DictConfig
from scene.R3GW_model import R3GW
from eval_with_gt_envmaps import process_environment_map_image
import hydra
import spaudiopy
import moviepy.video.io.ImageSequenceClip
from utils.envmap_utils import save_reconstructed_envmap_sh, process_environment_map_exr, create_video_rotation_envmap


os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

@torch.no_grad()
def render_single_and_relight(cfg: DictConfig, rot_angle_x: float = -np.pi/2):
    "The function is adapted from LumiGauss https://lumigauss.github.io/"
    
    model = R3GW(cfg, load_chkpt_iteration=cfg.relighting.iteration)
    iteration = model.load_chkpt_iteration

    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    tmp = cfg.relighting.view_name.split(".")[0]
    if cfg.relighting.training_view:
        camera = [c for c in model.scene.getTrainCameras() if c.image_name == tmp][0]
    else:
        camera = [c for c in model.scene.getTestCameras() if c.image_name == tmp][0]

    # Sky color sh coefficients can't be rotated, therefore the color of the sky Gaussians is fixed to white.
    fix_sky = True
    outdir_name = "relighting"
    relits_path = os.path.join(cfg.dataset.model_path, outdir_name,
                                "iteration_{}".format(iteration), camera.image_name)
    makedirs(relits_path, exist_ok=True)

    # Load environment light and sky color SH
    if cfg.relighting.trained_illumination_name is not None:
        rot_angle_x = 0
        fix_sky = False
        tmp = cfg.relighting.trained_illumination_name.split(".")[0]
        cam_illu = [c for c in model.scene.getTrainCameras() if c.image_name == tmp][0]
        cam_illu_id =  torch.tensor([cam_illu.uid], device = 'cuda')
        embedding_gt_image = model.embeddings(cam_illu_id)
        envmap_sh, sky_sh = model.mlp(embedding_gt_image)
        envmap_sh = envmap_sh.cpu().numpy().squeeze()
        envmap_name = cfg.relighting.trained_illumination_name
        relits_path = os.path.join(relits_path, "trained_envmap_" + envmap_name)
        makedirs(relits_path, exist_ok=True)
    else:
        envmap_name = os.path.basename(cfg.relighting.envmap_path)[:-4]
        relits_path = os.path.join(relits_path, "envmap_" + envmap_name)
        makedirs(relits_path, exist_ok=True)
        sky_sh =  torch.zeros((9,3), dtype=torch.float32, device="cuda")
        envmap_file_ext = os.path.splitext(cfg.relighting.envmap_path)[-1].lower()
        try:
            if envmap_file_ext in [".exr", ".hdr"]:
                envmap_sh = process_environment_map_exr(cfg.relighting.envmap_path)
            elif envmap_file_ext == ".jpg":
                envmap_sh = process_environment_map_image(cfg.relighting.envmap_path)
        except Exception as e:
            print(f"Only .exr and .jpg extensions are supported: {e}")
            raise
    save_reconstructed_envmap_sh(envmap_sh, os.path.join(relits_path, "envmap_reconstructed_" + envmap_name + ".png"))

    # Rotate environment map around x-axis by -pi/2 to align the ground plane of the scene with the horizon in the environment map.
    # Needed only for external environment map, trained environment map is already aligned.
    envmap_sh_rot_around_x = spaudiopy.sph.rotate_sh(envmap_sh.T, 0, 0,rot_angle_x, 'real')
    envmaps_sh_rot_around_x_torch = torch.tensor(envmap_sh_rot_around_x.T, dtype=torch.float32, device="cuda")
    envmap_sh_torch = torch.tensor(envmap_sh, dtype=torch.float32, device="cuda")

    # Render relighted view with no rotation of the environment map around x-axis
    model.envlight.set_base(envmap_sh_torch)
    render_pkg = render(camera, model.gaussians, model.envlight, sky_sh, cfg.sky_sh_degree,
                        background, debug=False, fix_sky=fix_sky, specular=model.config.specular)
    render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)
    torchvision.utils.save_image(render_pkg["render"], os.path.join(relits_path, "relit_envmap_" + envmap_name + ".png"))

    # Render relighted view with rotation of the environment map around x-axis 
    model.envlight.set_base(envmaps_sh_rot_around_x_torch)
    render_pkg = render(camera, model.gaussians, model.envlight, sky_sh, cfg.sky_sh_degree,
                        background, debug=False, fix_sky=fix_sky, specular=model.config.specular)
    render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)
    torchvision.utils.save_image(render_pkg["render"], os.path.join(relits_path, "relit_envmap_rotated_x_" + envmap_name + ".png"))
    # Save rotated envmap
    save_reconstructed_envmap_sh(envmap_sh_rot_around_x.T, os.path.join(relits_path, "envmap_rotated_x_reconstructed" + envmap_name + ".png"))

    # Render and relight view with N rotations of the environment map around the y axis.
    # Rotation angles are sampled in [0,2pi].
    N = 30
    line_points = np.linspace(0, 1, N)
    angle_start = 0
    angle_end = 3.14 * 2
    sun_angles = np.interp(line_points, [0, 1], [angle_start, angle_end])
    rot_envs = []
    for num, angle in enumerate(tqdm(sun_angles)):
        # Rotate envmap and render
        envmap_sh_rot = spaudiopy.sph.rotate_sh(envmap_sh.T, 0, 0,rot_angle_x, 'real')
        envmap_sh_rot = spaudiopy.sph.rotate_sh(envmap_sh_rot, 0, angle, 0, 'real')
        rot_envs.append(envmap_sh_rot.T)

        if num % 5 == 0 or num == len(sun_angles) - 1: 
            save_reconstructed_envmap_sh(envmap_sh_rot.T, os.path.join(relits_path, f"envmap_rotated_step{num}_reconstructed" + ".png"))

        envmap_sh_rot_torch = torch.tensor(envmap_sh_rot.T, dtype=torch.float32, device="cuda")
        model.envlight.set_base(envmap_sh_rot_torch)
        sky_sh = torch.zeros((9,3), dtype=torch.float32, device="cuda")
        render_pkg = render(camera, model.gaussians, model.envlight, sky_sh, cfg.sky_sh_degree,
                            background, debug=False, fix_sky=True, specular=model.config.specular)
        # Save rendered image
        render_pkg["render"] = torch.clamp(render_pkg["render"], 0.0, 1.0)
        torchvision.utils.save_image(render_pkg["render"], os.path.join(relits_path, str(num) + ".png"))

    # Generate videos
    image_files = [os.path.join(relits_path ,str(num) + ".png") for num in range(0, len(sun_angles)) ]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=15)
    clip.write_videofile(os.path.join(relits_path , 'relit_envmap_rots.mp4'))
    create_video_rotation_envmap(relits_path, rot_envs)

@hydra.main(version_base=None, config_path="configs", config_name="R3GW")
def main(cfg: DictConfig):

    print("Rendering under novel light view: " + cfg.relighting.view_name)
    
    # Initialize system state (RNG)
    safe_state(cfg.run.quiet)

    render_single_and_relight(cfg)

    print("\nRelighting complete.")


if __name__ == "__main__":
    main()
