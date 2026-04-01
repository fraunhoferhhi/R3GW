import matplotlib.pyplot as plt
import numpy as np
import os
import utils.sh_additional_utils as sh_utility
import moviepy.video.io.ImageSequenceClip
from utils.sh_additional_utils import get_coefficients_from_image
import torch
from typing import List
import cv2


os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


def process_environment_map_exr(envmap_path: str, l_max: int = 4) -> np.ndarray:
    
    envmap = cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED)
    envmap = cv2.cvtColor(envmap, cv2.COLOR_BGR2RGB)
    # drop alpha if present
    envmap = envmap[:,:,:3]
    envmap = envmap.astype(np.float32)
    sh_coeffs = get_coefficients_from_image(envmap, l_max=l_max)
    return sh_coeffs


def process_environment_map_image(img_path: str, scale_high: float = 10, threshold: float = 0.999, sh_degree: int = 4) -> np.ndarray:
    
    img = plt.imread(img_path)
    img = torch.from_numpy(img).float() / 255
    img[img > threshold] *= scale_high
    sh_coeffs = get_coefficients_from_image(img.numpy(), sh_degree)
    return sh_coeffs


def create_video_rotation_envmap(output_path: str, envs: List[torch.Tensor]) -> None:
    rendered_envs = []
    for env in envs:
        rendered_sh_env = sh_utility.sh_render(env, width=600)
        rendered_sh_env = torch.tensor(rendered_sh_env ** (1 / 2.2))
        rendered_sh_env = np.array(rendered_sh_env * 255).clip(0,255).astype(np.uint8)
        rendered_envs.append(rendered_sh_env)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(rendered_envs, fps=15)
    clip.write_videofile(os.path.join(output_path, 'rots_envmap.mp4'))


def save_reconstructed_envmap_sh(envmap_sh: str, outpath: str) -> None:
    render_sh_envmap = sh_utility.sh_render(envmap_sh, width = 360)
    render_sh_envmap = np.clip((render_sh_envmap ** (1 / 2.2)) * 255, 0, 255).astype(np.uint8)
    plt.imsave(outpath, render_sh_envmap)
