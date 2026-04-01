import torch
from scene.NVDIFFREC import EnvironmentLight
from scene.net_models import MLPNet
from omegaconf import OmegaConf, DictConfig
from scene import GaussianModel, Scene
from utils.system_utils import searchForMaxIteration
import os
from typing import Optional, List, Dict
import numpy as np
import torchvision
from utils.sh_utils import render_sh_map
import matplotlib.pyplot as plt


class R3GW:
    def __init__(self,
                 config: DictConfig,
                 load_chkpt_iteration: Optional[int] = None,
                 chkpt: Optional[Dict] = None,
                 training: bool = False
                 ):
        """
        R3GW model.
        
        Args:
            config: Configuration object containing model and dataset parameters.
            load_chkpt_iteration: Iteration to load checkpoint from. -1 loads the latest.
            chkpt: Pre-loaded checkpoint dictionary (optional).
            training: Whether to initialize in training mode.
        """
        self.config = config
        self.load_chkpt_iteration = load_chkpt_iteration
        self.chkpt = chkpt
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.training = training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._init_gaussians_and_scene()
        self._init_networks()
        self._init_checkpoint()
        self._init_envlight()


    def _init_gaussians_and_scene(self) -> None:
        """
        Initialize Gaussian model and scene.
        """
        self.gaussians: GaussianModel = GaussianModel(use_metalness=self.config.use_metalness)
        self.scene: Scene = Scene(self.config.dataset, self.gaussians, training=self.training)
        self.train_cameras: List = self.scene.getTrainCameras().copy()
        self.test_cameras: List = self.scene.getTestCameras().copy()


    def _init_networks(self) -> None:
        """
        Initialize MLP network and image embeddings.
        """
        self.mlp: MLPNet = MLPNet(sh_degree_envl=self.config.envlight_sh_degree, 
                                  sh_degree_sky=self.config.sky_sh_degree,
                                  embedding_dim=self.config.embeddings_dim)
        self.mlp.to(self.device)
        
        self.embeddings: torch.nn.Embedding = torch.nn.Embedding(len(self.train_cameras),
                                                    self.config.embeddings_dim)
        self.embeddings.to(self.device)
        self.embeddings_test: Optional[torch.nn.Embedding] = None

   
    def _init_checkpoint(self) -> None:
        """
        Load checkpoint or initialize fresh model.
        """
        if self.chkpt is not None:
            self.load_checkpoint()
        elif self.load_chkpt_iteration is not None:
            self._resolve_checkpoint_iteration()
            print("Loading trained model at iteration {}".format(self.load_chkpt_iteration))
            self.load_checkpoint()
        else:
            self.gaussians.augment_with_sky_gaussians()


    def _resolve_checkpoint_iteration(self) -> None:
        """
        Resolve checkpoint iteration path, assume -1 for latest.
        """
        outputs_paths = ["point_cloud", "checkpoint"]
        if self.load_chkpt_iteration == -1:
            load_iters = [searchForMaxIteration(os.path.join(self.config.dataset.model_path, op)) 
                         for op in outputs_paths]
            assert len(set(load_iters)) == 1, "Checkpoint iterations do not match across outputs"
            self.load_chkpt_iteration = load_iters[0]
        else:
            for op in outputs_paths:
                chkpt_path = os.path.join(self.config.dataset.model_path, 
                                         op + f"/iteration_{self.load_chkpt_iteration}")
                assert os.path.exists(chkpt_path), \
                    f"Load iteration {self.load_chkpt_iteration}: {op} path missing"

   
    def _init_envlight(self) -> None:
        """
        Initialize environment light.
        """
        sh_dim = (self.config.envlight_sh_degree + 1) ** 2
        init_base = torch.zeros(sh_dim, 3, device=self.device)
        self.envlight: EnvironmentLight = EnvironmentLight(
            base=init_base,
            sh_degree=self.config.envlight_sh_degree
        ).to(self.device)


    def training_setup(self) -> None:
        """
        Set up optimizer for training.
        """
        training_args = self.config.optimizer
        gaussians_opt_params = self.gaussians.training_setup(training_args)

        model_opt_params =  [
            {'params': self.mlp.parameters(), 'lr': training_args.mlp_lr, "name": 'mlp'} ,
            {'params': self.embeddings.parameters(), 'lr': training_args.embeddings_lr, "name": 'embeddings'}    
        ]
        model_opt_params.extend(gaussians_opt_params)

        self.optimizer = torch.optim.Adam(model_opt_params, lr=0.01, eps=1e-15)
        self.gaussians.set_optimizer(self.optimizer)

   
    def update_learning_rate(self, iteration: int) -> None:
        self.gaussians.update_learning_rate(iteration)


    def get_envlights_sh_all(self) -> Dict[str, np.ndarray]:
        envlights_sh = {}
        viewpoint_stack = self.train_cameras
        
        with torch.no_grad():
            for viewpoint_cam in viewpoint_stack:
                viewpoint_cam_id = torch.tensor([viewpoint_cam.uid], device=self.device)
                image_embed = self.embeddings(viewpoint_cam_id)
                envlights_sh[viewpoint_cam.image_name] = self.mlp(image_embed)[0].detach().cpu().numpy()
        return envlights_sh
    

    def get_sky_sh_all(self) -> Dict[str, np.ndarray]:
        skys_sh = {}
        viewpoint_stack = self.train_cameras

        with torch.no_grad():
            for viewpoint_cam in viewpoint_stack:
                viewpoint_cam_id = torch.tensor([viewpoint_cam.uid], device=self.device)
                image_embed = self.embeddings(viewpoint_cam_id)
                skys_sh[viewpoint_cam.image_name] = self.mlp(image_embed)[1].detach().cpu().numpy()
        return skys_sh
    

    def render_envlights_sh_all(self,
                                save_path: str,
                                save_sh_coeffs: bool = False
                                ):
        envlights_sh = self.get_envlights_sh_all()
        for im_name in envlights_sh.keys():
            envlight_sh = torch.from_numpy(envlights_sh[im_name]).to(self.device)
            self.envlight.set_base(envlight_sh)
            if save_sh_coeffs:
                np.save(os.path.join(save_path, im_name + ".npy"),
                        self.envlight.get_base.detach().cpu().numpy())
            rendered_sh = self.envlight.render_sh()
            save_path_im = os.path.join(save_path, im_name + ".jpg")
            rendered_sh = np.array(rendered_sh * 255).clip(0, 255).astype(np.uint8)
            plt.imsave(save_path_im, rendered_sh)


    def render_sky_sh_all(self,
                          save_path: str,
                          save_sh_coeffs: bool = False
                          ) -> None:
        sky_sh = self.get_sky_sh_all()
        for im_name in sky_sh.keys():
            if save_sh_coeffs:
                np.save(os.path.join(save_path, im_name + ".npy"), sky_sh[im_name])
            rendered_sh = render_sh_map(sky_sh[im_name].squeeze())
            save_path_im = os.path.join(save_path, im_name + ".jpg")
            torchvision.utils.save_image(rendered_sh.permute(2, 0, 1), save_path_im)


    def save_config(self) -> None:
        config_path = os.path.join(self.config.dataset.model_path, "R3GW_run.yaml")
        OmegaConf.save(self.config, config_path)


    def save_checkpoint(self,
                        iteration: int,
                        grad_threshold: Optional[float] = None
                        ) -> None:
        model_path = self.config.dataset.model_path
        chkpt_path = os.path.join(model_path, "checkpoint/iteration_{}".format(iteration))
        os.makedirs(chkpt_path, exist_ok=True)

        print("Saving all model's parameters\n")
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "gaussians": self.gaussians.capture(),
            "embeddings": self.embeddings.state_dict(),
            "mlp": self.mlp.state_dict(),
            "iteration": iteration,
            "grad_threshold": grad_threshold
        }, chkpt_path + "/checkpoint.pth")

        print("Saving Gaussians ply\n")
        self.scene.save(iteration)

      
    def load_checkpoint(self) -> None:
        """
        Load model state from checkpoint.
        """
        if self.chkpt is None:
            model_path = self.config.dataset.model_path
            chkpt_path = os.path.join(model_path, 
                                      f"checkpoint/iteration_{self.load_chkpt_iteration}/checkpoint.pth")
            assert os.path.exists(chkpt_path), \
                f"Checkpoint not found at iteration {self.load_chkpt_iteration}"
            checkpoint = torch.load(chkpt_path, map_location=self.device)
        else:
            checkpoint = self.chkpt
        
        self.embeddings.load_state_dict(checkpoint["embeddings"])
        self.mlp.load_state_dict(checkpoint["mlp"])
        self.gaussians.restore(checkpoint["gaussians"])
        
        if self.training:
            self.mlp.train()
            self.training_setup()
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            self.mlp.eval()
