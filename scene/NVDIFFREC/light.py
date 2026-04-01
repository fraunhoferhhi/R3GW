import numpy as np
import torch
import nvdiffrast.torch as dr
from . import util
from utils.sh_utils import  eval_sh
from utils.sh_additional_utils import sh_render
from utils.sh_utils import gauss_kernel
from typing import Dict, Tuple, Optional


class EnvironmentLight(torch.nn.Module):

    def __init__(self, base: torch.Tensor, sh_degree : int = 4):
        """
        The class implements a Cook-Torrance shader based on IBL. The code builds on https://github.com/NVlabs/nvdiffrecmc.

        Attributes:
            base (torch.Tensor): Spherical Harmonics (SH) coefficients
            sh_degree (int): SH degree,
            sh_dim (int): number of SH coefficients.
        Constants:
            NUM_CHANNELS (int): number of channels of base SH coefficients
            C1,C2,...,C5 (int): constants for computing diffuse irradiance
            _FG_LUT (torch.Tensor): lookup texture for Fresnel-Schlick reflectivity term
        """
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if sh_degree > 5:
            raise NotImplementedError
        else:
            self._sh_degree = sh_degree
        self._sh_dim = (sh_degree +1)**2 
        self._base = base.squeeze()
        self.NUM_CHANNELS = 3
        self.C1 = 0.429043
        self.C2 = 0.511664
        self.C3 = 0.743125
        self.C4 = 0.886227
        self.C5 = 0.247708
        lut = np.fromfile("scene/NVDIFFREC/irrmaps/bsdf_256_256.bin", dtype=np.float32).reshape(1, 256, 256, 2)
        self.register_buffer("_FG_LUT", torch.from_numpy(lut))


    def clone(self):
        return EnvironmentLight(self._base.clone().detach(), sh_degree=self._sh_degree)

    @property
    def get_shdim(self) -> int:
        return self._sh_dim

    @property
    def get_shdegree(self) -> int:
        return self._sh_degree

    @property
    def get_base(self):
        return self._base


    def set_base(self, base: torch.Tensor):
        assert base.squeeze().shape[0] == self._sh_dim, f"The number of SH coefficients must be {self._sh_dim}"
        self._base = base.squeeze()


    def get_diffuse_irradiance(self, normal: torch.Tensor) -> torch.Tensor:
        """
        The function computes the diffuse irradiance according to section 3.2 of "An efficient representaiton for Irradiance Environment Maps"
        by Ramamoorthi and Pat Hanrahan, https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf.
        The implementation follows LumiGauss implementation https://arxiv.org/abs/2408.04474.
        The diffuse irradiance is computed by convolving environment light and cosine term in frequency domain.
        In the SH expansion of the environment light only terms up to degree 2 are considered.

        Args:
            normal (torch.Tensor): tensor of shape Nx3 containing normal vectors in R³.
        Returns:
            diffuse_irradiance (torch.Tensor): tensor of shape Nx1 containing the diffuse irradiance for each normal vector.
        """

        x, y, z = normal[..., 0, None], normal[..., 1, None], normal[..., 2, None]

        diffuse_irradiance = (
            self.C1 * self._base[8,:] * (x ** 2 - y ** 2) +
            self.C3 * self._base[6,:] * (z ** 2) +
            self.C4 * self._base[0,:] -
            self.C5 * self._base[6,:] +
            2 * self.C1 * self._base[4,:]* x * y +
            2 * self.C1 * self._base[7,:] * x * z +
            2 * self.C1 * self._base[5,:] * y * z +
            2 * self.C2 * self._base[3,:]* x +
            2 * self.C2 * self._base[1,:] * y +
            2 * self.C2 * self._base[2,:] * z
        )

        return diffuse_irradiance


    def get_specular_light_sh(self, kr: torch.Tensor) -> torch.Tensor:
        """
        The function computes specular lighting SH coefficients by convolving
        in frequency domain envionment light and a Gaussian blur kernel of std = kr.
        The Gaussian blur filter SH coefficients are derived using the Gauss-Weierstrass kernel.

        Args: 
            kr (torch.Tensor): roughness tensor of shape Nx1
        Returns:
            spec_light: tensor of shape Nxself.sh_dimx3 storing the SH coefficients
                        of specular light for each roughness value and channel.
        """
        # Build coefficients of blur kernel in frequency (SH) domain
        gwk_sh = gauss_kernel(kr, self._sh_degree) # N x 25
        gwk_sh = gwk_sh.unsqueeze(-1) # N x 25 x 1
        # Adjust dimensions
        envlight_sh = self._base.unsqueeze(0)   # 1 x 25 x 3
        envlight_sh = envlight_sh.repeat(gwk_sh.shape[0], 1, 1) # N x 25 x 3
        # Perform convolution
        spec_light = gwk_sh * envlight_sh # N x 25 x 3

        return spec_light


    def sample_illumination(self,
                            gb_pos: torch.Tensor,
                            view_pos: torch.Tensor
                            ) -> torch.Tensor:
        wo = util.safe_normalize(view_pos - gb_pos).squeeze()
        illu_hdr = torch.nn.functional.relu(eval_sh(self._sh_degree,
                                                       self._base.unsqueeze(0).expand(wo.shape[0], -1, -1).transpose(1,2),
                                                       wo))

        return util.gamma_correction(illu_hdr) # linear --> sRGB


    def shade(self,
              gb_pos: torch.Tensor,
              gb_normal: torch.Tensor,
              albedo: torch.Tensor,
              view_pos: torch.Tensor,
              kr: Optional[torch.Tensor],
              km: Optional[torch.Tensor],
              specular: bool =  True,
              gamma_correct: bool = True,
              ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        The function returns the emitted radiance along the input viewing direction. 
        If specular is True a Cook-Torrane reflectance model is assumed,
        otherwise Lambertian reflections are assumed.
        Args:
            gb_pos (torch.Tensor): world positions HxWxNx3
            gb_normal (torch.Tensor): normal vectors HxWxNx3
            albedo (torch.Tensor): albedo of the surface, base color HxWxNx3
            view_pos (torch.Tensor): camera position HxWxNx3
            kr (torch.Tensor): roughness of points HxWxNx1
            km (torch.Tensor): metalness of points HxWxNx1
            specular bool): if True specular reflection is considered otherwise diffuse only
            gamma_correct (bool): if True gamma correction is applied to the output color
        Returns:
            rgb (torch.Tensor): shaded rgb color of shape HxWxNx3.
            extras (Dict[str, torch.Tensor]): dictionary storing diffuse and specular radiance.
        """

        diffuse_irradiance_hdr = torch.clamp_min(self.get_diffuse_irradiance(gb_normal.squeeze()), 1e-4)
        # Compute diffuse color
        diffuse_rgb = (albedo / torch.pi) * diffuse_irradiance_hdr
        # Gamma correction: linear --> sRGB
        if gamma_correct:
            diffuse_srgb = util.gamma_correction(diffuse_rgb)
        else:
            diffuse_srgb = diffuse_rgb
        extras = {"diffuse": diffuse_srgb}

        if not specular:
            extras.update({"specular": torch.zeros_like(extras["diffuse"])})

            return diffuse_srgb, extras
        else:
            wo = util.safe_normalize(view_pos - gb_pos) # (H, W, N, 3)
            reflvec = util.safe_normalize(util.reflect(wo, gb_normal))
            # Lookup FG term from lookup texture
            NdotV = torch.clamp(util.dot(wo, gb_normal), min=1e-4)
            fg_uv = torch.cat((NdotV, kr), dim=-1)
            fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')
            # Convovlve base SH coeffs with SH coeffs of Gaussian kernel
            spec_light = self.get_specular_light_sh(kr.squeeze([0,1])) # (N, 25, 3)
            # Compute specular irradiance in reflection direction
            spec_irradiance_hdr = eval_sh(self._sh_degree, spec_light.transpose(1,2), reflvec.squeeze())
            # Adjust dimensions and clamp
            spec_irradiance_hdr = torch.clamp_min(spec_irradiance_hdr[None, None, ...], 1e-4) # (H, W, N, 3)
            # Compute Fresnel-Schlick reflectivity
            if km is None:
                F0 = torch.ones_like(albedo) * 0.04  # [1, H, W, 3]
                kd = 1.0
            else:
                F0 = (1.0 - km) * 0.04 + albedo * km
                kd = (1.0 - km)
            reflectivity = F0 * fg_lookup[...,0:1] + fg_lookup[...,1:2]
            # Compute specular color
            specular_rgb = spec_irradiance_hdr * reflectivity
            if gamma_correct:
                specular_srgb = util.gamma_correction(specular_rgb)
            extras.update({'specular': specular_srgb})

            shaded_rgb = kd * diffuse_rgb + specular_rgb

            # Gamma correction: linear --> sRGB
            if gamma_correct:
                shaded_rgb = util.gamma_correction(shaded_rgb)
            
            return shaded_rgb, extras


    def render_sh(self, width: int = 600) -> torch.Tensor:
        """
        Render environment light SH coefficients in equirectangular format.
        Args:
            width (int): width of the output environment map. The height is set to width // 2.
        Returns:
            rendered_sh (torch.Tensor): rendered environment map.
        """
        self._base = self._base.squeeze()
        if isinstance(self._base, torch.Tensor):     
            rendered_sh = torch.tensor(sh_render(self._base.cpu().numpy(), width=width))
        else:
            rendered_sh = torch.tensor(sh_render(self._base, width=width))
        rendered_sh = util.gamma_correction(rendered_sh)

        return rendered_sh
