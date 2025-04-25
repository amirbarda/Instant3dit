import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ops import get_activation, dot

from utils.typing_utils import *



class DiffuseWithPointLightMaterial(nn.Module):
    @dataclass
    class Config():
        ambient_light_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
        diffuse_light_color: Tuple[float, float, float] = (0.9, 0.9, 0.9)
        ambient_only_steps: int = 1000
        diffuse_prob: float = 0.75
        textureless_prob: float = 0.5
        albedo_activation: str = "none"
        soft_shading: bool = False

    cfg: Config
    requires_normal: bool = True

    def configure(self) -> None:
        self.cfg = self.Config()
        self.ambient_light_color: Float[Tensor, "3"]
        self.register_buffer(
            "ambient_light_color",
            torch.as_tensor(self.cfg.ambient_light_color, dtype=torch.float32),
        )
        self.diffuse_light_color: Float[Tensor, "3"]
        self.register_buffer(
            "diffuse_light_color",
            torch.as_tensor(self.cfg.diffuse_light_color, dtype=torch.float32),
        )
        self.ambient_only = False

    
    def get_albedo(self,features):
        albedo = get_activation(self.cfg.albedo_activation)(features).clamp(0.0, 1.0)
        return albedo

    def forward(
        self,
        features: Float[Tensor, "B ... Nf"],
        positions: Float[Tensor, "B ... 3"],
        shading_normal: Float[Tensor, "B ... 3"],
        light_positions: Float[Tensor, "B ... 3"],
        ambient_ratio: Optional[float] = None,
        shading: Optional[str] = None,
        **kwargs,
    ) -> Float[Tensor, "B ... 3"]:
        
        shading = 'diffuse'
        albedo = get_activation(self.cfg.albedo_activation)(features[..., :3]).clamp(0.0, 1.0)

        if ambient_ratio is not None:
            # if ambient ratio is specified, use it
            diffuse_light_color = (1 - ambient_ratio) * torch.ones_like(
                self.diffuse_light_color, device=features.device
            )
            ambient_light_color = ambient_ratio * torch.ones_like(
                self.ambient_light_color, device=features.device
            )
        elif self.training and self.cfg.soft_shading:
            # otherwise if in training and soft shading is enabled, random a ambient ratio
            diffuse_light_color = torch.full_like(
                self.diffuse_light_color, random.random(), device=features.device
            )
            ambient_light_color = 1.0 - diffuse_light_color
        else:
            # otherwise use the default fixed values
            diffuse_light_color = self.diffuse_light_color.to(features.device)
            ambient_light_color = self.ambient_light_color.to(features.device)

        light_directions: Float[Tensor, "B ... 3"] = F.normalize(
            light_positions - positions, dim=-1
        )
        diffuse_light: Float[Tensor, "B ... 3"] = (
            dot(shading_normal, light_directions).clamp(min=0.0) * diffuse_light_color
        )
        textureless_color = diffuse_light + ambient_light_color
        # clamp albedo to [0, 1] to compute shading
        color = albedo * textureless_color

        # if shading is None:
        #     if self.training:
        #         # adopt the same type of augmentation for the whole batch
        #         if self.ambient_only or random.random() > self.cfg.diffuse_prob:
        #             shading = "albedo"
        #         elif random.random() < self.cfg.textureless_prob:
        #             shading = "textureless"
        #         else:
        #             shading = "diffuse"
        #     else:
        #         if self.ambient_only:
        #             shading = "albedo"
        #         else:
        #             # return shaded color by default in evaluation
        #             shading = "diffuse"

        # multiply by 0 to prevent checking for unused parameters in DDP
        if shading == "albedo":
            return albedo + textureless_color * 0
        elif shading == "textureless":
            return albedo * 0 + textureless_color
        elif shading == "diffuse":
            return color
        else:
            raise ValueError(f"Unknown shading type {shading}")

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        if global_step < self.cfg.ambient_only_steps:
            self.ambient_only = True
        else:
            self.ambient_only = False

    def export(self, features: Float[Tensor, "*N Nf"], **kwargs) -> Dict[str, Any]:
        albedo = get_activation(self.cfg.albedo_activation)(features[..., :3]).clamp(
            0.0, 1.0
        )
        return {"albedo": albedo}
