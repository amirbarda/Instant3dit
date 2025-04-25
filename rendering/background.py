import torch
import torch.nn as nn
from dataclasses import dataclass
from utils.typing_utils import *

class SolidColorBackground(nn.Module):
    
    @dataclass
    class Config():
        n_output_dims: int = 3
        color: Tuple = (1.0, 1.0, 1.0)
        learned: bool = False

    cfg: Config

    def __init__(self):
        super().__init__()

    def configure(self) -> None:
        self.cfg = self.Config()
        self.env_color: Float[Tensor, "Nc"]
        if self.cfg.learned:
            self.env_color = nn.Parameter(
                torch.as_tensor(self.cfg.color, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "env_color", torch.as_tensor(self.cfg.color, dtype=torch.float32, device='cuda')
            )

    def forward(self, dirs: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B Nc"]:
        return (
            torch.ones(*dirs.shape[:-1], self.cfg.n_output_dims).to(dirs)
            * self.env_color
        )
