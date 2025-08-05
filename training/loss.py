from typing import List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import ListConfig

from util import append_dims, instantiate_from_config, default
from lpips import LPIPS
from diffusers import DDIMScheduler
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
import torch.nn.functional as F

def idx_to_sigma(sigmas, idx):
    return sigmas[idx]

def sample_sigmas(sigmas, n_samples, rand=None):
    num_idx = sigmas.shape[0]
    idx = default(
        rand,
        torch.randint(0, num_idx, (n_samples,), device=sigmas.device),
    )
    return idx_to_sigma(sigmas,idx)

class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        pipe,
        noise_sched,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()

        self.pipe = pipe
        self.noise_scheduler = noise_sched
        assert type in ["l2", "l1", "lpips"]

        self.type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            self.lpips = LPIPS().eval()

    def __call__(self, input, batch):

        latents = input
        prompts = batch['txt']

        # TODO: move to dataloader collate
        input_ids = []
        for prompt in prompts:
            input_ids.append(self.pipe.tokenizer(
                prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.pipe.tokenizer.model_max_length,
            ).input_ids)
        input_ids = self.pipe.tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids.to(latents.device)
        # encode images and masked images
        # Sample noise that we'll add to the latents
        mask = batch['interpolated_mask']
        masked_latents = batch['masked_latents']
        noise = torch.randn_like(latents, device=latents.device)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)#latents + 0.0 

        # concatenate the noised latents with the mask and the masked latents
        latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.pipe.text_encoder(input_ids)[0]

        # Predict the noise residual
        noise_pred = self.pipe.unet(latent_model_input, timesteps, encoder_hidden_states).sample      

        return F.mse_loss(noise_pred, noise, reduction='mean')

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2 ).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
