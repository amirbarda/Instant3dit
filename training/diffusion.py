from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union
import os
import pytorch_lightning as pl
import torch
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionInpaintPipeline
)
#from our_diffusers.sd_inpainting_trainable_pipeline import TrainableStableDiffusionInpaintPipeline
from loss import StandardDiffusionLoss
from util import default, instantiate_from_config, get_obj_from_str
# from ..modules import UNCONDITIONAL_CONFIG
# from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
# from ..modules.ema import LitEma
# from ..util import (
#     default,
#     disabled_train,
#     get_obj_from_str,
#     instantiate_from_config,
#     log_txt_as_img,
# )

import imageio as iio
import numpy as np
from pathlib import Path

# TODO: change to InpaintingDiffusion class the inherhits from this class
class DiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        model_key,
        #ckpt_path = None,
        pipe_kwargs = {},
        # network_config,
        # denoiser_config,
        # first_stage_config,
        # conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        # sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        # network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        # use_ema: bool = False,
        # ema_decay_rate: float = 0.9999,
        # scale_factor: float = 1.0,
        # disable_first_stage_autocast=False,
        input_key: str = "jpg",
        mask_input_key: str = "mask",
        # log_keys: Union[List, None] = None,
        # no_cond_log: bool = False,
        # compile_model: bool = False,
    ):
        super().__init__()
        # self.log_keys = log_keys
        self.input_key = input_key
        self.mask_input_key = mask_input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        # model = instantiate_from_config(network_config)
        # self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
        #     model, compile_model=compile_model
        # )

        # self.denoiser = instantiate_from_config(denoiser_config)
        # self.sampler = (
        #     instantiate_from_config(sampler_config)
        #     if sampler_config is not None
        #     else None
        # )
        # self.conditioner = instantiate_from_config(
        #     default(conditioner_config, UNCONDITIONAL_CONFIG)
        # )
        self.scheduler_config = scheduler_config
        # self._init_first_stage(first_stage_config)

        # self.loss_fn = (
        #     instantiate_from_config(loss_fn_config)
        #     if loss_fn_config is not None
        #     else None
        # )


        # self.use_ema = use_ema
        # if self.use_ema:
        #     self.model_ema = LitEma(self.model, decay=ema_decay_rate)
        #     print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # self.scale_factor = scale_factor
        # self.disable_first_stage_autocast = disable_first_stage_autocast
        # self.no_cond_log = no_cond_log
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_key,
            **pipe_kwargs
        ).to(self.device)

        self.pipe = pipe 

        for p in self.pipe.text_encoder.parameters():
            p.requires_grad_(False)
        for p in self.pipe.unet.parameters():
            p.requires_grad_(True)
        for p in self.pipe.vae.parameters():
            p.requires_grad_(False)

        self.denoiser = pipe.scheduler #DDPMScheduler.from_pretrained(model_key, subfolder="scheduler")
        #self.pipe.scheduler.set_timesteps(100)
        # self.denoiser.alphas_cumprod = self.denoiser.alphas_cumprod.to('cuda')
        self.loss_fn = StandardDiffusionLoss(self.pipe, self.denoiser).to(self.device)

        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.unet = pipe.unet  
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    # def on_after_backward(self) -> None:
    #     for name,p in self.named_parameters():
    #         if p.grad is not None:
    #             print(name)
    #     pass

    def init_from_ckpt( #add ability to restore weights from trained diffusion model
        self,
        path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError
        if  sd['model.diffusion_model.input_blocks.0.0.weight'].shape[1] == 4:
            sd['model.diffusion_model.input_blocks.0.0.weight'] = F.pad(input=sd['model.diffusion_model.input_blocks.0.0.weight'], pad=(0,0,0,0,0,5), mode='constant', value=0)
        else:
            assert sd['model.diffusion_model.input_blocks.0.0.weight'].shape[1] == 9
        #sd['model.diffusion_model.label_emb.0.0.weight'] = F.pad(input=sd['model.diffusion_model.label_emb.0.0.weight'], pad=(0,256,0,0), mode='constant', value=0)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    # def _init_first_stage(self, config):
    #     model = instantiate_from_config(config).eval()
    #     model.train = disabled_train
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format

        # TODO: downscale to [512,512]
        img = batch[self.input_key]
        img = F.interpolate(img, size=(512, 512))
        mask = batch[self.mask_input_key]
        mask = F.interpolate(mask, size=(512, 512))
        masked_img = img*(mask < 0.5) #+ (mask > 0.5)*0.5 #img*(mask < 0) #+ (mask >0)*1

        return img, masked_img, mask

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            out = self.first_stage_model.decode(z)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.vae.encode(x).latent_dist.sample()*self.vae.config.scaling_factor

    def forward(self, x, batch):
        loss = self.loss_fn(x, batch)
        loss_mean = loss.mean()
        #x.requires_grad_(True)
        #loss_mean = x.mean()*0
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x, masked_img, mask = self.get_input(batch)
        # batch['instance_images_downsampled'] = x.clone()
        # batch['instance_masks_downsampled'] = mask.clone()
        x = self.encode_first_stage(x)#self.vae.encode(x).latent_dist.sample()
        masked_latents = self.encode_first_stage(masked_img)#self.vae.encode(masked_img).latent_dist.sample()
        mask = F.interpolate(mask, size=(masked_latents.shape[2], masked_latents.shape[3]))
        mask = mask.mean(dim=1, keepdim=True)
        batch['masked_latents'] = masked_latents
        batch['interpolated_mask'] = mask
        # resize mask.
        # concat to x
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

    def on_train_start(self, *args, **kwargs):
        # if self.sampler is None or self.loss_fn is None:
        #     raise ValueError("Sampler and loss function need to be set for training.")
        pass

    def on_train_batch_end(self, *args, **kwargs):
        # if self.use_ema:
        #     self.model_ema(self.model)
        pass

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            val_loss, _ = self.shared_step(batch)
            
            self.log(
            "validation_loss",
            val_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            )

            self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            )

            batch_size = 1
            log = self.log_images(batch, batch_size, sample=True)
            samples = log['samples']#.permute([0,2,3,1])
            gt = log['gt'].permute([0,2,3,1])
            masked_image = log['masked_gt'].permute([0,2,3,1])
            mask = log['mask'].permute([0,2,3,1])
            gt_image = torch.cat([gt, masked_image, mask], dim=2)
            result_image = torch.cat([samples, masked_image[:batch_size], mask[:batch_size]], dim=2)
            val_save_path = os.path.join(self.logger.log_dir, str(self.global_step))
            if not os.path.exists(val_save_path):
                Path(val_save_path).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(val_save_path,'info.txt'), 'w') as f:
                f.write(batch['txt'][0]+'\n')
                f.write(batch['mask_index'][0])
            iio.imwrite(os.path.join(val_save_path,'gt.png'), gt_image.clip(0,1).cpu().numpy()[0])
            iio.imwrite(os.path.join(val_save_path,'output.png'), result_image.clip(0,1).cpu().numpy()[0])

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.unet.parameters())
        # for embedder in self.conditioner.embedders:
        #     if embedder.is_trainable:
        #         params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        concat: torch.Tensor = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device) # concat the encoded masked image and downscaled map to randn
        # if concat is not None:
        #     randn = torch.cat((randn, concat), dim=1)
        # if concat is not None:
        #     cond['concat'] = concat

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc, **kwargs)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None, #can be a subset of the given condition keys
        **kwargs,
    ) -> Dict:
        # conditioner_input_keys = [e.input_key for e in self.conditioner.embedders] #full ist: ['txt', 'txt', 'original_size_as_tuple', 'crop_coords_top_left', 'target_size_as_tuple', 'mask_type']
        # if ucg_keys:
        #     assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
        #         "Each defined ucg key for sampling must be in the provided conditioner input keys,"
        #         f"but we have {ucg_keys} vs. {conditioner_input_keys}"
        #     )
        # else:
        #     ucg_keys = conditioner_input_keys
        log = dict()

        # x, masked_image, mask = self.get_input(batch)

        # log['gt'] = x
        # log['masked_gt'] = x*(mask < 0) #+ (mask >= 0)*1
        # log['mask'] = mask

        # c, uc = self.conditioner.get_unconditional_conditioning(
        #     batch,
        #     force_uc_zero_embeddings=ucg_keys #list of keys to ignore in conditioning
        #     if len(self.conditioner.embedders) > 0
        #     else [],
        # )

        # sampling_kwargs = {}

        # N = min(x.shape[0], N)
        # x = x.to(self.device)[:N]
        # log["inputs"] = x
        # z = self.encode_first_stage(x)
        # log["reconstructions"] = self.decode_first_stage(z)
        # log.update(self.log_conditionings(batch, N))

        # for k in c:
        #     if isinstance(c[k], torch.Tensor):
        #         c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        # uc_concat = torch.zeros_like(x, device='cuda')
        # uc_concat = self.encode_first_stage(uc_concat)+
        # uc_concat = torch.nn.functional.pad(uc_concat, (0, 0, 0, 0, 1, 0), value=1.0)

        # uc['concat'] = uc_concat

        # masked_latents = self.encode_first_stage(masked_image) 
        # mask = F.interpolate(mask, size=(masked_latents.shape[2], masked_latents.shape[3]))
        # mask = mask.mean(dim=1, keepdim=True)
        # c['concat'] = torch.cat([mask[:N],masked_latents[:N]],dim=1)
        # if type(self.sampler).__name__ == 'BlendedDiffusionSampler':
        #     c['init_z'] = z
        #     c['mask'] = mask[:N]
        # if sample:
        #     with self.ema_scope("Plotting"):
        #         samples = self.sample(
        #             c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
        #         )
        #     samples = self.decode_first_stage(samples)
        #     log["samples"] = samples

        x, masked_image, mask = self.get_input(batch)

        x = x.to(self.device)[:N]
        mask = mask.to(self.device)[:N,:,:,:]

        log['gt'] = x
        log['masked_gt'] = x*(mask < 0.5) #+ (mask >= 0.5)*0.5
        log['mask'] = mask*2-1

        prompt = batch['txt'][:N]

        # convert x from (-1,1) to (0,1)
        images = np.concatenate(self.pipe.to(self.device)(prompt = prompt, image = x, mask_image = mask[:,[0],:,:], output_type='np', num_inference_steps = 50).images, axis=0)

        log['samples'] = torch.tensor(images, dtype = torch.float, device = self.device)*2-1
        if len(log['samples'].shape) == 3:
            log['samples'] = log['samples'].unsqueeze(0)
        
        return log