import argparse
import datetime
import glob
import inspect
import os
import sys
from inspect import Parameter

import numpy as np
import torch
from omegaconf import OmegaConf
# from packaging import version
from PIL import Image
from pytorch_lightning import seed_everything
# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from diffusion import DiffusionEngine
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#from sgm.util import exists, instantiate_from_config, isheatmap change to load from diffusers
import imageio as iio
from dataset import get_dataloader_from_config

# MULTINODE_HACKS = True

def default_trainer_args():
    argspec = dict(inspect.signature(Trainer.__init__).parameters)
    argspec.pop("self")
    default_args = {
        param: argspec[param].default
        for param in argspec
        if argspec[param] != Parameter.empty
    }
    return default_args


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
    )
    parser.add_argument(
        # "-b",
        "--config",
        # nargs="*",
        # metavar="base_config.yaml",
        required=True,
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="seed for seed_everything",
    )

    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="path to initial checkpoint to train from. does not have to be an inpainting checkpoint (e.g stable-diffusion-v1-5/stable-diffusion-v1-5)"
    )
    # parser.add_argument(
    #     "--scale_lr",
    #     type=str2bool,
    #     nargs="?",
    #     const=True,
    #     default=False,
    #     help="scale base-lr by ngpu * batch_size * n_accumulate",
    # )

    default_args = default_trainer_args()
    for key in default_args:
        parser.add_argument("--" + key, default=default_args[key])
    return parser

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    parser = get_parser()

    #torch.cuda.set_device('cuda:1')
    opt, unknown = parser.parse_known_args()

    print(f"LOGDIR: {opt.logdir}")
    os.makedirs(opt.logdir, exist_ok=True)

    #opt.seed = 150
    seed_everything(opt.seed, workers=True)

    config = OmegaConf.load(opt.config)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, cli)
    with open(os.path.join(opt.logdir, "config.yaml"), "w") as f:
        OmegaConf.save(config, f)
    
    dataloader, val_dataloader = get_dataloader_from_config(config.data)

    pipe_kwargs = {
            "torch_dtype": torch.float32,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            #"revision" : "fp16"
        }
    config.model.params.loss_fn_config['target'] = 'loss.StandardDiffusionLoss'

    model = DiffusionEngine(opt.checkpoint_path, pipe_kwargs = pipe_kwargs)
    model.learning_rate = config.model.base_learning_rate
    
    #model = model.to('cuda')
    
    from pytorch_lightning.loggers import CSVLogger
    logger = CSVLogger(save_dir=os.path.join(opt.logdir, 'logs/'))
    
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(opt.logdir, 'checkpoints'), **config.modelcheckpoint,)

    trainer = Trainer(
            default_root_dir=opt.logdir,
            logger=logger,
            callbacks=[checkpoint_callback],
            limit_val_batches=1,
            **config.trainer,
            )
    
    trainer.fit(model, dataloader, val_dataloader)
    # trainer.fit(model, dataloader)