import torch
from omegaconf import OmegaConf
import os
import sys
from utils.camera_utils import FOV_to_intrinsics, get_circular_camera_poses
import imageio
import numpy as np

def save_video(
    frames: torch.Tensor,
    output_path: str,
    fps: int = 30,
) -> None:
    # images: (N, C, H, W)
    frames = [(frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for frame in frames]
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

def get_render_cameras(batch_size=1, M=120, radius=1.5, elevation=20.0):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    Ks = FOV_to_intrinsics(39.6).unsqueeze(0).repeat(M, 1, 1).float()
    c2ws = c2ws[None].repeat(batch_size, 1, 1, 1)
    Ks = Ks[None].repeat(batch_size, 1, 1, 1)
    return c2ws, Ks

def reconstruct_using_3DGS_lrm(output_path, images, input_c2ws, input_Ks):
    sys.path.append("/a/home/cc/cs/amirbarda/slurm_home_folder/GeoLRM")
    from src.geolrm_wrapper import GeoLRM
    gauss_path = str(output_path)
    video_path = str(output_path)
    name = 'guassian'
    distance = 1.5
    config_path = "/a/home/cc/cs/amirbarda/slurm_home_folder/GeoLRM/configs/geolrm.yaml"
    config = OmegaConf.load(config_path)
    config_name = os.path.basename(config_path).replace('.yaml', '')
    model_config = config.model_config
    infer_config = config.infer_config

    IS_FLEXICUBES = False

    device = torch.device('cuda')

    # load reconstruction model
    print('Loading 3DGS reconstruction model ...')
    model_config['params'].init_ckpt = '/a/home/cc/cs/amirbarda/slurm_home_folder/GeoLRM/ckpts/geolrm.ckpt'
    model = GeoLRM(**model_config['params'])

    model = model.to(device)
    model = model.eval()
    with torch.no_grad():
        # get latents
        xyzs, _ = model.serializer(images, input_c2ws, input_Ks)

        print('images shape: {}'.format(images.shape))
        print('input_c2ws shape: {}'.format(input_c2ws.shape))
        print('input_Ks shape: {}'.format(input_Ks.shape))

        latents = model.lrm_generator.forward_latents(xyzs, images, input_Ks, input_c2ws)

        # get gaussians
        gaussians = model.lrm_generator.renderer.get_gaussians(xyzs, latents)
        model.lrm_generator.renderer.save_ply(gaussians, os.path.join(gauss_path, f'{name}.ply'))

        # get video
        video_path_idx = os.path.join(video_path, f'{name}.mp4')
        render_size = infer_config.render_resolution
        render_c2ws, render_Ks = get_render_cameras(
            batch_size=1, 
            M=120,
            radius= 1.5, 
            elevation=20.0
        )
        render_c2ws, render_Ks = render_c2ws.to(device), render_Ks.to(device)


        out = model.lrm_generator.renderer.render(
            gaussians, 
            render_c2ws,
            render_Ks,
            render_size=render_size
        )
        frames = out["img"][0]

        save_video(
            frames,
            video_path_idx,
            fps=30,
        )
        print(f"Video saved to {video_path_idx}")
