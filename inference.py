import torch
import argparse  
from cameras.I3D_camera import I3D_cameras
from rendering.renderer import get_renderer
from ROAR.util.geometry_utils import calculate_vertex_normals
from pathlib import Path
from scipy.spatial import ConvexHull
import imageio as iio
import datetime
import numpy as np
import json
from PIL import Image
from torchvision.transforms import v2
from utils.camera_utils import convert_c2w_between_opengl_opencv, get_K_from_intrinsics, FOV_to_intrinsics, get_zero123plus_input_cameras
import sys
parser = argparse.ArgumentParser()


def split_2x2_image(img, return_list = False):
    imgs = []
    imgs.append(img[512:,512:])
    imgs.append(img[512:,:512])
    imgs.append(img[:512,512:])
    imgs.append(img[:512,:512])

    if return_list:
         return imgs
    return torch.stack(imgs,dim=0)

def create_grid_image(images_tensor, flip_image = False):
    images_tensor = images_tensor.permute([0,3,1,2]).unsqueeze(0)
    imgs0,imgs1,imgs2,imgs3 = images_tensor.split(1, dim=1)
    if flip_image:
        imgs0 = imgs0.flip(dims=[4])
        imgs1 = imgs1.flip(dims=[4])
        imgs2 = imgs2.flip(dims=[4])
        imgs3 = imgs3.flip(dims=[4])
    pred_rgb = torch.concatenate([torch.concatenate([imgs3,imgs2], dim=-1), torch.concatenate([imgs1,imgs0], dim=-1)],dim=-2)
    pred_rgb = pred_rgb.squeeze(1)
    return pred_rgb

def normalize_mesh(mesh, mask,):
    with torch.no_grad():
        mesh_v = mesh.vertices
        mask_v = mask.vertices
        v = torch.cat([mesh_v,mask_v], dim=0)
        offset = -(v.max(dim=0)[0] + v.min(dim=0)[0])/2
        scale = (v.max(dim=0)[0] - v.min(dim=0)[0]).max()
        mesh_v += offset
        mesh_v *= 1.5/scale
        mask_v += offset
        mask_v *= 1.5/scale
    return offset,scale

def process_mask(mask, dilation, make_ch):
    with torch.no_grad():
        v = mask.vertices
        f = mask.faces
        vn = calculate_vertex_normals(v,f)
        v += vn*dilation
        if make_ch:
            ch = ConvexHull(v.cpu().numpy())
            v, f = torch.tensor(ch.points, device = 'cuda'), torch.tensor(ch.simplices, dtype = torch.int32, device = 'cuda')

### Run Options #####
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--mask_path", type=str, required=True)
parser.add_argument("--instant3dit_model_path", type=str, default="Path/to/Instant3dit_model")
parser.add_argument("--instantMesh_path", type=str, default="Path/to/InstantMesh", help = "Required to use the mesh LRM")
parser.add_argument("--geoLRM_path", type=str, default="Path/to/geoLRM", help = "Required to use the 3DGS LRM")
parser.add_argument("--input_type", type=str, default="mesh",  choices=['mesh'])
parser.add_argument("--mask_type", type=str, default="mesh", choices=['mesh', 'mv_depth_image'])
parser.add_argument("--output_type", type=str, default="mv_grid_image", choices=['mv_grid_image', 'mesh', '3DGS'])
parser.add_argument("--output_path", type=str, default="experiments")

### Inference Options #####
parser.add_argument("--prompt", required=True, help="prompt for inpainting")
parser.add_argument("--seed", type=int, default = 10000)
parser.add_argument("--fp16", type=bool, default=False , help="use half precision weights")

### Mask and Input Options ###
parser.add_argument("--flip_image", action = 'store_true', default=False)

### if mask type is mesh ###
parser.add_argument("--mask_dilation", type=float, default=0.05, help="amount of iterations for mask dilation")
parser.add_argument("--guassian_blur_iters", type=int, default=0, help="amount of iterations for mask dilation")
parser.add_argument("--azimuth_offset", type=float , default=0.0, help="in degrees")
parser.add_argument("--use_mask_convex_hull", action = 'store_true', default=False)

args, extras = parser.parse_known_args()

timestamp =  str(datetime.datetime.now()).replace(' ', '_')

output_path = Path(args.output_path, args.prompt.replace(' ', '_') + timestamp)

output_path.mkdir(exist_ok=True, parents=True)

deg = args.azimuth_offset
CameraSetup = I3D_cameras('cameras/opencv_cameras.json')
c2w, intrinsics = get_zero123plus_input_cameras(azimuth_offset = 0)
cameras = CameraSetup({'azimuth': torch.tensor(-deg, dtype=torch.float), 'c2w': c2w.to('cuda'), 'intrinsic': intrinsics.to('cuda')})

if args.input_type in ['mesh', 'NeRF', '3DGS']:
    input_renderer = get_renderer(args.input_type, args.input_path, 'cuda')

if args.mask_type in ['mesh', 'NeRF', '3DGS']:
    mask_renderer = get_renderer(args.mask_type, args.mask_path, 'cuda')
    if args.mask_type == 'mesh':
        process_mask(mask_renderer.geometry, args.mask_dilation, args.use_mask_convex_hull) # optional

if args.input_type == 'mesh' and args.mask_type == 'mesh':
    offset, scale = normalize_mesh(input_renderer.geometry, mask_renderer.geometry)
    scale_dict = {'offset' : offset.cpu().numpy().tolist(), 'scale' : scale.cpu().numpy().tolist(), 'azimuth_offset' : args.azimuth_offset, 'flipped' : args.flip_image}
    with open(Path(output_path,'data.json'), 'w') as f:
        json.dump(scale_dict, f)

input_renders = input_renderer(**cameras)
mask_renders = mask_renderer(**cameras)

# create input grid images and save to output path
rgb_grid_image = create_grid_image(input_renders['comp_rgb'], args.flip_image)

pred_depth = input_renders['depth']
pred_depth1 = create_grid_image(pred_depth, flip_image = args.flip_image).repeat([1,3,1,1])

pred_depth = mask_renders['depth']
mask_pred_depth1 = create_grid_image(pred_depth,flip_image = args.flip_image).repeat([1,3,1,1])

mask_grid_image = (mask_pred_depth1*(1+1e-2) > pred_depth1)

overlay_grid_image = rgb_grid_image*(mask_grid_image<0.99) + \
    (torch.tensor([220,36,36], device='cuda')/255).reshape([1,-1,1,1])*(mask_grid_image>0.99)

iio.imwrite(Path(output_path, '2x2_image.png'), (rgb_grid_image[0].permute(1,2,0)*255).cpu().numpy().astype(np.uint8))
iio.imwrite(Path(output_path, '2x2_mask.png'), (mask_grid_image[0].permute(1,2,0)*255).cpu().numpy().astype(np.uint8))
iio.imwrite(Path(output_path, '2x2_mask_depth.png'), (mask_pred_depth1[0].permute(1,2,0)*255).cpu().numpy().astype(np.uint8))
iio.imwrite(Path(output_path, '2x2_image_depth.png'), (pred_depth1[0].permute(1,2,0)*255).cpu().numpy().astype(np.uint8))
iio.imwrite(Path(output_path, '2x2_overlay.png'), (overlay_grid_image[0].permute(1,2,0)*255).cpu().numpy().astype(np.uint8))

#load network

from diffusers.utils import load_image
from models.modified_SDXL_pipeline import ModifiedStableDiffusionXLInpaintPipeline, AutoencoderKL

prompt = [args.prompt]

model_key = args.instant3dit_model_path
precision_t = torch.float16 if args.fp16 else torch.float32

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    use_safetensors=True,
    torch_dtype=precision_t
)

with open('./TOKEN', 'r') as f:
    token = f.read().replace('\n', '')  # remove the last \n!
    print(f'[INFO] loaded hugging face access token from ./TOKEN!')

cache_dir = "pretrained/SDXL"

general_kwargs = {
    "cache_dir": cache_dir,
    "torch_dtype": precision_t,
    "use_safetensors": True,
    "variant": "fp16" if args.fp16 else None,
    "use_auth_token": token,
}

pipe = ModifiedStableDiffusionXLInpaintPipeline.from_pretrained(
    model_key,
    vae=vae,
    **general_kwargs
).to('cuda')

# inference
image = load_image(str(Path(output_path, '2x2_image.png'))).resize((1024, 1024))
mask_image = load_image(str(Path(output_path, '2x2_mask.png'))).resize((1024, 1024))

output = pipe(prompt = prompt, image = image, mask_image = mask_image,
            guidance_scale=8.0, 
            num_inference_steps=30,  # steps between 15 and 30 work well for us
            strength=0.99,  # make sure to use `strength` below 1.0
            generator=torch.Generator(device="cuda").manual_seed(args.seed)).images[0]

output.save(Path(output_path, '{}.png'.format(prompt[0].replace(' ', '_'))))

# output to LRM
print ('output_type : {}'.format(args.output_type), flush=True)

# Free VRAM used by the diffusion pipeline
del pipe

if args.output_type in ['mesh']:
    from LRMs.mesh import reconstruct_using_mesh_lrm
    c2w, intrinsics = get_zero123plus_input_cameras(azimuth_offset = args.azimuth_offset)
    extrinisic = c2w.flatten(-2)[:,:12].to('cuda')
    intrinsic = intrinsics.to('cuda')/512
    input_cameras = torch.cat([extrinisic,intrinsic], dim=-1).unsqueeze(0) # [1,4,16]
    images =  split_2x2_image(torch.tensor(iio.imread(Path(output_path, '{}.png'.format(prompt[0].replace(' ', '_'))))/255)).permute(0,3,1,2).unsqueeze(0)
    if not args.flip_image:
        images = images.flip(dims=[4])
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)
    images = images.to('cuda') # [1,4,3,320,320]
    sys.path.append(args.instantMesh_path)
    reconstruct_using_mesh_lrm(output_path, images, input_cameras)

if args.output_type == '3DGS':
    from LRMs.gaussians import reconstruct_using_3DGS_lrm
    c2w, intrinsics = get_zero123plus_input_cameras(radius=1.5, fov=30, azimuth_offset = args.azimuth_offset)
    intrinsic = FOV_to_intrinsics(39.6,device='cuda').reshape([1,1,3,3]).repeat([1,4,1,1]).float()
    c2w = c2w.to(device="cuda").unsqueeze(0).float()
    images =  split_2x2_image(torch.tensor(iio.imread(Path(output_path, '{}.png'.format(prompt[0].replace(' ', '_'))))/255)).permute(0,3,1,2).unsqueeze(0)
    if not args.flip_image:
        images = images.flip(dims=[4])
    images = v2.functional.resize(images, 448, interpolation=3, antialias=True).clamp(0, 1).float()
    images = images.to('cuda') # [1,4,3,448,448]
    input_Ks = intrinsic
    input_Ks = input_Ks.to(device='cuda') #[1,4,3,3]
    input_c2ws = c2w.to(device='cuda') #[1,4,4,4]
    sys.path.append(args.geoLRM_path)
    reconstruct_using_3DGS_lrm(output_path, images, input_c2ws, input_Ks)
