
import torch
import argparse
from pathlib import Path
import json
from rendering.material import DiffuseWithPointLightMaterial
from cameras.star import StarCameraIterableDataset
from cameras.I3D_camera import I3D_cameras
from rendering.mesh_renderer import NVDiffRasterizer
import trimesh
import igl
import imageio as iio
from ROAR.util.func import save_images
from utils.mv_utils import split_2x2_image
import numpy as np
from PIL import Image
import shutil
from ROAR.util.embree_utils import RayRatioVisibility
from geometry.dynamic_mesh import DynamicMesh
import cyobj.io as mio
from utils.config import load_config
from rendering.background import SolidColorBackground
from utils.ops import get_activation
import os
import xatlas
from ROAR.util.geometry_utils import calculate_vertex_normals
import nvdiffrast.torch as dr

glctx = dr.RasterizeCudaContext()

# for uv extrusion
def supercat(tensors, dim: int = 0):
    """
    Similar to `torch.cat`, but supports broadcasting. For example:

    [M, 32], [N, 1, 64] -- supercat 2 --> [N, M, 96]
    """
    ndim = max(x.ndim for x in tensors)
    tensors = [x.reshape(*[1] * (ndim - x.ndim), *x.shape) for x in tensors]
    shape = [max(x.size(i) for x in tensors) for i in range(ndim)]
    shape[dim] = -1
    tensors = [torch.broadcast_to(x, shape) for x in tensors]
    return torch.cat(tensors, dim)

def gpu_f32(inputs):
    return torch.tensor(inputs, dtype=torch.float32, device=torch.device('cuda:0')).contiguous()

def saturate(x: torch.Tensor):
    return torch.clamp(x, 0.0, 1.0) 

def float4(*tensors):
    tensors = [x if torch.is_tensor(x) else gpu_f32(x) for x in tensors]
    tensors = supercat(tensors, dim=-1)
    assert tensors.shape[-1] == 4
    return tensors

def alpha_blend(background: torch.Tensor, foreground: torch.Tensor):
    alpha = saturate((foreground[:,:,[3]] > 0).float())
    return background * (1 - alpha) + foreground * alpha

def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

@torch.no_grad()
def extrude(x: torch.Tensor, iterations: int = 16):
    kernel = torch.tensor([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])[None, None].to(x)
    for _ in range(iterations):
        extrude = x
        rgb = extrude[:,:,:3]
        alpha = (extrude[:,:,[3]] > 0).float()
        extrude = float4(rgb * alpha, alpha)
        x = extrude
        extrude = extrude.permute(2, 0, 1)[:, None]
        extrude = torch.conv2d(extrude, kernel, padding=1)
        extrude = extrude.squeeze(1).permute(1, 2, 0)
        rgb = extrude[:,:,:3]
        alpha = extrude[:,:,[3]]
        extrude = float4(rgb*(alpha > 0).float() / (alpha + 1e-7), (alpha > 0).float())
        x = alpha_blend(extrude, x)
    return x.clip(0,1)

def render_uv(init_mesh, target_mesh, init_material, target_material, resolution = [512,512]):

    with torch.no_grad():
        mesh_vertices = init_mesh.mesh_optimizer.vertices
        mesh_faces = init_mesh.mesh_optimizer.faces
    # clip space transform 

    vmapping, indices, uvs = xatlas.parametrize(mesh_vertices.detach().cpu().numpy(), mesh_faces.cpu().numpy())
    FT = torch.tensor(indices.astype(np.int32), device = mesh_vertices.device)

    uv_clip = torch.tensor(uvs[None, ...]*2.0 - 1.0, device = mesh_vertices.device)

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[...,0:1]), torch.ones_like(uv_clip[...,0:1])), dim = -1)

    # rasterize
    rast, _ = dr.rasterize(glctx, uv_clip4, FT.int(), resolution)

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh_vertices[None, ...], rast, mesh_faces.int())
    
    # TODO: find which face in rast is in allowed faces
    allowed_faces, _ = init_mesh.mesh_optimizer.get_allowed_faces_and_edges(mode='relaxed')
    allowed_faces = torch.nn.functional.pad(allowed_faces,(1,0))
    pixels_in_allowed_faces = allowed_faces[rast.to(dtype=torch.int64)][...,3].view(1, -1)

    sh = gb_pos.shape
    shaded_col = init_mesh.forward_color(gb_pos.view(1, -1, 3)).clip(0,1)
    if pixels_in_allowed_faces.shape[-1] > 0:
        shaded_col[pixels_in_allowed_faces] = target_mesh.forward_color(gb_pos.view(1, -1, 3)[pixels_in_allowed_faces.squeeze(-1)]).clip(0,1)
    shaded_col = shaded_col.view(*sh)        

    # fix the texture seams
    shaded_col = extrude(torch.cat((shaded_col[0], rast[0, ...,[3]]), dim=-1), iterations=3)

    return shaded_col.float()


def export_with_texture(init_mesh, init_material, target_mesh, target_material, save_dir = ''):

    with torch.no_grad():
        mesh_vertices = init_mesh.mesh_optimizer.vertices
        mesh_faces = init_mesh.mesh_optimizer.faces

        # output texture of mesh as .mtl file
        vmapping, indices, uvs = xatlas.parametrize(mesh_vertices.cpu().numpy(), mesh_faces.cpu().numpy())
        VT = mesh_vertices[vmapping.astype(np.int32)]
        VN = calculate_vertex_normals(mesh_vertices, mesh_faces)
        FT = indices.astype(np.int64)
        uvs = torch.tensor(uvs, device = mesh_vertices.device)
        indices = torch.tensor(indices.astype(np.int64), device = mesh_vertices.device, dtype = torch.long)
        im = render_uv(init_mesh, target_mesh, init_material, target_material)
        image = Image.fromarray((im.squeeze(0)*255).cpu().numpy().astype(np.uint8))
        image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
        image.save(os.path.join(save_dir,"texture.png"))
        uvs = uvs[:,:2].cpu().numpy()
        rotation_matrix = torch.tensor([[1,0,0],
                                        [0,0, 1],
                                        [0,-1,0]], device = 'cuda', dtype=torch.float).unsqueeze((0))
        mesh_vertices = torch.bmm(rotation_matrix.expand([mesh_vertices.shape[0],-1,-1]),  mesh_vertices.unsqueeze(-1))
        mesh_vertices = mesh_vertices.squeeze(-1)
        mio.write_obj(os.path.join(save_dir,"final_mesh.obj"), mesh_vertices.detach().cpu().numpy().astype(np.float64), mesh_faces.cpu().numpy().astype(np.int64), uvs.astype(np.float64), FT, None, None, path_img = "texture.png")


def read_mtlfile(fname):
    materials = {}
    with open(fname) as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        if line:
            prefix, data = line.split(' ', 1)
            if 'newmtl' in prefix:
                material = {}
                materials[data] = material
            elif materials:
                if len(data.split(' ')) > 1:
                    material[prefix] = tuple(float(d) for d in data.split(' '))
                else:
                    if data.isdigit():
                        try:
                            material[prefix] = int(data)
                        except ValueError:
                            material[prefix] = float(data)
                    else:
                        material[prefix] = data

    return materials



def load_target_data(path):
    mesh_path = Path(path, 'target_mesh.obj')
    colors = Path(path,'vertex_colors.npy')
    if colors.exists(): # target was .ply
        return ['system.geometry.initial_mesh_path={}'.format(str(mesh_path)), 'system.geometry.vertex_colors_path={}'.format(colors)] 
    else: # target was .obj
        mtl_path = Path(path, 'lrm_mesh.mtl')
        _,vc,_,_,fc,_ = igl.read_obj(str(Path(path, 'lrm_mesh.obj')))
        v,_,_,f,_,_ = igl.read_obj(str(Path(path, 'target_mesh.obj')))
        if not mtl_path.exists():
            vc = None
            fc = None
            texture_image_name = None 
        else:
            material = read_mtlfile(str(mtl_path))
            texture_image_name = Path(material[next(iter(material))]['map_Kd'])
            shutil.copy(Path(path,texture_image_name), Path(path, 'mesh_evolution'))
        out_mesh_path = Path(path, 'target_mesh.obj')
        out_mtl_path = Path(path, 'target_mesh.mtl')
        mio.write_obj(str(out_mesh_path), v, f, vc, fc, None, None, path_img = 'lrm_mesh.png')
        if not mtl_path.exists():
            mtl_path = ""
        allowed_vertices_path = '\"\"'
        return ['system.geometry.initial_mesh_path={}'.format(str(out_mesh_path)), 'system.geometry.initial_mtl_path={}'.format(str(out_mtl_path)), 'system.geometry.allowed_vertices_path={}'.format(str(allowed_vertices_path))]


def load_input_data(path, out_path,do_normalize_scene = True, scale = 1, offset = 0):
    mesh_path = Path(path,'mesh.obj')
    mtl_path = Path(path, 'mesh.mtl')
    v,vc,_,f,fc,_ = igl.read_obj(str(mesh_path))
    v = torch.tensor(v, device = 'cuda')
    if do_normalize_scene:
        #v  = normalize_scene(torch.tensor(v, device = 'cuda'))
        print('normalizing...')
        v += offset
        v *= 1.5/scale
    if not mtl_path.exists():
        vc = None
        fc = None
        texture_image_name = None 
    else:
        material = read_mtlfile(str(mtl_path))
        texture_image_name = Path(material[next(iter(material))]['map_Kd'])
        shutil.copy(Path(path,texture_image_name), Path(out_path.parent, 'mesh_evolution'))
    out_mesh_path = Path(out_path.parent, 'mesh_evolution', 'mesh.obj')
    out_mtl_path = Path(out_path.parent, 'mesh_evolution', 'mesh.mtl')
    mio.write_obj(str(out_mesh_path), v.detach().cpu().numpy().astype(np.float64), f, vc, fc, None, None, path_img = texture_image_name.stem + '.png')
    allowed_vertices_path = Path(path,'allowed_vertices_ids.txt')
    if not mtl_path.exists():
        mtl_path = ""
    if not allowed_vertices_path.exists():
        allowed_vertices_path = '\"\"'
    return ['system.geometry.initial_mesh_path={}'.format(str(out_mesh_path)), 'system.geometry.initial_mtl_path={}'.format(str(out_mtl_path)), 'system.geometry.allowed_vertices_path={}'.format(str(allowed_vertices_path))]


def evolve_mesh(args, root_path, target_path, normal_estimator = None, save_video = False, rescaling_data = None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, nargs='+', help="path to config file")
    args, extras = parser.parse_known_args(args)

    input_data_folder = Path(root_path)#, "blender_data") 

    # with open(Path(target_path, 'data.json')) as f:
    #     data = json.load(f)

    offset = 0
    scale = 1
    if rescaling_data is not None:
        offset = torch.tensor(rescaling_data['offset'], device='cuda')
        scale = rescaling_data['scale']
    input_data = load_input_data(input_data_folder,target_path, do_normalize_scene =  rescaling_data is not None, scale = scale, offset = offset) # do_normalize_scene=False)

    # load target data
    #target_data_path = Path(target_path.parent, 'mesh_evolution')#(root_path,"LRM_data", "lrm_mesh") 
    output_data = load_target_data(target_path.parent)

    init_cfg = load_config(args.config[0], cli_args=input_data, n_gpus=1)
    target_cfg = load_config(args.config[1], cli_args=output_data, n_gpus=1)
    init_mesh = DynamicMesh()
    init_mesh.configure(init_cfg.system.geometry,'cuda')

    init_material = DiffuseWithPointLightMaterial().to('cuda')
    init_material.configure()
    # # TODO: Initialize Target mesh as DynamicMesh
    target_mesh = DynamicMesh()
    target_mesh.configure(target_cfg.system.geometry,'cuda')
    target_material = DiffuseWithPointLightMaterial().to('cuda')
    target_material.configure()
    init_background = SolidColorBackground().to('cuda')
    init_background.configure()

    renderer_init = NVDiffRasterizer().to('cuda')
    renderer_init.configure( init_mesh, init_material, init_background)
    renderer_target = NVDiffRasterizer().to('cuda')
    renderer_target.configure(target_mesh, target_material, init_background)

    # generate star cameras
    cameras = StarCameraIterableDataset(init_cfg.data)
    cameras_data = cameras.collate(batch = {'light_positions': torch.tensor([1,1,1], device = 'cuda' ).unsqueeze(0).repeat([16,1])})

    # generate fixed cameras
    CameraSetup = I3D_cameras('cameras/opencv_cameras.json')
    fixed_camera_batch = CameraSetup({'azimuth': torch.tensor(0, dtype=torch.float)})
    
    #input_data = load_input_data(input_data_folder, do_normalize_scene = True)
    
    init_output = renderer_init(**cameras_data, use_supersampled=True)
    save_images(init_output['comp_rgb'].flip(dims=[2]), Path(target_path.parent,'mesh_evolution', 'initial_images'))
  
    if normal_estimator is not None:
        fixed_init_output = renderer_init(**fixed_camera_batch, use_supersampled=True)
        fixed_init_output['comp_normal'] = fixed_init_output['comp_normal'].flip(dims=[1])
        save_images(fixed_init_output['comp_normal'], Path(target_path,'mesh_evolution', 'initial_images_fixed_cameras'))
        
        input_image = iio.imread(Path(target_path.parent,'turntable_cameras', 'result_0.png'))
        input_images = split_2x2_image(input_image, return_list = True)
        input_images = [Image.fromarray(img) for img in input_images]
        pred_depth = []
        pred_normal = []
        c2w = CameraSetup.fixed_cameras_data['c2w']
        rotations = c2w[:,:3,:3]
        for i,input_image in enumerate(input_images):
            pipe_out = normal_estimator(input_image,
                denoising_steps = 10,
                ensemble_size= 3,
                processing_res = 0,
                match_input_res = False,
                domain = 'object',
                color_map = 'Spectral',
                show_progress_bar = True,
            )
            pred_depth.append(torch.tensor(pipe_out.depth_np, device='cuda'))
            pred_normal.append(torch.tensor(pipe_out.normal_np , device='cuda').permute([1,2,0]))
            rotation_matrix = torch.tensor([[1, 0,  0],
                                            [0, 1, 0],
                                            [0, 0,  1]], device = 'cuda', dtype=torch.float).unsqueeze((0))
            
            pred_normal[-1][...,0] *= -1
            #pred_normal[-1] = torch.bmm(rotation_matrix.expand([pred_normal[-1].shape[0]*pred_normal[-1].shape[1],-1,-1]), pred_normal[-1].reshape([-1,3,1])).reshape(*pred_normal[-1].shape)
            pred_normal[-1] = torch.bmm(rotations[[i]].expand([pred_normal[-1].shape[0]*pred_normal[-1].shape[1],3,3]), pred_normal[-1].reshape([-1,3,1])).reshape(*pred_normal[-1].shape)
        
        input_images = torch.stack([torch.tensor(np.array(i), device='cuda') for i in input_images]).float()/255
        pred_normal = torch.stack(pred_normal)
        pred_depth = torch.stack(pred_depth)
        rotations = CameraSetup.fixed_cameras_data['c2w'][:,:3,:3]
        pred_normal = (pred_normal+1)/2
        pred_normal[(pred_depth>0.9)] = 0
        save_images(pred_normal, Path(target_path.parent,'mesh_evolution', 'target_images_fixed_cameras'))

    target_output = renderer_target(**cameras_data, use_supersampled=True)
    save_images(target_output['comp_rgb'].flip(dims=[2]), Path(target_path.parent,'mesh_evolution', 'target_images'))
    # create low res mesh for rayCasting
    rayCaster = RayRatioVisibility()
    if target_mesh.mesh.faces.shape[0]>50000:
        _,U,G,J,I = igl.qslim(target_mesh.mesh.vertices.detach().cpu().numpy(), target_mesh.mesh.faces.cpu().numpy(),50000)
    else:
        U = target_mesh.mesh.vertices.detach().cpu().numpy()
        G = target_mesh.mesh.faces.cpu().numpy()
    rayCaster.set_up(U ,G)

    init_mesh.mesh_optimizer.rayCaster = rayCaster
    face_score, _, debug_info = init_mesh.mesh_optimizer.calculate_face_score(output_debug_info=True)
    allowed_faces, _ = init_mesh.mesh_optimizer.get_allowed_faces_and_edges()
    debug_info['projected_points_arr'][~allowed_faces[init_mesh.mesh_optimizer.reg_sampler.face_ids].cpu().numpy()] = init_mesh.mesh_optimizer.reg_sampler.sampled_vertices[~allowed_faces[init_mesh.mesh_optimizer.reg_sampler.face_ids]].cpu().numpy()
    face_score[~allowed_faces] = 0
    if save_video:
        video_path = Path(root_path,'output', 'video.mp4')
        video_imgs = [init_output['comp_rgb'][0].clip(0,1).detach().cpu().numpy()]

    for i in range(800):
        if i % 100 ==0:
            print(i, flush=True)
        init_mesh.zero_grad()
        #target_mesh.zero_grad()
        init_output = renderer_init(**cameras_data, color_net = target_mesh.forward_color, use_supersampled=True)

        if normal_estimator is not None:
            init_output_fixed = renderer_init(**fixed_camera_batch, color_net = target_mesh.forward_color, use_supersampled=True)

        if save_video:
            video_imgs.append(init_output['comp_rgb'][0].clip(0,1).detach().cpu().numpy())
        loss = (init_output['comp_normal'] - target_output['comp_normal'].detach()).abs().mean()
        loss += (init_output['comp_rgb'] - target_output['comp_rgb'].detach()).abs().mean()
        if normal_estimator is not None:
            loss += 2*(init_output_fixed['comp_normal'].flip(dims=[1]) - pred_normal.detach()).abs().mean()
            loss += 2*(init_output_fixed['comp_rgb'].flip(dims=[1]) - input_images.detach()).abs().mean()
        loss.backward()
        #print(loss.item())
        init_mesh.step()
        with torch.no_grad():
            init_mesh.remesh(i)

    if save_video:
        iio.mimsave(video_path, video_imgs, fps=15)
    save_images(init_output['comp_rgb'].flip(dims=[2]), Path(target_path.parent,'mesh_evolution', 'final_images'))
    if normal_estimator is not None:
        save_images(init_output_fixed['comp_normal'].flip(dims=[1]) , Path( target_path.parent,'mesh_evolution', 'final_images'))
    #save_mesh_properly(init_mesh.mesh.vertices, init_mesh.mesh.faces, Path(root_path,'output','output_mesh.obj'))
    export_with_texture(init_mesh, init_material, target_mesh, target_material, save_dir=str(Path(target_path.parent,'mesh_evolution')))
    np.savetxt(Path(Path(target_path.parent,'mesh_evolution'),'allowed_vertices_ids.txt'), torch.where(init_mesh.allowed_vertices)[0].cpu().numpy())

    obj_path = Path(target_path.parent,'mesh_evolution', 'final_mesh.obj')
    mtl_path = Path(target_path.parent,'mesh_evolution', 'final_mesh.mtl')
    texture_path = Path( target_path.parent,'mesh_evolution', 'texture.png')
    return {'obj':obj_path, 'mtl': mtl_path, 'texture' :texture_path}


parser = argparse.ArgumentParser()
parser.add_argument("--mesh_input_path", type = str, required=True)
parser.add_argument("--lrm_mesh_output_path", type = str, required=True)
parser.add_argument("--use_normal_estimation", action = "store_true")
opt, unknown = parser.parse_known_args()


guidance_mesh_path = Path(opt.lrm_mesh_output_path)

# load mesh with colors 
guidance_mesh = trimesh.load(guidance_mesh_path)
# TODO: rotate mesh for zero azimuth offset
#if not opt.no_scale_data:
with open(Path(guidance_mesh_path.parent,'data.json'), 'r') as f:
    data = json.load(f)
rot = trimesh.transformations.rotation_matrix(-data['azimuth_offset']/90*np.pi, [0, 0, 1])
guidance_mesh = guidance_mesh.apply_transform(rot)

guidance_mesh_vertices = guidance_mesh.vertices
guidance_mesh_faces = guidance_mesh.faces

if guidance_mesh_path.suffix == '.ply':
    guidance_mesh_colors = guidance_mesh.visual.vertex_colors[:,:3]/255
    igl.write_obj(str(Path(guidance_mesh_path.parent, 'target_mesh.obj')), guidance_mesh_vertices, guidance_mesh_faces)
    np.save(Path(opt.lrm_mesh_output_path, 'color_vertex.npy'), guidance_mesh_colors)
elif guidance_mesh_path.suffix == '.obj':
    igl.write_obj(str(Path(guidance_mesh_path.parent, 'target_mesh.obj')), guidance_mesh_vertices, guidance_mesh_faces)
else:
    raise NotImplementedError('LRM mesh must be .ply or .obj!, recieved {}'.format(guidance_mesh_path.suffix))


mesh_evolution_path = Path(guidance_mesh_path.parent, "mesh_evolution")
mesh_evolution_path.mkdir(exist_ok = True, parents=True)

# if not opt.no_scale_data:
with open(Path(guidance_mesh_path.parent,'data.json'), 'r') as file:    
    rescaling_data = json.load(file)
# else:
#     rescaling_data = None

normal_estimator = None

evolve_mesh_args = [
            "--config",
            "configs/LRM/lrm_reconstruction.yaml",
            "configs/LRM/target_lrm_reconstruction.yaml",
        ]


output_image_grid_input_path = Path(guidance_mesh_path.parent, 'a_tanuki_riding_a_beagle.png')

evolve_mesh(evolve_mesh_args, opt.mesh_input_path, output_image_grid_input_path, normal_estimator = normal_estimator, rescaling_data = rescaling_data)
