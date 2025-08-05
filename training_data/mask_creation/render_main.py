import torch
import os
import bpy
import sys
sys.path.append('/home/abarda/magicclay/')
from mathutils import Vector
import igl
import numpy as np
from pathlib import Path
from create_mask import create_3d_blob, is_point_inside_cylinder, generate_random_planes, is_point_outside_of_plane, create_convex_hull
import random
from render_utils.renderer import NVDiffRasterizerContext
from utils.regular_sampler import RegularSampler
import imageio
import trimesh
import objaverse
import multiprocessing
import json
from PIL import Image

processes = multiprocessing.cpu_count()

def save_images(
        images: torch.Tensor,  # B,H,W,CH
        dir: Path,
):
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    for i in range(images.shape[0]):
        imageio.imwrite(dir / (f'{i:06d}.png'),
                        (images[i, :, :, :] * 255).clamp(max=255).type(torch.uint8).detach().cpu().numpy())



def projection_from_intrinsics(fx,fy,cx,cy,img_size, n=1, f=50, device='cuda'):
    projection = torch.zeros([fx.shape[0], 4,4], device=device, dtype=torch.float)
    projection[:,0,0] = fx / (0.5*img_size[0])
    projection[:,1,1] = fy / (0.5*img_size[1])
    projection[:,0,2] = 1 - cx/(0.5*img_size[0])
    projection[:,1,2] = cy/(0.5*img_size[1]) - 1
    projection[:,2, 2] = -(f + n) / (f - n)
    projection[:,2, 3] = -(2 * f * n) / (f - n)
    projection[:,3, 2] = -1
    return projection

def c2w_to_w2c(c2w):
    w2c = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0
    return w2c


def convert_c2w_between_opengl_opencv(c2w):
    right = c2w[:,:,0]
    up = c2w[:,:,1]
    forward = c2w[:,:,2]
    loc = c2w[:,:,3]
    return torch.stack([right,-up,-forward, loc],dim=-1)

def get_midpoints(v,f):
    fv = v[f]
    midpoint = fv.mean(dim=1)
    return midpoint

def c2w_and_intrinsics_to_mvp():
    pass

def get_bbox(v):
    bbox_max= v.max(dim=0)[0]
    bbox_min =v.min(dim=0)[0]
    return bbox_min, bbox_max

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (np.inf,) * 3
    bbox_max = (-np.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return np.array(bbox_min), np.array(bbox_max)

def normalize_scene(v):
    bbox_min, bbox_max= scene_bbox()
    scale = 1.8/ (bbox_max - bbox_min).max()
    offset = -(bbox_max + bbox_min) / 2 
    v += offset
    v *= scale
    return v

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1.8 / np.max(bbox_max - bbox_min)
    offset = -(bbox_min + bbox_max) / 2
    offset = Vector((offset[0], offset[1], offset[2]))
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
        obj.matrix_world.translation *= scale
        obj.scale *= scale
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    
def load_object(object_path):
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def convert_one_glb(path_to_glb, path_to_obj):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    glb_prefix = path_to_glb.split(os.sep)[-1][:-4]
    target_obj = path_to_obj + os.sep + glb_prefix + ".obj"
    bpy.ops.import_scene.gltf(filepath=path_to_glb)
    normalize_scene()
    bpy.ops.export_scene.obj(filepath=target_obj)
    for collection in bpy.data.collections:
       for obj in collection.all_objects:
           obj.select_set(True)
    bpy.ops.object.delete()

def get_objaverse_id_from_s3_path(path):
    path = path.strip('/').split('/')
    return path[-2]

if __name__ == "__main__":

    make_video = True

    data_root = 'instant3d/objaverse_data'
    s3_dirs_filename = 'level1_10k_s3_dirs.txt'
    captions_filename = 'Cap3D_automated_Objaverse.csv'

    print('Loading image dirs...')
    with open(os.path.join(data_root, s3_dirs_filename)) as f:
        paths = [x.strip() for x in f]
    print('Loaded image dirs.')

    caption_path = os.path.join(data_root, captions_filename)
    print(f'Loading captions from {caption_path}...')
    caption_dict = {}
    for line in open(caption_path):
        line = line.strip().split(',')
        caption_dict[line[0]] = ','.join(line[1:])
    print('Loaded captions.')

    print('Filtering valid paths...')
    pruned_paths = [x for x in paths if get_objaverse_id_from_s3_path(x) in caption_dict]
    print('Filtering done.')
    rendered_mask_root = Path('/sensei-fs-3/users/abarda/data/instant3d/masks/')

    for j in range(len(pruned_paths)):
        model_name = os.path.join(*pruned_paths[j].split('/')[3:5])
        surface_exists = Path(rendered_mask_root,"surface", model_name).exists()
        volume_exists = Path(rendered_mask_root,"volume", model_name).exists()
        convexhull_exists = Path(rendered_mask_root,"convexhull", model_name).exists()
        if surface_exists and volume_exists and convexhull_exists:
            print('{} exists - skipping.'.format(model_name))
            continue

        local_glb_path = 'tmp_glb/res.glb'
        local_obj_path = 'tmp_glb'
        
        if not Path(local_obj_path).exists():
            Path(local_obj_path).mkdir(parents=True, exist_ok=True)

        objects = objaverse.load_objects(
        uids=[model_name.split('/')[1]],
        download_processes=processes
        )
        local_glb_path = objects[model_name.split('/')[1]]
        ###### open in blender and convert to obj # #######
        convert_one_glb(local_glb_path, local_obj_path)
        ###### normalize scene ########
        mesh = trimesh.load(os.path.join(local_obj_path,'res.obj'), force='mesh')
        v, f = mesh.vertices, mesh.faces
        v  = torch.tensor(v, dtype=torch.float, device = 'cuda')
        f = torch.tensor(f, dtype=torch.int32, device = 'cuda')
        reg_sampler = RegularSampler(device='cuda', delta=0.8/100, max_sampling_n=15)
        rotation_matrix = torch.tensor([[1,0,0],
                                        [0,0, -1],
                                        [0,1,0]], device = 'cuda', dtype=torch.float).unsqueeze((0))
        v = torch.bmm(rotation_matrix.expand([v.shape[0],-1,-1]), v.unsqueeze(-1))
        v = v.squeeze(-1)
        if v.shape[0]<=300000:
            reg_sampler.sample_regular(v.unsqueeze(0),f.unsqueeze(0), offset=1e-5)
            active_v, active_f = reg_sampler.sampled_vertices, reg_sampler.sampled_faces
        else: 
            active_v, active_f = v, f
        midpoint = get_midpoints(active_v,active_f)
        ###### download camera information #######

        if make_video:
            rgb_img_list = []
            for k in range(16):
                rgb_path = os.path.join('training_data/renders', model_name, 'renderings')
                s3_img_uri = os.path.join(rgb_path,f'{k:08d}_rgb.png')
                rgb_img = Image.open(s3_img_uri)#load_s3_image(s3_img_uri, s3_client)
                rgb_img = np.array(rgb_img)
                rgb_img_list.append(rgb_img)

        camera_JSON_path= 'cameras/opencv_cameras.json'
        with open(camera_JSON_path, 'r') as f:
            camera_data = json.load(f)

        num_cameras = 16
        w,h = 512,512
        ####### create mvp matrix #######
        w2cs = [camera_data['frames'][i]['w2c'] for i in range(num_cameras)]
        w2c_opencv= torch.tensor(w2cs, device='cuda')
        fxs = [camera_data['frames'][i]['fx'] for i in range(num_cameras)]
        fx = torch.tensor(fxs, device='cuda')
        fys = [camera_data['frames'][i]['fy'] for i in range(num_cameras)]
        fy = torch.tensor(fys, device='cuda')
        cxs = [camera_data['frames'][i]['cx'] for i in range(num_cameras)]
        cx = torch.tensor(cxs, device='cuda')
        cys = [camera_data['frames'][i]['cy'] for i in range(num_cameras)]
        cy = torch.tensor(cys, device='cuda')
        intrinsic_vector = torch.tensor([fxs,fys,cxs,cys], device='cuda')
        c2w_opencv = c2w_to_w2c(w2c_opencv)
        c2w_opengl = convert_c2w_between_opengl_opencv(c2w_opencv)
        mv = c2w_to_w2c(c2w_opengl).to('cuda')

        proj = projection_from_intrinsics(fx, fy, cx, cy, [w,h], n=0.1, f=1000.0)
        mvp = proj @ mv

        for i in range(10):
            ###### create 3d consistant masks ######
            if not surface_exists:
                bbox_min, bbox_max = get_bbox(v)
                sx,sy,sz = (0.01*(bbox_max - bbox_min)).split(1)
                a, b, c = (0.3*(bbox_max - bbox_min)).split(1)
                N = random.randrange(3,6)
                initial_point = active_v[[random.randrange(0,active_v.shape[0])]]
                v_blob, f_blob, cylinders_info = create_3d_blob(N,a,b,c,sx,sy,sz, initial_point=initial_point)
                v_all = torch.cat([active_v, v_blob])
                f_all = torch.cat([active_f,f_blob + active_v.shape[0]])
                ###### marks faces to mask #######
                is_inside = is_point_inside_cylinder(midpoint, cylinders_info)
                renderer = NVDiffRasterizerContext('cuda', device = 'cuda')
                render_mask = renderer.render(active_v,active_f, is_inside, mvp, w,h)
                render_mask = render_mask.permute([0,3,2,1]).flip(dims=[1])
                save_images(render_mask, os.path.join(rendered_mask_root, model_name, 'masks_{}').format(i))

                if make_video:
                    img_list = []
                    for k in range(16):
                        mask = render_mask[k].bool().cpu().numpy()
                        masked_rgb_img = rgb_img_list[k]*(~mask) + mask*128
                        img_list.append(masked_rgb_img.astype(np.uint8))
                    imageio.mimsave(os.path.join(rendered_mask_root, 'surface', model_name,'masks_{}'.format(i),"masked_model.mp4"), img_list, fps=15)

            if not volume_exists:
                bbox_min, bbox_max = get_bbox(v)
                sx,sy,sz = (0.5*(bbox_max - bbox_min)).split(1)
                initial_point = active_v[[random.randrange(0,active_v.shape[0])]]*0
                N_planes = 1
                locs, normals = generate_random_planes(initial_point, N=N_planes, sx=sx, sy=sy,sz=sz)
                ###### marks faces to mask #######
                midpoint_is_nan = [~torch.isnan(midpoint).any(dim=-1)] # some vertices are NaN in dataset
                is_inside = is_point_outside_of_plane(normals, locs, midpoint)
                ###### render using nvdiffrast ######
                #is_inside[:] = True
                import igl
                from create_mask import rotation_angle_between_two_vectors_and_axis, rotate_vectors_by_angles_around_axis
                def create_disk_mesh(locs, normals,r=2.0, n_polar = 10):
                    v_all, f_all = [],[]
                    phi = (torch.arange(n_polar, device='cuda') * 2*torch.pi/(n_polar)).unsqueeze(-1)
                    # create circle with n_polar vertices
                    x = phi.cos()
                    y = phi.sin()
                    z = torch.zeros_like(x)
                    v  = torch.stack([x,y,z], dim=-1).reshape([-1,3])
                    ids = torch.arange(n_polar)
                    cap_ids = ids[2:]
                    f = torch.stack([cap_ids*0, cap_ids-1, cap_ids],dim=1)
                    up = torch.tensor([0,0,1], dtype=torch.float, device= 'cuda').unsqueeze(0)
                    rotation_axis = up.cross(normals, dim=-1)
                    rotation_axis /= torch.norm(rotation_axis, dim=-1, keepdim=True)
                    angles = rotation_angle_between_two_vectors_and_axis(up, normals, rotation_axis)
                    for i in range(N_planes):
                        v_rotated = rotate_vectors_by_angles_around_axis(v, angles[[i]], rotation_axis[[i]])
                        v_rotated += locs[[i]]
                        v_all.append(v_rotated)
                        f_all.append(f+i*v.shape[0])
                    v_all = torch.cat(v_all).to('cuda')
                    f_all = torch.cat(f_all).to('cuda')
                    return v_all, f_all

                v_planes, f_planes = create_disk_mesh(locs,normals)
                
                renderer = NVDiffRasterizerContext('cuda', device = 'cuda')
                render_mask = renderer.render(active_v,active_f, is_inside, mvp, w,h)
                render_mask = render_mask.permute([0,3,2,1]).flip(dims=[1])
                save_images(render_mask, os.path.join(rendered_mask_root, 'volume', model_name, 'masks_{}').format(i))

                if make_video:
                    img_list = []
                    for k in range(16):
                        mask = render_mask[k].bool().cpu().numpy()
                        masked_rgb_img = rgb_img_list[k]*(~mask) + mask*128
                        img_list.append(masked_rgb_img.astype(np.uint8))
                    imageio.mimsave(os.path.join(rendered_mask_root, 'volume', model_name,'masks_{}'.format(i),"masked_model.mp4"), img_list, fps=15)   

            if not convexhull_exists:
                bbox_min, bbox_max = get_bbox(v)
                sx,sy,sz = (0.5*(bbox_max - bbox_min)).split(1)
                initial_point = active_v[[random.randrange(0,active_v.shape[0])]]*0
                N_planes = 1
                is_inside = torch.zeros(midpoint.shape[0], dtype = bool, device='cuda')
                while not is_inside.any():
                    locs, normals = generate_random_planes(initial_point, N=N_planes, sx=sx, sy=sy,sz=sz)
                    is_inside = is_point_outside_of_plane(normals, locs, midpoint)
                v_ch, f_ch = create_convex_hull(midpoint[is_inside], locs, normals, N=1000)
                # scale ch by 1.2
                offset = v_ch.mean(dim=0)
                v_ch -= offset
                v_ch *= 1.2
                v_ch += offset

                v_concat = torch.cat([active_v, v_ch], dim=0)
                f_concat = torch.cat([active_f, f_ch + active_v.shape[0]], dim=0)
                is_inside = torch.zeros(f_concat.shape[0], dtype=bool, device='cuda')
                is_inside[active_f.shape[0]:] = True
                renderer = NVDiffRasterizerContext('cuda', device = 'cuda')
                render_mask = renderer.render(v_concat.float(),f_concat, is_inside, mvp, w,h)
                render_mask = render_mask.permute([0,3,2,1]).flip(dims=[1])
                save_images(render_mask, os.path.join(rendered_mask_root, 'convexhull', model_name, 'masks_{}').format(i))

                if make_video:
                    img_list = []
                    for k in range(16):
                        mask = render_mask[k].bool().cpu().numpy()
                        masked_rgb_img = rgb_img_list[k]*(~mask) + mask*128
                        img_list.append(masked_rgb_img.astype(np.uint8))
                    imageio.mimsave(os.path.join(rendered_mask_root, 'convexhull', model_name,'masks_{}'.format(i),"masked_model.mp4"), img_list, fps=15)   

    print('done')


