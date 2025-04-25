import torch
import importlib
import os
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import trimesh
import numpy as np
from PIL import Image
import cv2
import sys

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def save_obj(pointnp_px3, facenp_fx3, colornp_px3, fpath):

    pointnp_px3 = pointnp_px3 @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    facenp_fx3 = facenp_fx3[:, [2, 1, 0]]

    mesh = trimesh.Trimesh(
        vertices=pointnp_px3, 
        faces=facenp_fx3, 
        vertex_colors=colornp_px3,
    )
    mesh.export(fpath, 'obj')


def save_obj_with_mtl(pointnp_px3, tcoords_px2, facenp_fx3, facetex_fx3, texmap_hxwx3, fname):
    fol, na = os.path.split(fname)
    na, _ = os.path.splitext(na)

    matname = '%s/%s.mtl' % (fol, na)
    fid = open(matname, 'w')
    fid.write('newmtl material_0\n')
    fid.write('Kd 1 1 1\n')
    fid.write('Ka 0 0 0\n')
    fid.write('Ks 0.4 0.4 0.4\n')
    fid.write('Ns 10\n')
    fid.write('illum 2\n')
    fid.write('map_Kd %s.png\n' % na)
    fid.close()
    ####

    fid = open(fname, 'w')
    fid.write('mtllib %s.mtl\n' % na)

    for pidx, p in enumerate(pointnp_px3):
        pp = p
        fid.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))

    for pidx, p in enumerate(tcoords_px2):
        pp = p
        fid.write('vt %f %f\n' % (pp[0], pp[1]))

    fid.write('usemtl material_0\n')
    for i, f in enumerate(facenp_fx3):
        f1 = f + 1
        f2 = facetex_fx3[i] + 1
        fid.write('f %d/%d %d/%d %d/%d\n' % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2]))
    fid.close()

    # save texture map
    lo, hi = 0, 1
    img = np.asarray(texmap_hxwx3, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = img.clip(0, 255)
    mask = np.sum(img.astype(np.float32), axis=-1, keepdims=True)
    mask = (mask <= 3.0).astype(np.float32)
    kernel = np.ones((3, 3), 'uint8')
    dilate_img = cv2.dilate(img, kernel, iterations=1)
    img = img * (1 - mask) + dilate_img * mask
    img = img.clip(0, 255).astype(np.uint8)
    Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(f'{fol}/{na}.png')

def reconstruct_using_mesh_lrm(output_path, images, input_cameras):
    mesh_path = str(output_path)
    name = 'lrm_mesh'
    export_texmap = True
    device = images.device
    config_path = 'LRMs/configs/instant-mesh-large.yaml'
    config = OmegaConf.load(config_path)
    config_name = os.path.basename(config_path).replace('.yaml', '')
    model_config = config.model_config
    infer_config = config.infer_config
    # load reconstruction model
    print('Loading mesh reconstruction model ...')
    model = instantiate_from_config(model_config)
    if os.path.exists(infer_config.model_path):
        model_ckpt_path = infer_config.model_path
    else:
        model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
    state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False
    if IS_FLEXICUBES:
        model.init_flexicubes_geometry(device, fovy=30.0)
    model = model.eval()

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(images, input_cameras)

        # get mesh
        mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=export_texmap,
            **infer_config,
        )
        if export_texmap:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_path_idx,
            )
        else:
            vertices, faces, vertex_colors = mesh_out
            save_obj(vertices, faces, vertex_colors, mesh_path_idx)
        print(f"Mesh saved to {mesh_path_idx}")
