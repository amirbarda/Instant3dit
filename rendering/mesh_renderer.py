from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from rendering.nvdiffrast_rasterizer import NVDiffRasterizerContext
from utils.typing_utils import *
from torchvision.transforms import GaussianBlur
from ROAR.util.func import save_images, load_images

def flip(f):
    tmp = f[:,0]
    f[:,0] = f[:,1]
    f[:,1] = tmp
    return f.contiguous()


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")


class NVDiffRasterizer(nn.Module):
    @dataclass
    class Config():
        context_type: str = "gl"

    cfg: Config

    def configure(
        self,
        geometry,
        material,
        background,
        device
    ) -> None:
        self.geometry = geometry
        self.material = material
        self.background = background
        self.context_type = 'cuda'
        self.ctx = NVDiffRasterizerContext(self.context_type, device) #self.cfg.context_type

    # expects camera paramters in opengl format
    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        light_positions: Float[Tensor, "B 3"] = None,
        render_normal: bool = True,
        render_rgb: bool = True,
        render_depth: bool = True,
        color_net = None,
        bg_net = None,
        material = None,
        use_supersampled = True,
        shading = None,
        output_mask = False,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]

        #todo: make change isosurface to get_mesh() for Magic3d
        #mesh = self.geometry.get_mesh()
        # if use_supersampled:
        #     mesh = self.geometry.get_supersampled_mesh(orient_normals_strategy = 'smooth')
        if hasattr(self.geometry, 'get_mesh'):
            mesh = self.geometry.get_mesh()
        elif hasattr(self.geometry, 'isosurface'):
            mesh = self.geometry.isosurface()
        else:
            mesh = self.geometry
            use_supersampled = False
            if 'allowed_vertices' in mesh.extras:
                mesh.allowed_vertices = mesh.extras['allowed_vertices']
            else:
                mesh.allowed_vertices = None

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

        out = {"opacity": mask_aa, "mesh": mesh}

        # returned sampled vertices
        hit = torch.norm(rast,dim=-1)>0
        fv = mesh.v_pos[mesh.t_pos_idx[rast[:,:,:,-1].long()-1]]
        fv0 = fv[:,:,:,0]
        fv1 = fv[:,:,:,1]
        fv2 = fv[:,:,:,2]
        u = rast[:,:,:,[0]]
        v = rast[:,:,:,[1]]
        uv = torch.cat([u,v], dim=-1)
        x = (fv0)*u + (fv1)*v + fv2*(1-u-v)
        # get face id for each pixel
        face_ids = rast[:,:,:,-1].long() - 1
        out.update({"x": x, "hit": hit,  "uv": uv, "face_ids": face_ids})
        

        # import igl
        # import numpy as np
        # igl.write_obj('/home/dcor/amirbarda/MeshSDF/tmp_tests/mesh.obj', mesh.v_pos.detach().cpu().numpy(), mesh.t_pos_idx.cpu().numpy())
        # np.savetxt('/home/dcor/amirbarda/MeshSDF/tmp_tests/x.txt', x[hit].detach().cpu().numpy())
        # np.savetxt('/home/dcor/amirbarda/MeshSDF/tmp_tests/face_ids.txt', face_ids[hit].cpu().numpy())

            # todo: sample color from appearance network if faces are allowed
        if render_normal:
            pertrubations = 0
            if color_net is not None and self.geometry.cfg.optimize_normal_map:
                pertrubations = torch.randn_like(mesh.v_nrm, device=mesh.v_nrm.device) #(torch.sigmoid(features[:,3:])-0.5)*0.2
            
            gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm + pertrubations  , rast, mesh.t_pos_idx)
            gb_normal = F.normalize(gb_normal, dim=-1)
            gb_normal_aa = torch.lerp(
                torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
            )
            gb_normal_aa = self.ctx.antialias(
                gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
            )
            out.update({"comp_normal": gb_normal_aa})  # in [0, 1]
            # save_images(gb_normal_aa, '/home/dcor/amirbarda/MeshSDF/tmp_tests/')

        if render_depth:
            # each pixel is colored according to its distance on the forward camera axis (camera dir). according to depth_to_normal in MVEdit's nerf_utils, depth is inverse 1/z.
            v_forward = torch.tensor([0,0,1], device = 'cuda', dtype=torch.float).reshape([1,-1,1]).expand([c2w.shape[0],-1,-1])
            cams_forward_dir = torch.bmm(-c2w[:,:3,:3],v_forward).transpose(dim0=-1, dim1=-2).unsqueeze(-2)
            inv_z = torch.sum(((x-camera_positions.reshape([c2w.shape[0],1,1,-1]))*cams_forward_dir),dim=-1, keepdim=True)**(-1)
            inv_z[~hit] = 0
            out.update({"depth": inv_z.clip(0,1)})  # in [0, 1]

        if render_rgb:

            if use_supersampled:
                mesh = self.geometry.get_supersampled_mesh(orient_normals_strategy = 'smooth', offset=1e-3)
                v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
                    mesh.v_pos, mvp_mtx
                )
                rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
                mask = rast[..., 3:] > 0
                mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)
                fv = mesh.v_pos[mesh.t_pos_idx[rast[:,:,:,-1].long()-1]]
                fv0 = fv[:,:,:,0]
                fv1 = fv[:,:,:,1]
                fv2 = fv[:,:,:,2]
                u = rast[:,:,:,[0]]
                v = rast[:,:,:,[1]]
                x = (fv0)*u + (fv1)*v + fv2*(1-u-v)
                # get face id for each pixel
                face_ids = rast[:,:,:,-1].long() - 1

            # v_original_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            #     self.geometry.original_mesh.v_pos, mvp_mtx)

            selector = mask[..., 0]

            features = None
            geo_out = None
            if hasattr(self.geometry, 'forward_color') and self.geometry.forward_color is not None:
                features = self.geometry.forward_color(mesh.v_pos).detach() #color_net(mesh.v_pos).detach() 
  
                if color_net is not None and features is not None:
                    #features = self.geometry.forward_color(mesh.v_pos) 
                    features[mesh.allowed_vertices] = color_net(mesh.v_pos[mesh.allowed_vertices]) 
                elif color_net is not None:
                    features = color_net(mesh.v_pos) 

                #geo_out = self.geometry(positions, output_normal=False) #per pixel features
                #geo_out, _ = self.ctx.interpolate_one(mesh.color, rast, mesh.t_pos_idx)
                #geo_out = {'features' : geo_out[selector]}
            elif hasattr(self.geometry, 'has_texture') and self.geometry.has_texture is True :
                # TODO: needs to be differentiable through Nvdiffrast?
                bc = torch.cat([u,v, 1-u-v], dim=-1)
                geo_out = self.geometry.get_color(face_ids.flatten(), bc.reshape(-1,3)).detach().reshape_as(bc)#color_net(mesh.v_pos).detach() 


            if features is not None:
                colors = features[:,:3]            
            else:
                colors = torch.ones_like(mesh.v_pos, device=mesh.v_pos.device, dtype=torch.float)*0.5#mesh.color/255

            if geo_out is None:
                geo_out, _ = self.ctx.interpolate_one(colors.contiguous(), rast , mesh.t_pos_idx)

            gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos.contiguous(), rast, mesh.t_pos_idx)
            gb_viewdirs = F.normalize(
                gb_pos - camera_positions[:, None, None, :], dim=-1
            )
            gb_light_positions = light_positions[:, None, None, :].expand(
                -1, height, width, -1
            )
            positions = gb_pos[selector]
            #save_images(geo_out, '/home/dcor/amirbarda/MeshSDF/tmp_tests/')
            #import numpy as np
            #np.savetxt('/home/dcor/amirbarda/MeshSDF/tmp_tests/allowed_supersampled.txt', mesh.allowed_vertices.cpu().numpy())
            geo_out = {'features' : geo_out[selector]}
            if material is None:
                material = self.material

            if material is not None:
                rgb_fg1 = material(
                    viewdirs=gb_viewdirs[selector],
                    positions=positions,
                    light_positions=gb_light_positions[selector],
                    shading_normal=gb_normal[selector],
                    shading = shading,
                    **geo_out
                )
            else:
                rgb_fg1 = geo_out['features']
                
            if self.material is not None:
                rgb_fg2 = self.material(
                    viewdirs=gb_viewdirs[selector],
                    positions=positions,
                    light_positions=gb_light_positions[selector],
                    shading_normal=gb_normal[selector],
                    shading = shading,
                    **geo_out
                )
            else:
                rgb_fg2 = geo_out['features']

            if hasattr(self.geometry, 'forward_color') and self.geometry.forward_color is not None:

                # rast_origial, _ = self.ctx.rasterize(v_original_pos_clip, self.geometry.original_mesh.t_pos_idx, (height, width))
                # mask_original = rast_origial[..., 3:] > 0
                # mask_aa_original = self.ctx.antialias(mask_original.float(), rast_origial, v_original_pos_clip, self.geometry.original_mesh.t_pos_idx)
                # fv = self.geometry.original_mesh.v_pos[self.geometry.original_mesh.t_pos_idx[rast_origial[:,:,:,-1].long()-1]]
                # fv0 = fv[:,:,:,0]
                # fv1 = fv[:,:,:,1]
                # fv2 = fv[:,:,:,2]
                # u = rast_origial[:,:,:,[0]]
                # v = rast_origial[:,:,:,[1]]
                # x = (fv0)*u + (fv1)*v + fv2*(1-u-v)
                # # get face id for each pixel
                # face_ids = rast_origial[:,:,:,-1].long() - 1

                # todo: output mask should be rendered according to original mesh
                allowed_vertices_mask = torch.zeros(mesh.v_pos.shape[0], dtype = torch.bool).to(rgb_fg1)
                allowed_vertices_mask[mesh.allowed_vertices] = True
                allowed_faces_mask = torch.any(allowed_vertices_mask[mesh.t_pos_idx],dim=-1)
                render_mask =  ((face_ids >= 0) & (allowed_faces_mask[face_ids])).int().unsqueeze(-1).repeat([1,1,1,3])
                render_mask = render_mask.float()
                render_mask = render_mask.transpose(1,-1)
                gaussianBlur = GaussianBlur(kernel_size=3)
                for _ in range(3):
                    render_mask = gaussianBlur.forward(render_mask)
                render_mask = render_mask.transpose(1,-1)
                out.update({"mask": (render_mask>0)})
                #rgb_fg = rgb_fg1*mesh.allowed_vertices + rgb_fg2*(~render_mask)
                #np.savetxt('/home/dcor/amirbarda/MeshSDF/tmp_tests/img_points_material_colors.txt', colors.detach().cpu().numpy())
                # np.savetxt('/home/dcor/amirbarda/MeshSDF/tmp_tests/img_points.txt', x[hit].detach().cpu().numpy()) 
                # np.savetxt('/home/dcor/amirbarda/MeshSDF/tmp_tests/img_points_material_colors.txt', rgb_fg.detach().cpu().numpy())
                # np.savetxt('/home/dcor/amirbarda/MeshSDF/tmp_tests/img_points_shading_normals.txt', gb_normal[selector].detach().cpu().numpy())

                gb_rgb_fg1 = torch.zeros(batch_size, height, width, 3).to(rgb_fg1)
                gb_rgb_fg1[selector] = rgb_fg1
                gb_rgb_fg2 = torch.zeros(batch_size, height, width, 3).to(rgb_fg2)
                gb_rgb_fg2[selector] = rgb_fg2
                gb_rgb_fg = gb_rgb_fg1*(render_mask) + gb_rgb_fg2*(1-render_mask)
                #out.update({"mask": (render_mask>0)})
            else:
                gb_rgb_fg = torch.zeros(batch_size, height, width, 3).to(rgb_fg1)
                gb_rgb_fg[selector] = rgb_fg1
                out.update({"mask": mask_aa})
            #save_images(gb_rgb_fg, '/home/dcor/amirbarda/MeshSDF/tmp_tests/')
            if bg_net is not None:
                #gb_rgb_bg = bg_net(dirs=rays_d)
                gb_rgb_bg = bg_net(dirs=gb_viewdirs)
            else:
                gb_rgb_bg = self.background(dirs=gb_viewdirs)
            gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
            gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

            out.update({"comp_rgb": gb_rgb_aa})

        # if output_mask and mesh.allowed_vertices is not None:
        #     sculpted_vertices_mask = torch.zeros(mesh.v_pos.shape[0], device = 'cuda', dtype = torch.bool)
        #     sculpted_vertices_mask[mesh.allowed_vertices] = True
        #     sculpted_faces_mask = torch.any(sculpted_vertices_mask[mesh.t_pos_idx],dim=-1)
        #     render_mask =  ((out['face_ids'] >= 0) & (sculpted_faces_mask[out['face_ids']])).int().unsqueeze(-1).repeat([1,1,1,3])
        #     render_mask = render_mask.float()
        #     render_mask = render_mask.transpose(1,-1)
        #     gaussianBlur = GaussianBlur(kernel_size=3)
        #     for _ in range(2):
        #         render_mask = gaussianBlur.forward(render_mask)
        #     render_mask = render_mask.transpose(1,-1)
        #     out.update({"mask": (render_mask>0)})
        # else:
        #     out.update({"mask": mask_aa})
        return out

