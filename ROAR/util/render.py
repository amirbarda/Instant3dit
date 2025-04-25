import nvdiffrast.torch as dr
import torch
# from pytorch3d import renderer as py3d_renderer
# from pytorch3d.structures import Meshes, Pointclouds
# from pytorch3d.ops import interpolate_face_attributes
from ROAR.util.geometry_utils import calculate_vertex_normals
import numpy as np
import igl


def flip_normals(
        faces: torch.Tensor,  # F,3
):
    """flip the orientation of all faces"""
    new_faces = faces.flip(dims=[1])
    return new_faces


def three_color_triangular_mesh(f):
    """
    colors a triangular mesh with 3 colors
    assumes mesh is not a pyramid and is 2-manifold
    """
    from scipy.sparse import csr_matrix
    f = f.cpu().numpy()
    colors = np.ones(f.shape[0], dtype=np.int32) - 2
    possible_colors = np.array([0, 1, 2], dtype=np.int32)
    ff, _ = igl.triangle_triangle_adjacency(f)
    neighbours = ff[0]
    if neighbours[1] not in ff[neighbours[0]]:
        colors[neighbours[1]] = 0
        colors[neighbours[0]] = 0
        bfs_start = neighbours[2]
    elif neighbours[1] not in ff[neighbours[2]]:
        colors[neighbours[1]] = 0
        colors[neighbours[2]] = 0
        bfs_start = neighbours[0]
    else:
        raise NotImplementedError("graph has a pyramid and thus is not 3-colorable")
    row = np.arange(f.shape[0]).repeat(3)
    col = ff.reshape(-1)
    data = np.ones_like(col)
    graph = csr_matrix((data, (row, col)), shape=(f.shape[0], f.shape[0]))
    d, p = igl.bfs(graph, bfs_start)
    for node in d:
        neighbours = ff[node]
        used_colors = colors[neighbours]
        diff = np.setdiff1d(possible_colors, used_colors)
        if len(diff) == 0:
            raise NotImplementedError("can't 3-color this mesh")
        selected_color = diff[0]
        colors[node] = selected_color
    RGB_colors = np.zeros((colors.size, colors.max() + 1))
    RGB_colors[np.arange(colors.size), colors] = 1
    return RGB_colors


class NormalsRenderer:
    _glctx: dr.RasterizeCudaContext = None

    def __init__(
            self,
            image_size,  # tuple (W, H)
            device='cuda:0'
    ):
        self._image_size = image_size
        self._glctx = dr.RasterizeCudaContext(device=device)
        #self._glctx_cull_bf = dr.RasterizeCudaContext(cull_bf=True, device=device)

    def render(self,
               vertices: torch.Tensor,  # V,3 float
               v_normals: torch.Tensor,  # V,3 float
               faces: torch.Tensor,  # F,3 long
               v_colors = False,
               mv: torch.Tensor = None,
               projection: torch.Tensor = None,
               mvp: torch.Tensor = None,
               flat_shade=False,
               f_normals=None,  # F,3 float
               get_visible_faces=False,
               cull_bf=False,
               no_aa=False,
               # should result not be anti-aliased? set to true ONLY for visualizations or gradients will be wrong
               soft=False,  # unused for now
               silhouette=False,  # unused for now
               is_pointcloud=False,  # unused for now
               random_color_faces=False,
               three_color_faces=False,
               xyz_color=False,
               super_sample=False,
               uv = None,
               uv_idx = None,
               tex = None
               ) -> torch.Tensor:  # C,H,W,4
        if is_pointcloud:
            raise NotImplementedError  # nvdiffrast doesnt support pc rendering
        V = vertices.shape[0]
        if v_normals is None:
            v_normals = calculate_vertex_normals(vertices, faces)
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V, 1, device=vertices.device)), axis=-1)  # V,3 -> V,4
        if (mv is None or projection is None) and mvp is None:
            vertices_clip = vert_hom @ self._mvp.transpose(-2, -1)  # C,V,4
        elif mvp is None:
            mvp = projection @ mv
            vertices_clip = vert_hom @ mvp.transpose(-2, -1)
        else:
            vertices_clip = vert_hom @ mvp.transpose(-2, -1)
        if cull_bf:
            assert xyz_color is False, "xyz_colors not supported with cull_bf"
            glctx = self._glctx_cull_bf
        else:
            glctx = self._glctx
        if super_sample:
            cur_res = [x * 4 for x in self._image_size]
        else:
            cur_res = self._image_size
        rast_out, _ = dr.rasterize(glctx, vertices_clip, faces, resolution=cur_res,
                                   grad_db=False)  # C,H,W,4

        if uv is not None:
            rast_out, _ = dr.rasterize(glctx, vertices_clip, faces, resolution=cur_res,
                                       grad_db=False)  # C,H,W,4
            texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
            color = dr.texture(tex[None, ...], texc, filter_mode='linear')
            color = color * torch.clamp(rast_out[..., -1:], 0, 1)
            return color, None

        if get_visible_faces:
            visible_faces, counts = torch.unique((rast_out[..., 3].long() - 1), return_counts=True)
            # visible_faces = visible_faces[counts > 10]
            mask = visible_faces != -1
            visible_faces = visible_faces[mask]
            extra = {"vis_faces": visible_faces, "counts": counts[mask]}
        else:
            extra = {}
        if random_color_faces:
            visible_faces = torch.unique(rast_out[..., -1:].long() - 1)
            # mask = visible_faces != -1
            # visible_faces = visible_faces[mask]
            colors = torch.rand((faces.shape[0], 3), device=vertices.device)
            col = colors[rast_out[..., -1].long() - 1]
            col = torch.where(rast_out[..., -1:] != 0, col, torch.zeros_like(col))
            result = dr.antialias(col, rast_out, vertices_clip, faces)
        elif three_color_faces:
            visible_faces = torch.unique(rast_out[..., -1:].long() - 1)
            colors = three_color_triangular_mesh(faces)
            colors = torch.tensor(colors, dtype=torch.float32, device=vertices.device)
            col = colors[rast_out[..., -1].long() - 1]
            col = torch.where(rast_out[..., -1:] != 0, col, torch.zeros_like(col))
            result = dr.antialias(col, rast_out, vertices_clip, faces)
        else:
            if flat_shade:
                if xyz_color:
                    raise NotImplementedError("not supported (though easy to implement)")
                else:
                    if f_normals is None:
                        raise NotImplementedError("how can I flat shade with no face normals?")
                    tri_facecolor = torch.arange(0, faces.shape[0], dtype=torch.int32, device=vertices.device)[:,
                                    None].expand(-1, 3).contiguous()
                    col, _ = dr.interpolate(f_normals, rast_out, tri_facecolor)
                    #col = (col + 1) / 2
                    # col = (f_normals[rast_out[..., 3].long() - 1] + 1) / 2 #C,H,W,3
                    alpha = torch.clamp(rast_out[..., -1:], max=1)  # C,H,W,1
                    col = torch.concat((col, alpha), dim=-1)  # C,H,W,4
            else:
                if xyz_color:
                    vert_col = vertices / (vertices.norm(dim=-1, keepdim=True) + 1e-6)
                    vert_col = (vert_col + 1) / 2
                elif not v_colors:
                    vert_col = (v_normals + 1) / 2  # V,3
                else:
                    vert_col = v_normals
                col, _ = dr.interpolate(vert_col, rast_out, faces)  # C,H,W,3
                alpha = torch.clamp(rast_out[..., -1:], max=1)  # C,H,W,1
                col = torch.concat((col, alpha), dim=-1)  # C,H,W,4
            col = torch.where(rast_out[..., -1:] != 0, col, torch.zeros_like(col))
            if not no_aa:
                result = dr.antialias(col, rast_out, vertices_clip, faces)  # C,H,W,4
            else:
                result = col  # C,H,W,4
            if silhouette:
                result = result[..., 3:]

        return result, extra


# class PulsarNormalsRenderer:
#     def __init__(
#             self,
#             mv: torch.Tensor,  # C,4,4 (a batch of c2w)
#             image_size,  # (int, int)
#             device
#     ):
#         self.num_of_views = mv.shape[0]
#         self.mv = mv  # pulsar expects wc2
#         self.image_size = image_size
#         self.device = device
#         self.focal_length = torch.ones((self.num_of_views, 2), device=device) * 5
#         self.principal_point = torch.zeros((self.num_of_views, 2), device=device)
#         # create a (super hideous) camera model that pulsar can use
#         # self.cameras = py3d_renderer.OpenGLPerspectiveCameras(device=self.device,
#         #                                                 R=self.mv[:, :3, :3],
#         #                                                 T=self.mv[:, :3, 3],
#         #                                                 fov=45,
#         #                                                 znear=0.1)
#         self.cameras = py3d_renderer.PerspectiveCameras(device=self.device,
#                                                         R=self.mv[:, :3, :3],
#                                                         T=self.mv[:, :3, 3],
#                                                         focal_length=self.focal_length,
#                                                         principal_point=self.principal_point)
#         self.sigma = 1e-4
#         self.point_radius = 0.001
#         self.point_per_pixel = 10
#         self.blend_params = py3d_renderer.BlendParams(sigma=self.sigma,
#                                                       gamma=1e-4,
#                                                       background_color=(0.0, 0.0, 0.0))
#         self.sil_shader = py3d_renderer.SoftSilhouetteShader(self.blend_params)
#         self.soft_raster_settings = py3d_renderer.RasterizationSettings(
#             image_size=self.image_size[0],
#             blur_radius=np.log(1. / 1e-4 - 1.) * self.sigma,
#             faces_per_pixel=50,
#             perspective_correct=False
#         )
#         self.raster_settings = py3d_renderer.RasterizationSettings(
#             image_size=self.image_size[0],
#             blur_radius=0,
#             faces_per_pixel=1
#         )
#         self.soft_rasterizer = py3d_renderer.MeshRasterizer(
#             cameras=self.cameras,
#             raster_settings=self.soft_raster_settings
#         )
#         self.rasterizer = py3d_renderer.MeshRasterizer(
#             cameras=self.cameras,
#             raster_settings=self.raster_settings
#         )
#         self.pc_raster_settings = py3d_renderer.PointsRasterizationSettings(
#             image_size=self.image_size[0],
#             radius=self.point_radius,
#             points_per_pixel=8
#         )
#         self.pc_rasterizer = py3d_renderer.PointsRasterizer(cameras=self.cameras,
#                                                             raster_settings=self.pc_raster_settings)
#         # self.pc_renderer = py3d_renderer.PointsRenderer(rasterizer=self.pc_rasterizer,
#         #                                                 compositor=py3d_renderer.AlphaCompositor(background_color=(0, 0, 0))
#         # )
#         self.pc_renderer = py3d_renderer.points.PulsarPointsRenderer(rasterizer=self.pc_rasterizer).to(self.device)

#     def normal_shading(self, fragments, meshes, v_normals) -> torch.Tensor:  # C,H,W,3
#         """shades interpolated normals"""
#         faces = meshes.faces_packed()  # (F, 3)
#         # vertex_normals = meshes.verts_normals_packed()  # (V, 3)
#         faces_normals = (v_normals[faces] + 1) / 2
#         # ones = torch.ones_like(fragments.bary_coords)
#         colors = interpolate_face_attributes(
#             fragments.pix_to_face, fragments.bary_coords, faces_normals
#         )
#         if colors.shape[3] > 1:
#             # pixel_normals = pixel_normals.mean(dim=3)
#             final_images = py3d_renderer.blending.softmax_rgb_blend(colors,
#                                                                     fragments,
#                                                                     self.blend_params,
#                                                                     znear=1.0,
#                                                                     zfar=100.0)
#         else:
#             final_images = colors
#             final_images = final_images.squeeze()
#             is_foreground = fragments.pix_to_face[..., 0] >= 0
#             final_images = torch.cat((final_images, is_foreground[..., None]), dim=-1)
#         return final_images

#     def render(self,
#                vertices: torch.Tensor,  # V,3 float
#                v_normals: torch.Tensor,  # V,3 float
#                faces: torch.Tensor,  # F,3 long
#                mv: torch.Tensor = None,  # C,4,4
#                projection: torch.Tensor = None,  # C,4,4
#                soft=False,  # set to true for differentiable rendering
#                silhouette=False,  # set to true silhouette rendering (works only with soft=True)
#                is_pointcloud=False
#                ) -> torch.Tensor:  # C,H,W,4
#         if mv is None:
#             cameras = self.cameras
#             pc_renderer = self.pc_renderer
#         else:
#             # cameras = py3d_renderer.OpenGLPerspectiveCameras(device=self.device,
#             #                                             R=mv[:, :3, :3],
#             #                                             T=mv[:, :3, 3],
#             #                                             fov=45,
#             #                                             znear=0.1)
#             cameras = py3d_renderer.PerspectiveCameras(device=self.device,
#                                                        R=mv[:, :3, :3],
#                                                        T=mv[:, :3, 3],
#                                                        focal_length=self.focal_length,
#                                                        principal_point=self.principal_point)
#             rasterizer = py3d_renderer.PointsRasterizer(cameras=cameras,
#                                                         raster_settings=self.pc_raster_settings)
#             pc_renderer = py3d_renderer.points.PulsarPointsRenderer(rasterizer=rasterizer).to(self.device)
#         if is_pointcloud:
#             return self.render_pointcloud(vertices, v_normals, pc_renderer)
#         else:
#             mesh = Meshes(verts=[vertices], faces=[faces])
#             meshes = mesh.extend(self.num_of_views)
#             if soft:
#                 fragments = self.soft_rasterizer(meshes, cameras=cameras)
#             else:
#                 fragments = self.rasterizer(meshes, cameras=cameras)
#             if silhouette:
#                 assert soft == True, "For hard silhouette rendering, pass silhouette=False and take the last channel"
#                 return self.sil_shader(fragments, meshes)
#             else:
#                 v_normals = v_normals.repeat(self.num_of_views, 1)
#                 images = self.normal_shading(fragments, meshes, v_normals)
#                 return images.squeeze()

#     def render_pointcloud(self,
#                           points: torch.Tensor,  # V,3 float
#                           p_normals: torch.Tensor,  # V,3 float
#                           pc_renderer,
#                           p_colors: torch.Tensor = None,  # V,3 float
#                           ) -> torch.Tensor:  # C,H,W,4
#         pc = Pointclouds(points=[points], features=[(p_normals + 1) / 2])
#         pcs = pc.extend(self.num_of_views)
#         images = pc_renderer(
#             pcs,
#             gamma=torch.tensor([1e-5])[None, :].repeat(self.num_of_views, 1).to(self.device),
#             znear=torch.tensor([1.0])[None, :].repeat(self.num_of_views, 1).to(self.device),
#             zfar=torch.tensor([10.0])[None, :].repeat(self.num_of_views, 1).to(self.device)
#         )
#         return images.squeeze()