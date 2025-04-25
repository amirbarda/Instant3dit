
import torch
import trimesh
import numpy as np
from ROAR.util.geometry_utils import calculate_vertex_normals
from ROAR.util.func import load_mesh, normalize_vertices
from pathlib import Path
from PIL import Image
from ROAR.util.func import read_mtlfile
import igl
import imageio as iio

class Mesh:

    def __init__(self, geom_args = None, v = None, f = None, vn = None, v_mask = None):
        self.extras = {}
        if geom_args is None:
            self.vertices = v
            self.faces = f
            self.color = None
            self.vn = vn
        else:
            self.initialize_shape(geom_args)

    @property
    def v_pos(self):
        return self.vertices

    @property
    def t_pos_idx(self):
        return self.faces

    @property
    def v_nrm(self):
        if self.vn is None:
            return calculate_vertex_normals(self.vertices, self.faces)
        else:
            return self.vn 

    def get_color(self,face_ids,bc): # if texture exists, get barycentric coordinate of meshes 
        points_tc = torch.sum(self.tc[self.ftc[face_ids]]*bc[...,None], dim=1).clip(0)
        points_tc[:,0] *= self.texture_image.shape[0]-1
        points_tc[:,1] *= self.texture_image.shape[1]-1
        points_tc = points_tc.round().to(torch.long)
        return self.texture_image[points_tc[:,1], points_tc[:,0]]/255

    def export_mesh(self,output_path):
        trimesh.Trimesh(vertices=self.vertices.detach().cpu().numpy(),faces=self.faces.cpu().numpy()).export(output_path)

    def make_icosphere(self, radius=1, subdivision_level=2, color=None):
        icosphere = trimesh.creation.icosphere(subdivision_level, radius)
        self.vertices = torch.tensor(icosphere.vertices, dtype=torch.float, device='cuda')
        self.faces = torch.tensor(icosphere.faces, dtype=torch.long, device='cuda')
        if color is None:
            self.color = torch.ones_like(self.vertices, device='cuda') * 0.5

    def load_mesh(self, mesh_path, normalize, colors = None):
        self.vertices, self.faces, _, _, valid = load_mesh(Path(mesh_path))
        # todo: allow loading colors from mesh file is exists
        texture_path = Path(Path(mesh_path).parent, Path(mesh_path).stem + '.mtl')
        self.has_texture = False
        if not colors is None:
            pass
        elif texture_path.exists():
            self.has_texture = True
            _,tc,_,_,ftc,_ = igl.read_obj(mesh_path)
            self.tc = torch.tensor(tc, device= self.vertices.device, dtype = torch.float)
            self.ftc = torch.tensor(ftc, device= self.vertices.device, dtype = torch.long)
            material = read_mtlfile(str(texture_path))
            parent_path = texture_path.parent
            texture_image = iio.imread(Path(parent_path,material[next(iter(material))]['map_Kd']))[:,:,:3]
            image = Image.fromarray(texture_image)
            image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
            self.texture_image = torch.tensor(np.array(image), device=self.vertices.device)
            tc = tc.clip(0,1)
        else:
            self.color = torch.ones_like(self.vertices, device='cuda') * 0.5
        if normalize:
            self.vertices = normalize_vertices(self.vertices, scale_by='norm')

    def initialize_shape(self, args):
        self.vn = None
        if args.init_mesh_mode == 'sphere':
            self.make_icosphere(**args.init_mesh_info)
        elif args.init_mesh_mode == 'initial_mesh':
            self.load_mesh(**args.init_mesh_info)
        else:
            raise NotImplementedError("invalid initial mesh mode")

