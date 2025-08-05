import nvdiffrast.torch as dr
import torch

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt

# PyTorch Tensor type
from torch import Tensor
class NVDiffRasterizerContext:
    def __init__(self, context_type: str, device: torch.device) -> None:
        self.device = device
        self.ctx = self.initialize_context(context_type, device)

    def initialize_context(
        self, context_type: str, device: torch.device
    ) -> Union[dr.RasterizeCudaContext, dr.RasterizeCudaContext]:
        if context_type == "gl":
            return dr.RasterizeCudaContext(device=device)
        elif context_type == "cuda":
            return dr.RasterizeCudaContext(device=device)
        else:
            raise ValueError(f"Unknown rasterizer context type: {context_type}")

    def vertex_transform(
        self, verts: Float[Tensor, "Nv 3"], mvp_mtx: Float[Tensor, "B 4 4"]
    ) -> Float[Tensor, "B Nv 4"]:
        verts_homo = torch.cat(
            [verts, torch.ones([verts.shape[0], 1]).to(verts)], dim=-1
        )
        return torch.matmul(verts_homo, mvp_mtx.permute(0, 2, 1))

    def rasterize(
        self,
        pos: Float[Tensor, "B Nv 4"],
        tri: Integer[Tensor, "Nf 3"],
        resolution: Union[int, Tuple[int, int]],
    ):
        # rasterize in instance mode (single topology)
        return dr.rasterize(self.ctx, pos.float(), tri.int(), resolution, grad_db=True)

    def rasterize_one(
        self,
        pos: Float[Tensor, "Nv 4"],
        tri: Integer[Tensor, "Nf 3"],
        resolution: Union[int, Tuple[int, int]],
    ):
        # rasterize one single mesh under a single viewpoint
        rast, rast_db = self.rasterize(pos[None, ...], tri, resolution)
        return rast[0], rast_db[0]

    def antialias(
        self,
        color: Float[Tensor, "B H W C"],
        rast: Float[Tensor, "B H W 4"],
        pos: Float[Tensor, "B Nv 4"],
        tri: Integer[Tensor, "Nf 3"],
    ) -> Float[Tensor, "B H W C"]:
        return dr.antialias(color.float(), rast, pos.float(), tri.int())

    def interpolate(
        self,
        attr: Float[Tensor, "B Nv C"],
        rast: Float[Tensor, "B H W 4"],
        tri: Integer[Tensor, "Nf 3"],
        rast_db=None,
        diff_attrs=None,
    ) -> Float[Tensor, "B H W C"]:
        return dr.interpolate(
            attr.float(), rast, tri.int(), rast_db=rast_db, diff_attrs=diff_attrs
        )

    def interpolate_one(
        self,
        attr: Float[Tensor, "Nv C"],
        rast: Float[Tensor, "B H W 4"],
        tri: Integer[Tensor, "Nf 3"],
        rast_db=None,
        diff_attrs=None,
    ) -> Float[Tensor, "B H W C"]:
        return self.interpolate(attr[None, ...], rast, tri, rast_db, diff_attrs)
    


    def render(self,v,f, masked_faces, mvp, height = 512, width = 512):


        v_pos_clip: Float[Tensor, "B Nv 4"] = self.vertex_transform(
            v, mvp
        )
        rast, _ = self.rasterize(v_pos_clip, f, (height, width))

        # returned sampled vertices
        # fv = v[f[rast[:,:,:,-1].long()-1]]
        # fv0 = fv[:,:,:,0]
        # fv1 = fv[:,:,:,1]
        # fv2 = fv[:,:,:,2]
        #u = rast[:,:,:,[0]]
        #v = rast[:,:,:,[1]]
        #x = (fv0)*u + (fv1)*v + fv2*(1-u-v)
        # get face id for each pixel
        
        face_ids = rast[:,:,:,-1].long() - 1
        render_mask =  ((face_ids >= 0) & (masked_faces[face_ids])).int().unsqueeze(-1).repeat([1,1,1,3])
        render_mask = render_mask.float()
        render_mask = render_mask.transpose(1,-1)
        
        return render_mask
