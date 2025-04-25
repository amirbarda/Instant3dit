# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
This module implements utility functions for sampling points from
batches of meshes.
"""
import sys
sys.path.append("/home/abarda/MeshSDF")
from typing import Tuple, Union
import torch
from ROAR.util.geometry_utils import calculate_dihedral_angles, calc_edge_length
import time

def calculate_centeroid(vs):
    return torch.mean(vs,dim=1)

def bounding_sphere_radius(vs):
    centered_vs = vs - calculate_centeroid(vs)
    distances = torch.norm(centered_vs,dim=-1)
    return torch.max(distances)

def sample_points_in_cube(num_samples, edge_length, device = 'cuda'):
    """
    returns $num_samples points on the cube with edge_length=2*$half_length centered at the origin
    """
    samples = (torch.rand([num_samples,3],device=device)-0.5) * edge_length
    return samples

def sample_pertrubed_points_from_mesh(vs,f,num_samples, factor=1024):
    samples, _ = sample_points_from_meshes(vs,f,num_samples=num_samples)
    r = bounding_sphere_radius(vs)
    pertrubation_vectors = torch.randn_like(samples, device=samples.device)*(r/factor)
    return samples + pertrubation_vectors

def sample_points_from_meshes(
        vs, f,
        num_samples: int = 10000,
        return_normals: bool = False,
        return_bc_coords = False,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Convert a batch of meshes to a batch of pointclouds by uniformly sampling
    points on the surface of the mesh with probability proportional to the
    face area.

    Args:
        meshes: A Meshes object with a batch of N meshes.
        num_samples: Integer giving the number of point samples per mesh.
        return_normals: If True, return normals for the sampled points.
        return_textures: If True, return textures for the sampled points.

    Returns:
        3-element tuple containing

        - **samples**: FloatTensor of shape (N, num_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch. For empty
          meshes the corresponding row in the samples array will be filled with 0.
        - **normals**: FloatTensor of shape (N, num_samples, 3) giving a normal vector
          to each sampled point. Only returned if return_normals is True.
          For empty meshes the corresponding row in the normals array will
          be filled with 0.
        - **textures**: FloatTensor of shape (N, num_samples, C) giving a C-dimensional
          texture vector to each sampled point. Only returned if return_textures is True.
          For empty meshes the corresponding row in the textures array will
          be filled with 0.

        Note that in a future releases, we will replace the 3-element tuple output
        with a `Pointclouds` datastructure, as follows

        .. code-block:: python

            Pointclouds(samples, normals=normals, features=textures)
    """
    # if meshes.isempty():
    #     raise ValueError("Meshes are empty.")

    # dont support batch for now.
    verts = vs[0]
    faces = f[0]
    # if not torch.isfinite(verts).all():
    #     raise ValueError("Meshes contain nan or inf.")

    # if return_textures and meshes.textures is None:
    #     raise ValueError("Meshes do not contain textures.")

    device = verts.device

    # faces = soa[6][0]#meshes.faces_packed()
    # mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = 1  # len(meshes)
    num_valid_meshes = 1  # torch.sum(meshes.valid)  # Non empty meshes.

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=device)

    # Only compute samples for non empty meshes
    with torch.no_grad():
        # areas, _ = mesh_face_areas_normals(verts, faces[:-1])  # Face areas can be zero.
        face_normals = torch.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                                   verts[faces[:, 2]] - verts[faces[:, 1]])
        areas = torch.linalg.norm(face_normals, dim=-1) / 2
        # assert(torch.all(torch.isclose(areas, areas2)))
        assert not torch.all(torch.isclose(areas, torch.zeros_like(areas)))
        # max_faces = meshes.num_faces_per_mesh().max().item()
        # areas_padded = packed_to_padded(
        #     areas, mesh_to_face[meshes.valid], max_faces
        # )  # (N, F)

        # TODO (gkioxari) Confirm multinomial bug is not present with real data.
        sample_face_idxs = areas.multinomial(
            num_samples, replacement=True
        )  # (N, num_samples)
        # sample_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces[:]]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(
        num_valid_meshes, num_samples, verts.dtype, verts.device
    )

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]
    samples[0] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c

    if return_normals:
        # Initialize normals tensor with fill value 0 for empty meshes.
        # Normals for the sampled points are face normals computed from
        # the vertices of the face in which the sampled point lies.
        normals = torch.zeros((num_meshes, num_samples, 3), device=device)
        vert_normals = (v1 - v0).cross(v2 - v1, dim=1)
        vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
            min=sys.float_info.epsilon
        )
        vert_normals = vert_normals[sample_face_idxs]
        normals[0] = vert_normals

    # return
    # TODO(gkioxari) consider returning a Pointclouds instance [breaking]
    if return_normals:  # return_textures is False
        # pyre-fixme[61]: `normals` may not be initialized here.
        return samples, normals, sample_face_idxs
    if return_bc_coords:
        return samples, torch.cat([w0,w1,w2], dim=0).T, sample_face_idxs
    return samples, sample_face_idxs


def _rand_barycentric_coords(
        size1, size2, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to generate random barycentric coordinates which are uniformly
    distributed over a triangle.

    Args:
        size1, size2: The number of coordinates generated will be size1*size2.
                      Output tensors will each be of shape (size1, size2).
        dtype: Datatype to generate.
        device: A torch.device object on which the outputs will be allocated.

    Returns:
        w0, w1, w2: Tensors of shape (size1, size2) giving random barycentric
            coordinates
    """
    uv = torch.rand(2, size1, size2, dtype=dtype, device=device)
    u, v = uv[0], uv[1]
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    # pyre-fixme[7]: Expected `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` but
    #  got `Tuple[float, typing.Any, typing.Any]`.
    return w0, w1, w2


def sample_points_from_edges(verts, edges, num_samples, edge_face_normals = None):

    device = verts.device

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_samples, 3), device=device)

    # Only compute samples for non empty meshes
    with torch.no_grad():
        lengths = calc_edge_length(verts, edges)
        assert not torch.all(torch.isclose(lengths, torch.zeros_like(lengths)))

        # TODO (gkioxari) Confirm multinomial bug is not present with real data.
        sample_face_idxs = lengths.multinomial(
            num_samples, replacement=True
        )  # (N, num_samples)
        # sample_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

    # Get the vertex coordinates of the sampled faces.
    edge_verts = verts[edges[:]]
    v0, v1 = edge_verts[:, 0], edge_verts[:, 1]

    # Randomly generate barycentric coords.
    t = torch.rand(num_samples, device=verts.device)

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    samples = t[:, None] * a + (1-t[:, None]) * b

    if edge_face_normals is not None:
        normals = torch.zeros((num_samples, 3), device=device)
        normal_idx = torch.round(torch.rand(num_samples, device=verts.device))
        normals = edge_face_normals[sample_face_idxs][:,normal_idx]

    # return
    # TODO(gkioxari) consider returning a Pointclouds instance [breaking]
    if edge_face_normals is not None:  # return_textures is False
        # pyre-fixme[61]: `normals` may not be initialized here.
        return samples, normals, sample_face_idxs
    return samples, sample_face_idxs


if __name__ == "__main__":
    """
    test all samplings of points from mesh
    """
    from ROAR.util.func import load_mesh, normalize_vertices, save_pc, save_mesh
    from pathlib import Path
    import numpy as np

    #v,f,_,_,_ = load_mesh(Path("tmp_tests/NGLOD/sampling/armadillo.obj"))
    v,f,_,_,_ = load_mesh(Path("tmp_tests/NGLOD/sampling/sphere/sphere.obj"))

    v = normalize_vertices(v)

    num_samples = 1000000

    #save_mesh(v,f,Path("tmp_tests/NGLOD/sampling/armadillo_normalized.obj"))
    save_mesh(v,f,Path("tmp_tests/NGLOD/sampling/sphere/sphere_normalized.obj"))

    start = time.time()
    samples_on_mesh, _ = sample_points_from_meshes(v.unsqueeze(0),f.unsqueeze(0),num_samples=num_samples//4)
    sampled_pertrubed_on_mesh = sample_pertrubed_points_from_mesh(v.unsqueeze(0),f.unsqueeze(0),num_samples=num_samples*3//8)
    samples_in_cube = sample_points_in_cube(num_samples//8,2)

    end = time.time()

    print('sampled {} points in {}s'.format(num_samples, end-start))

    # save_samples
    save_pc("tmp_tests/NGLOD/sampling/samples_on_mesh.ply", samples_on_mesh[0])    
    save_pc("tmp_tests/NGLOD/sampling/sampled_pertrubed_on_mesh.ply", sampled_pertrubed_on_mesh[0])    
    save_pc("tmp_tests/NGLOD/sampling/samples_in_cube.ply", samples_in_cube)    
    print('done')