import torch
from ROAR.util.topology_utils import calc_edges
from ROAR.util.geometry_utils import calc_face_normals, project_pc_to_mesh, calculate_dihedral_angles, calc_edge_length
from ROAR.sampling.random_sampler import sample_points_from_meshes, sample_points_from_edges
import igl
import numpy as np


def triangle_soup_to_point_cloud_random(v, f, num_samples, return_occlusion_score=False, threshold=0.2 , face_color_id=None):
    """
    :param v:
    :param f:
    :param c: #Fx3 Tensor for face colors
    :return: point cloud sampled on mesh, without occluded vertices
    """
    sampled_v, sampled_n, face_ids = sample_points_from_meshes(v.unsqueeze(0), f.unsqueeze(0), num_samples=num_samples,
                                                        return_normals=True)
    if face_color_id is not None:
        sampled_color_id = face_color_id[face_ids]
    sampled_v = sampled_v.squeeze(0)
    sampled_n = sampled_n.squeeze(0)
    sampled_v_occlusion = igl.fast_winding_number_for_meshes(v.cpu().numpy(), f.cpu().numpy(),
                                                             sampled_v.cpu().numpy() + 1e-3 * sampled_n.cpu().numpy())
    if return_occlusion_score:
        if face_color_id is None:
            return sampled_v, sampled_n, sampled_v_occlusion
        else:
            return sampled_v, sampled_n, sampled_color_id, sampled_v_occlusion
    sampled_v = sampled_v[np.abs(sampled_v_occlusion) < threshold]
    sampled_n = sampled_n[np.abs(sampled_v_occlusion) < threshold]
    if face_color_id is not None:
        sampled_color_id = sampled_color_id[np.abs(sampled_v_occlusion) < threshold]
    sampled_n = sampled_n / (sampled_n.norm(dim=-1)[:, None] + 1e-8)

    if face_color_id is not None:
        return sampled_v, sampled_n, sampled_color_id
    else:
        return sampled_v, sampled_n


def project_points_to_triangle_soup(points, v, f, face_colors_id = None, point_normals=None):
    _, parent_faces, projected_points = project_pc_to_mesh(points, v, f)
    face_normals = calc_face_normals(v, f, normalize=True)
    projected_normals = face_normals[parent_faces]
    if face_colors_id is not None:
        point_colors = face_colors_id[parent_faces]
    if point_normals is not None:
        # orient normals according to point normals
        score = torch.sum(point_normals * projected_normals, dim=-1)
        projected_normals[score < 1e-5] *= -1
    if face_colors_id is not None:
        return projected_points, projected_normals, point_colors
    else:
        return projected_points, projected_normals, None


def sample_points_from_masked_edges(vs, f,
                                    num_samples: int = 10000,
                                    dihedral_thres=0.5,
                                    min_edge_lim=0.005,
                                    return_normals: bool = False):
    # todo: remove points that are closer than epsilon to each other
    dihedral_angles = calculate_dihedral_angles(vs, f).squeeze(-1)
    edges, _, ef = calc_edges(f, with_edge_to_face=True, with_dummies=False)
    edge_lengths = calc_edge_length(vs, edges)
    masked_edges = (dihedral_angles > dihedral_thres) & (edge_lengths > min_edge_lim)
    face_normals = calc_face_normals(vs, f, normalize=True)
    edge_face_normals = None
    if return_normals:
        edge_face_normals = face_normals[torch.stack(ef[:, 0, 0], ef[:, 0, 1], dim=-1)]
    return sample_points_from_edges(vs, edges[masked_edges], num_samples, edge_face_normals)
