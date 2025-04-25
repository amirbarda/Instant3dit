from ROAR.util.geometry_utils import calculate_face_normals_and_areas, calculate_vertex_normals, \
    calculate_head_angles_per_vertex_per_face, calculate_edge_normals, calculate_generalize_vertex_normals
from ROAR.util.topology_utils import calc_edges, calculate_vertex_incident_vector, calculate_adjacent_faces, \
    calculate_vertex_incident_scalar, calculate_vertex_incident_vector_on_edges
from ROAR.util.remeshing_utils import remove_dummies, prepend_dummies
from ROAR.util.func import load_mesh, normalize_vertices, save_mesh_properly
import torch
import numpy as np
import torch_scatter
import os
import igl
import torch.nn.functional as tfunc


def calc_for_face_vertices_normals(fv, eps=1e-8):
    fv0 = fv[:, 0]
    fv1 = fv[:, 1]
    fv2 = fv[:, 2]
    face_normals = torch.cross(fv1 - fv0, fv2 - fv0, dim=-1)
    normal_mask = torch.norm(face_normals, dim=-1) < eps
    face_normals[~normal_mask] /= torch.norm(face_normals[~normal_mask], dim=-1, keepdim=True)
    face_normals[normal_mask] = 0
    return face_normals


def calc_head_angle(fv, head_vertex_id):
    selection_ids = torch.arange(fv.shape[0], device=fv.device)
    fv0 = fv[selection_ids, head_vertex_id]
    fv1 = fv[selection_ids, (head_vertex_id + 1) % 3]
    fv2 = fv[selection_ids, (head_vertex_id + 2) % 3]
    mask = (torch.norm(fv1 - fv0, dim=-1) < 1e-9) | (torch.norm(fv2 - fv0, dim=-1) < 1e-9)
    cos = torch.sum(tfunc.normalize(fv1 - fv0) * tfunc.normalize(fv2 - fv0), dim=-1, keepdim=True)
    cos = torch.clip(cos, max=1, min=-1)
    angle = torch.acos(cos)
    #assert torch.all(~torch.isnan(angle))
    angle[mask] = 0
    return angle


def calculate_double_face_areas(fv):
    fv0 = fv[:, 0]
    fv1 = fv[:, 1]
    fv2 = fv[:, 2]
    double_face_areas = torch.norm(torch.cross(fv1 - fv0, fv2 - fv0, dim=-1), dim=-1, keepdim=True)
    return double_face_areas


def calculate_frozen_faces(fv_frozen):
    # for each potential face calculate if is allowed
    fv_frozen0 = fv_frozen[:, 0]
    fv_frozen1 = fv_frozen[:, 1]
    fv_frozen2 = fv_frozen[:, 2]
    faces_frozen = fv_frozen0 & fv_frozen1 & fv_frozen2
    return faces_frozen


def calc_face_vertices_quality(fv):
    # lower is better
    fv0 = fv[:, 0]
    fv1 = fv[:, 1]
    fv2 = fv[:, 2]
    # calculate quality
    a = (fv0 - fv1).norm(dim=-1)
    b = (fv0 - fv2).norm(dim=-1)
    c = (fv1 - fv2).norm(dim=-1)
    s = (a + b + c) / 2
    quality = a * b * c / (8 * (s - a) * (s - b) * (s - c))
    # degenerate (collapsed) faces have quality 1 (best)
    quality = torch.nan_to_num(quality, nan=1)
    return quality


def vertex_color_change(v, f, v_edge, face_color, with_dummies=True):
    """

    :param v:
    :param f:
    :param v_edge:
    :param quality_threshold:
    :return:
    """
    # what-if analysis of incident color changes if edge is collapsed
    
    # todo: change face_colors to one hot encoding

    adj_faces, num_of_incident_faces, starting_idx = calculate_adjacent_faces(f)
    edges, _ = calc_edges(f, with_dummies=with_dummies)
    v_ids_for_incident_faces_per_edge = torch.repeat_interleave(edges.flatten(),
                                                                repeats=num_of_incident_faces[edges.flatten()])
    offset_vector = torch.arange(v_ids_for_incident_faces_per_edge.shape[0], device=v.device)
    cumsum = torch.cumsum(num_of_incident_faces[edges.flatten()], dim=0).roll(1)
    cumsum[0] = 0
    to_subtract = cumsum.repeat_interleave(repeats=num_of_incident_faces[edges.flatten()])
    offset_vector -= to_subtract
    starting_idx_for_v_ids = starting_idx[v_ids_for_incident_faces_per_edge]
    starting_idx_for_v_ids += offset_vector
    # create tensor that contains incident faces per edge
    faces_per_edge = adj_faces[starting_idx_for_v_ids]
    incident_face_vertices_ids_per_edge = f[faces_per_edge]
    incident_face_vertices_per_edge = v[incident_face_vertices_ids_per_edge]
    edge_degrees = torch.sum(num_of_incident_faces[edges], dim=-1)
    # replace vertices to v_edge wherever mask is true
    edge_vertices0 = edges[:, [0]].repeat_interleave(repeats=edge_degrees, dim=0)
    edge_vertices0 = torch.tile(edge_vertices0, [1, 3])
    edge_vertices1 = edges[:, [1]].repeat_interleave(repeats=edge_degrees, dim=0)
    edge_vertices1 = torch.tile(edge_vertices1, [1, 3])
    edge_vertices0_replace_mask = (edge_vertices0 == incident_face_vertices_ids_per_edge)
    if with_dummies:
        edge_vertices0_replace_mask &= (edge_vertices0 != 0)
    edge_vertices1_replace_mask = (edge_vertices1 == incident_face_vertices_ids_per_edge)
    if with_dummies:
        edge_vertices1_replace_mask &= (edge_vertices1 != 0)
    v0_replace = v_edge.repeat_interleave(repeats=edge_degrees, dim=0)[torch.any(edge_vertices0_replace_mask, dim=-1)]
    incident_face_vertices_per_edge[edge_vertices0_replace_mask] = v0_replace
    v1_replace = v_edge.repeat_interleave(repeats=edge_degrees, dim=0)[torch.any(edge_vertices1_replace_mask, dim=-1)]
    incident_face_vertices_per_edge[edge_vertices1_replace_mask] = v1_replace
    # calculate vertex color after collapse
    # get face color per face per edge
    colors_per_face_per_edge = face_color[faces_per_edge]
    # scatter the angle-weighted color to each vertex after collapse and return color
    vertex_color_after = torch.zeros([v_edge.shape[0], face_color.shape[-1]], device=v.device)
    head_vertex_id = torch.max(edge_vertices0_replace_mask | edge_vertices1_replace_mask, dim=-1)[1]
    head_angles = calc_head_angle(incident_face_vertices_per_edge, head_vertex_id)
    sum_head_angles = torch.zeros(v_edge.shape[0], device=v.device)
    repeated_edge_ids = torch.arange(edges.shape[0], device=v.device).repeat_interleave(repeats=edge_degrees)
    torch_scatter.scatter_add(out=vertex_color_after, dim=0, index=repeated_edge_ids,
                              src=colors_per_face_per_edge * head_angles)
    # todo: bug in scatter_add? sometimes get NaNs (solved?)
    vertex_color_after = torch.nan_to_num(vertex_color_after, nan=0)
    torch_scatter.scatter_add(out=sum_head_angles, dim=0, index=repeated_edge_ids, src=head_angles.flatten())
    vertex_color_after /= sum_head_angles.unsqueeze(-1)
    vertex_color_after = torch.nan_to_num(vertex_color_after, nan=0)
    # trimesh.visual.color.face_to_vertex_color(m,self.face_colors.detach().cpu().numpy())
    # if torch.any(torch.isnan(vertex_color_after)):
    #     import trimesh
    #     m = trimesh.Trimesh(vertices=v.detach().cpu().numpy(),
    #                         faces=f.cpu().numpy(),
    #                         face_colors=face_color.detach().cpu().numpy())
    #     obj = trimesh.exchange.ply.export_ply(m)
    #     path = '/disk2/amirb/ROAR/tests/full_runs/color_cube_with_face_attributes_with_only_vertex_movement/result/failed.ply'
    #     with open(str(path), 'wb') as file:
    #         file.write(obj)
    #     import igl
    #     igl.write_obj('/disk2/amirb/ROAR/tests/full_runs/color_cube_with_face_attributes_with_only_vertex_movement/result/failed.obj',
    #                   v.detach().cpu().numpy(),
    #                   f.cpu().numpy())
    # assert torch.all(~torch.isnan(vertex_color_after))
    return vertex_color_after


def simulated_faces_after_collapse(v, f, v_edge, quality_threshold=5, with_dummies=True, v_mask = None):
    """

    :param v:
    :param f:
    :param v_edge:
    :param quality_threshold:
    :return:
    """
    adj_faces, num_of_incident_faces, starting_idx = calculate_adjacent_faces(f)
    edges, _ = calc_edges(f, with_edge_to_face=False, with_dummies = with_dummies)
    v_ids_for_incident_faces_per_edge = torch.repeat_interleave(edges.flatten(),
                                                                repeats=num_of_incident_faces[edges.flatten()])
    offset_vector = torch.arange(v_ids_for_incident_faces_per_edge.shape[0], device=v.device)
    cumsum = torch.cumsum(num_of_incident_faces[edges.flatten()], dim=0).roll(1)
    cumsum[0] = 0
    to_subtract = cumsum.repeat_interleave(repeats=num_of_incident_faces[edges.flatten()])
    offset_vector -= to_subtract
    starting_idx_for_v_ids = starting_idx[v_ids_for_incident_faces_per_edge]
    starting_idx_for_v_ids += offset_vector
    # create tensor that contains incident faces per edge
    faces_per_edge = adj_faces[starting_idx_for_v_ids]
    incident_face_vertices_ids_per_edge = f[faces_per_edge]
    incident_face_vertices_per_edge = v[incident_face_vertices_ids_per_edge]
    edge_degrees = torch.sum(num_of_incident_faces[edges], dim=-1) # num_of_incident_faces is per vertex, edge degrees is the sum of its vertex degrees
    # replace vertices to v_edge wherever mask is true
    edge_vertices0 = edges[:, [0]].repeat_interleave(repeats=edge_degrees, dim=0)
    edge_vertices0 = torch.tile(edge_vertices0, [1, 3])
    edge_vertices1 = edges[:, [1]].repeat_interleave(repeats=edge_degrees, dim=0)
    edge_vertices1 = torch.tile(edge_vertices1, [1, 3])
    edge_vertices0_replace_mask = (edge_vertices0 == incident_face_vertices_ids_per_edge)
    if with_dummies:
        edge_vertices0_replace_mask &= (edge_vertices0 != 0)
    edge_vertices1_replace_mask = (edge_vertices1 == incident_face_vertices_ids_per_edge)
    if with_dummies:
        edge_vertices1_replace_mask &= (edge_vertices1 != 0)
    v0_replace = v_edge.repeat_interleave(repeats=edge_degrees, dim=0)[torch.any(edge_vertices0_replace_mask, dim=-1)]
    incident_face_vertices_per_edge[edge_vertices0_replace_mask] = v0_replace
    v1_replace = v_edge.repeat_interleave(repeats=edge_degrees, dim=0)[torch.any(edge_vertices1_replace_mask, dim=-1)]
    incident_face_vertices_per_edge[edge_vertices1_replace_mask] = v1_replace

    # incident_face_vertices_per_edge now contains state after collapse
    normals_before = calc_for_face_vertices_normals(v[incident_face_vertices_ids_per_edge])
    # calc face normals after
    normals_after = calc_for_face_vertices_normals(incident_face_vertices_per_edge)
        

    flip_detection = torch.sum(normals_before * normals_after, dim=-1) < 0
    # scatter max for all vertices to edges
    repeated_edge_ids = torch.arange(edges.shape[0], device=v.device).repeat_interleave(repeats=edge_degrees)
    edge_mask = torch_scatter.scatter_max(flip_detection.float(), repeated_edge_ids, dim=0)[0]

    if quality_threshold is not None:
        incident_faces_quality = calc_face_vertices_quality(incident_face_vertices_per_edge)
        bad_face_detection = incident_faces_quality > quality_threshold
        repeated_edge_ids = torch.arange(edges.shape[0], device=v.device).repeat_interleave(repeats=edge_degrees)
        face_quality_mask = torch_scatter.scatter_max(bad_face_detection.float(), repeated_edge_ids, dim=0)[0]
        edge_mask = edge_mask + face_quality_mask


    if v_mask is not None:
        fv_frozen = v_mask[incident_face_vertices_ids_per_edge]   
        frozen_faces_before = calculate_frozen_faces(fv_frozen)
        # find if edge is border edge, for and repeat for each vertex incidient to edge
        border_edge_mask = v_mask[edges].sum(dim=-1)==1 # #EV x 1, true if edge is border edge, false otherwise
        border_edge_vertices_mask = border_edge_mask.repeat_interleave(repeats=edge_degrees).unsqueeze(-1)
        # replace all border edges with frozen vertices after simulated collapse
        fv_frozen_after = fv_frozen
        fv_frozen_after[edge_vertices0_replace_mask & border_edge_vertices_mask] = True
        fv_frozen_after[edge_vertices1_replace_mask & border_edge_vertices_mask] = True
        # find list of incident faces 
        frozen_faces_after = calculate_frozen_faces(fv_frozen_after)
        # edge collapse should not turn an allowed face to a frozen one, but in case of border edge the two incident triangles are destroyed so they can become frozen
        is_incident_to_edge_to_collapse = (edge_vertices0_replace_mask | edge_vertices1_replace_mask).sum(dim=-1) == 2
        bad_border_detection = (frozen_faces_after != frozen_faces_before)
        bad_border_detection[is_incident_to_edge_to_collapse] = False
        repeated_edge_ids = torch.arange(edges.shape[0], device=v.device).repeat_interleave(repeats=edge_degrees)
        allowed_vertices_mask = torch_scatter.scatter_max(bad_border_detection.float(), repeated_edge_ids, dim=0)[0]
        edge_mask = edge_mask + allowed_vertices_mask

    return edge_mask > 0


def color_to_geometry_delta(v, f, face_colors):
    edges, ef, fe = calc_edges(f, with_edge_to_face=True, with_dummies=False)
    # todo: calculate vertex colors
    _, face_areas = calculate_face_normals_and_areas(v, f)
    vertex_colors = calculate_vertex_incident_vector(v.shape[0], f, (1 - face_colors) * face_areas)
    face_areas = calculate_vertex_incident_scalar(v.shape[0], f, face_areas)
    vertex_colors /= face_areas
    # todo: calculate vertex deltas
    E = edges.shape[0]
    neighbor_smooth = torch.zeros_like(v)  # V,S - mean of 1-ring vertices
    torch_scatter.scatter_mean(src=vertex_colors[edges].flip(dims=[1]).reshape(E * 2, -1),
                               index=edges.reshape(E * 2, 1),
                               dim=0, out=neighbor_smooth)
    laplace = vertex_colors - neighbor_smooth[:, :3]
    # todo: calculate vertex normals
    vertex_normals = calculate_vertex_normals(v, f)
    return torch.norm(vertex_colors, dim=-1, keepdim=True) * vertex_normals


def calculate_Q_f(v, f):
    # todo: optional color to vertices
    # if face_colors is None:
    face_normals, _ = calculate_face_normals_and_areas(v, f)
    d = -torch.sum(v[f[:, 0]] * face_normals, dim=-1, keepdim=True)  # OK
    A = torch.bmm(face_normals.unsqueeze(-1), face_normals.unsqueeze(-1).transpose(-1, -2))
    b = d * face_normals
    c = d ** 2
    return A, b, c


def calculate_Q_e(v, f, edges):
    edge_normal = calculate_edge_normals(v, f)
    # todo: calculate quadric per edge
    d = -torch.sum(v[edges[:, 0]] * edge_normal, dim=-1, keepdim=True)  # OK
    A = torch.bmm(edge_normal.unsqueeze(-1), edge_normal.unsqueeze(-1).transpose(-1, -2))
    b = d * edge_normal
    c = d ** 2
    # todo: distribute quadrics per vertex
    return A, b, c


def calculate_Q_vv(v, f):
    # todo: calculate vertex normal
    # todo: calculate extra quadric per vertex
    vertex_normals = calculate_generalize_vertex_normals(v, f)
    d = -torch.sum(v * vertex_normals, dim=-1, keepdim=True)  # OK
    A = torch.bmm(vertex_normals.unsqueeze(-1), vertex_normals.unsqueeze(-1).transpose(-1, -2))
    b = d * vertex_normals
    c = d ** 2
    return A, b, c


def calculate_Q_v(v, f, with_dummies):
    edges, _ = calc_edges(f, with_dummies = with_dummies)
    # face_normals, _ = calculate_face_normals_and_areas(v, f)
    A, b, c = calculate_Q_f(v, f)
    A_e, b_e, c_e = calculate_Q_e(v, f, edges)
    A_vv, b_vv, c_vv = calculate_Q_vv(v, f)
    # if face_colors is not None:
    #     color_to_geometry_delta(v, f, face_colors)
    # _, face_areas = calculate_face_normals_and_areas(v, f)
    Abc = torch.cat([A.reshape(f.shape[0], -1), b, c], dim=-1)
    Abc_v = calculate_vertex_incident_vector(v.shape[0], f, Abc*100)
    Abc_e = torch.cat([A_e.reshape(edges.shape[0], -1), b_e, c_e], dim=-1)
    Abc_v += calculate_vertex_incident_vector_on_edges(v.shape[0], edges, Abc_e)
    # vertex_areas = calculate_vertex_incident_vector(v.shape[0], f, face_areas)
    # Abc_v /= vertex_areas
    A_v = Abc_v[:, :9].reshape(-1, 3, 3) + A_vv*100
    b_v = Abc_v[:, 9:12] + b_vv*100
    c_v = Abc_v[:, 12] + c_vv.flatten()*100
    # todo: use torch scatter to add the vectors according to
    return A_v, b_v, c_v


def Qslim_plane_distance(v_bar, A_tot, b_tot, c_tot):
    A_score = torch.bmm(v_bar.unsqueeze(-2), torch.bmm(A_tot, v_bar.unsqueeze(-1))).flatten()
    b_score = 2 * torch.sum(b_tot * v_bar, dim=-1)
    distance = A_score + b_score + c_tot
    distance[distance < 0] = 0
    return distance


def calculate_QSlim_score(v, f, face_colors=None, color_score = 1, quality_threshold=None, penalty_weight = 10, avoid_folding_faces=True, min_edge_length = 1e-5, with_dummies=False, output_debug_info = False, v_mask = None):
    """
    higher QSlim score = edge encodes more geometry
    :param face_colors:
    :param face_colors:
    :param v:
    :param f:
    :param quality_threshold:
    :param avoid_folding_faces:
    :param vertex_norm_threshold:
    :param with_dummies:
    :return:
    """

    debug_info = {}

    A_v, b_v, c_v = calculate_Q_v(v, f, with_dummies)
    edges, _ = calc_edges(f, with_dummies = with_dummies)
    v0 = edges[:, 0]
    v1 = edges[:, 1]
    # v_mid_bar = (v[v0] + v[v1]) / 2
    v0_bar = v[v0]
    v1_bar = v[v1]
    # return the minimal scores AND the location of collapse
    A_tot = A_v[v0] + A_v[v1]
    b_tot = b_v[v0] + b_v[v1]
    c_tot = c_v[v0] + c_v[v1]
    # qslim_score_mid = Qslim_plane_distance(v_mid_bar, A_tot, b_tot, c_tot)
    qslim_score_0 = Qslim_plane_distance(v0_bar, A_tot, b_tot, c_tot) # score of moving v1 to v0
    qslim_score_1 = Qslim_plane_distance(v1_bar, A_tot, b_tot, c_tot) # score of moving v0 to v1
    edge_lengths = torch.norm(v1_bar - v0_bar, dim=-1)
    if avoid_folding_faces or v_mask is not None:
        # multiply qslim score by this mask
        qslim_score_0[simulated_faces_after_collapse(v, f, v0_bar, quality_threshold=quality_threshold,
                                                     with_dummies=with_dummies, v_mask = v_mask) & (edge_lengths>min_edge_length)] = np.inf
        qslim_score_1[simulated_faces_after_collapse(v, f, v1_bar, quality_threshold=quality_threshold,
                                                     with_dummies=with_dummies, v_mask = v_mask) & (edge_lengths>min_edge_length)] = np.inf
        
        # assert torch.all(~torch.isnan(qslim_score_0))
        # assert torch.all(~torch.isnan(qslim_score_1))
    # calculate color scores
    # todo: calculate using one-hot encoding
    if face_colors is not None:
        vertex_colors_after_0 = vertex_color_change(v, f, v0_bar, face_colors, with_dummies=with_dummies)
        vertex_colors_after_1 = vertex_color_change(v, f, v1_bar, face_colors, with_dummies=with_dummies)
        # assert torch.all(~torch.isnan(vertex_colors_after_0))
        # assert torch.all(~torch.isnan(vertex_colors_after_1))
        # calculate current vertex color weighted by head angle areas
        head_angle, vertex_ids, face_ids = calculate_head_angles_per_vertex_per_face(v, f)
        weighted_colors = torch_scatter.scatter_add(face_colors[face_ids] * head_angle, vertex_ids, 0)
        head_angle_sum = torch_scatter.scatter_add(head_angle, vertex_ids, 0)
        weighted_colors /= head_angle_sum
        #assert torch.all(~torch.isnan(weighted_colors))
        # if nan - make zero
        color_score0 = penalty_weight * torch.norm(torch.abs(weighted_colors[v0] - vertex_colors_after_0), dim=-1) * torch.max(
            (1 + edge_lengths) ** 3)
        color_score0 = torch.nan_to_num(color_score0, nan=0)
        color_score1 = penalty_weight * torch.norm(torch.abs(weighted_colors[v1] - vertex_colors_after_1), dim=-1) * torch.max(
            (1 + edge_lengths) ** 3)
        color_score1 = torch.nan_to_num(color_score1, nan=0)
        qslim_score_0 += color_score0
        qslim_score_1 += color_score1

    if v_mask is not None:
        #v_mask is True if vertex is frozen
        qslim_score_1[v_mask[v0]] = np.inf
        qslim_score_0[v_mask[v1]] = np.inf

    qslim_scores = torch.stack([qslim_score_0, qslim_score_1])  # qslim_score_mid])
    min_qslim_val, min_qslim_id = torch.min(qslim_scores, dim=0)
    # min_qslim_id = min_qslim_id.float()
    # min_qslim_id[min_qslim_id > 1.5] = 0.5
    # qslim_score = 1-qslim_score/torch.max(qslim_score)

    if output_debug_info:
        debug_info['q0_arr'] = qslim_score_0.cpu().numpy()
        debug_info['q1_arr'] = qslim_score_1.cpu().numpy()

    return min_qslim_val, min_qslim_id.float(), debug_info


if __name__ == "__main__":
    torch.cuda.set_device('cuda:3')
    output_path = "/disk2/amirb/GenerativeGraphMesh/continous_remeshing/test_qslim/"
    mesh_path = "/disk2/amirb/GenerativeGraphMesh/continous_remeshing/results_summary/deer_simplified.obj"
    v, f, _ = load_mesh(mesh_path)
    v = normalize_vertices(v)

    # for each edge, calculate the midpoint
    save_mesh_properly(v, f, os.path.join(output_path, "mesh.obj"))
    face_normals_before, face_areas = calculate_face_normals_and_areas(v, f)
    np.savetxt(os.path.join(output_path, "collapsed_face_normals.txt"), face_normals_before.cpu().numpy())
    v, f = prepend_dummies(v, f)
    edges, _ = calc_edges(f)
    v0 = v[edges[:, 0]]
    v1 = v[edges[:, 1]]
    v_mid = (v0 + v1) / 2
    simulated_faces_after_collapse(v, f, v_mid)
    v, f = remove_dummies(v, f)
    mid = (v[471] + v[0]) / 2
    v[0] = mid
    v[471] = mid
    save_mesh_properly(v, f, os.path.join(output_path, "collapsed_mesh.obj"))
    face_normals_after, face_areas = calculate_face_normals_and_areas(v, f)
    np.savetxt(os.path.join(output_path, "face_normals.txt"), face_normals_after.cpu().numpy())

    QSlim_score, _ = calculate_QSlim_score(v, f)
    edge, _ = calc_edges(f)
    edge = edge.cpu().numpy()
    igl.write_obj(os.path.join(output_path, "source_mesh.obj"), v.cpu().numpy(), f.cpu().numpy())
    np.savetxt(os.path.join(output_path, "edges.txt"), edge)
    np.savetxt(os.path.join(output_path, "qslim.txt"), QSlim_score.cpu().numpy())
    print('done')
