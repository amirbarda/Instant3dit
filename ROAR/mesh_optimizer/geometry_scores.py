import torch
from torch_scatter import scatter_max
from ROAR.util.geometry_utils import calculate_face_folds, calculate_vertex_normals, calc_face_normals
from ROAR.util.topology_utils import calculate_vertex_incident_scalar, get_maximal_face_value_over_vertices, calc_edges
from ROAR.util.remeshing_utils import calc_face_collapses
import numpy as np
import igl


def gradient(x, f, method='autodiff'):
    """Compute gradient.
    """
    if method == 'autodiff':
        with torch.enable_grad():
            x = x.requires_grad_(True)
            y = f(x)
            grad = torch.autograd.grad(y, x, 
                                       grad_outputs=torch.ones_like(y), create_graph=True)[0]
    elif method == 'tetrahedron':
        h = 1.0 / (64.0 * 3.0)
        k0 = torch.tensor([ 1.0, -1.0, -1.0], device=x.device, requires_grad=False)
        k1 = torch.tensor([-1.0, -1.0,  1.0], device=x.device, requires_grad=False)
        k2 = torch.tensor([-1.0,  1.0, -1.0], device=x.device, requires_grad=False)
        k3 = torch.tensor([ 1.0,  1.0,  1.0], device=x.device, requires_grad=False)
        h0 = torch.tensor([ h, -h, -h], device=x.device, requires_grad=False)
        h1 = torch.tensor([-h, -h,  h], device=x.device, requires_grad=False)
        h2 = torch.tensor([-h,  h, -h], device=x.device, requires_grad=False)
        h3 = torch.tensor([ h,  h,  h], device=x.device, requires_grad=False)
        h0 = x + h0
        h1 = x + h1
        h2 = x + h2
        h3 = x + h3
        h0 = h0.detach()
        h1 = h1.detach()
        h2 = h2.detach()
        h3 = h3.detach()
        h0 = k0 * f(h0)
        h1 = k1 * f(h1)
        h2 = k2 * f(h2)
        h3 = k3 * f(h3)
        grad = (h0+h1+h2+h3) / (h*4.0)
    elif method == 'finitediff':
        min_dist = 1.0/(64.0 * 3.0)
        eps_x = torch.tensor([min_dist, 0.0, 0.0], device=x.device)
        eps_y = torch.tensor([0.0, min_dist, 0.0], device=x.device)
        eps_z = torch.tensor([0.0, 0.0, min_dist], device=x.device)

        grad = torch.cat([f(x + eps_x) - f(x - eps_x),
                          f(x + eps_y) - f(x - eps_y),
                          f(x + eps_z) - f(x - eps_z)], dim=-1)
        grad = grad / (min_dist*2.0)
    elif method == 'multilayer':
        # TODO: Probably remove this
        grad = []
        with torch.enable_grad():
            _y = f.sdf(x, return_lst=True)
            for i in range(len(_y)):
                _grad = torch.autograd.grad(_y[i], x, 
                                           grad_outputs=torch.ones_like(_y[i]), create_graph=True)[0]
                grad.append(_grad)
        return grad
    else:
        raise NotImplementedError

    return grad

# todo: remove internal mask
def face_split_gradients(vertices_etc, faces, lr, reg_sampler, betas):
    extra_data = {}
    device = vertices_etc.device
    g = torch.zeros_like(vertices_etc[:, :3], device=device)
    # get barycentric coordinates
    barycentric_coords = torch.zeros([reg_sampler.sampled_vertices.shape[0], 3], device=device)
    barycentric_coords[:, :2] = reg_sampler.coordinates / (reg_sampler.n_tag - 1).unsqueeze(
        -1).repeat_interleave(
        repeats=reg_sampler.n_tag * (reg_sampler.n_tag + 1) // 2, dim=0)
    barycentric_coords[:, 2] = 1 - barycentric_coords[:, 0] - barycentric_coords[:, 1]
    barycentric_coords = barycentric_coords.unsqueeze(-1)
    m2 = vertices_etc[:, 3]
    m1 = vertices_etc[:, 5:8]
    ref_len = vertices_etc[:, 8]
    m1_sampled = torch.sum(m1[faces[reg_sampler.face_ids]] * barycentric_coords, dim=-2)
    m2_sampled = torch.sum(m2[faces[reg_sampler.face_ids]].unsqueeze(-1) * barycentric_coords,
                           dim=-2)
    ref_len_sampled = torch.sum(
        ref_len[faces[reg_sampler.face_ids]].unsqueeze(-1) * barycentric_coords, dim=-2)
    m1_sampled = m1_sampled[reg_sampler.internal_sampled_vertices_mask]
    m2_sampled = m2_sampled[reg_sampler.internal_sampled_vertices_mask]
    ref_len_sampled = ref_len_sampled[reg_sampler.internal_sampled_vertices_mask]
    m1_sampled = m1_sampled * (betas[0]) + g[reg_sampler.internal_sampled_vertices_mask] * (
            1 - betas[0])
    m2_sampled = m2_sampled * (betas[1]) + torch.sum(
        g[reg_sampler.internal_sampled_vertices_mask] ** 2, dim=-1, keepdim=True) * (1 - betas[1])
    m1_sampled /= (1 - betas[0])
    m2_sampled /= (1 - betas[1])
    nu_sampled = m1_sampled / (torch.sqrt(m2_sampled) + 1e-20)
    extra_data['nu_sampled'] = nu_sampled
    extra_data['gradients'] = g
    extra_data['m1'] = m1
    extra_data['m1_sampled'] = m1_sampled
    extra_data['m2_sampled'] = m2_sampled
    extra_data['m2'] = m2
    extra_data['speeds'] = nu_sampled

    projected_points = torch.tensor(reg_sampler.sampled_vertices, device=device)
    projected_points[reg_sampler.internal_sampled_vertices_mask] = reg_sampler.sampled_vertices[
                                                                       reg_sampler.internal_sampled_vertices_mask] \
                                                                   - ref_len_sampled * nu_sampled * lr
    return projected_points, extra_data


def max_distance_score(projected_points, faces, reg_sampler):
    face_ids = reg_sampler.sampled_face_ids
    dists = torch.norm(projected_points - reg_sampler.sampled_vertices, dim=-1)
    out, argmax = scatter_max(dists, face_ids)
    face_scores = torch.zeros(faces.shape[0], device=projected_points.device)
    face_scores[argmax != 50000] = dists[argmax[argmax != 50000]]
    return face_scores


def random_face_score(faces):
    face_scores = torch.rand(faces.shape[0], device = faces.device)
    return face_scores


# def projection_along_vector_normals(s, t, source_normals=None, target_normals=None,
#                                     s2t_neighbours=5, mode='weighted_average'):

#     dists_s2t, idx_s2t, nn_s2t = knn_points(
#         s, t, K=s2t_neighbours, return_nn=True)
#     nn_normals = target_normals[idx_s2t][0]
#     if source_normals is not None:
#         projections, mask = project_points_on_plane_along_vector(s.unsqueeze(-2)[0].expand(-1, s2t_neighbours, -1),
#                                                                  nn_s2t[0], nn_normals,
#                                                                  source_normals.unsqueeze(-2).expand(-1, s2t_neighbours,
#                                                                                                      -1))
#         # amit_Q = torch.norm(
#         # s.unsqueeze(-2)[0].expand(-1, s2t_neighbours, -1) - projections - nn_s2t[0], dim=-1)
#         # amits_correction = amit_Q[None, ...]
#         amits_correction = dists_s2t
#     else:
#         projections, mask = project_points_on_plane_along_vector(s.unsqueeze(-2)[0].expand(-1, s2t_neighbours, -1),
#                                                                  nn_s2t[0], nn_normals,
#                                                                  None)
#     if mode == 'weighted_average':
#         weights = 1 / (amits_correction[0]+1e-5)
#         weights[~mask] = 0
#         weights /= torch.sum(weights, dim=-1, keepdim=True)
#         weights = torch.nan_to_num(weights, nan=0)
#         # average projections using (1/dists) as weights
#         projections = torch.sum(projections * weights.unsqueeze(-1), dim=-2)
#         projection_norms = torch.norm(projections, dim=-1)
#         dists_s2t = projection_norms
#     elif mode == 'nearest':
#         projection_norms = torch.norm(projections, dim=-1)
#         projection_norms[~mask] = np.inf
#         nn_to_select = torch.argmin(projection_norms, dim=-1)
#         ids = torch.arange(
#             projections.shape[0], device=s.device, dtype=torch.long)
#         projections = projections[ids, nn_to_select.flatten(), :]
#         dists_s2t = torch.gather(
#             projection_norms, dim=-1, index=nn_to_select.unsqueeze(-1))
#         dists_s2t = dists_s2t.flatten()
#     else:
#         raise NotImplementedError(
#             "invalid projection mode ('weighted_average' or 'nearest' currently supported)")

#     mask = torch.all(~mask, dim=-1)
#     dists_t2s, idx_t2s, _ = knn_points(t, s, K=1)
#     # scatter the maximal distance idx for each source index in idx_t2s
#     dists_t2s = dists_t2s.flatten()
#     idx_t2s = idx_t2s.flatten()
#     max_dists_t2s, max_idx_t2s = scatter_max(dists_t2s, idx_t2s)
#     if source_normals is not None and target_normals is not None:
#         cleaned_max_idx_t2s = torch.where(max_idx_t2s != t[0].shape[0])[0]
#         alignment_dot = torch.sum(
#             target_normals[max_idx_t2s[cleaned_max_idx_t2s]] * source_normals[cleaned_max_idx_t2s], dim=-1)
#         mask_t2s = cleaned_max_idx_t2s[alignment_dot < 0.95]
#         projections_t2s = torch.zeros_like(
#             s[0], device=s.device, dtype=torch.float)
#         # calculate projection distance between target to s
#         nn_t2s = s[0, cleaned_max_idx_t2s]
#         nn_t2s_normals = source_normals[cleaned_max_idx_t2s]
#         projections_t2s[cleaned_max_idx_t2s], _ = project_points_on_plane_along_vector(nn_t2s,
#                                                                                        t[0, max_idx_t2s[cleaned_max_idx_t2s]],
#                                                                                        target_normals[max_idx_t2s[cleaned_max_idx_t2s]],
#                                                                                        nn_t2s_normals)
#         projections_t2s[mask_t2s] = 0
#         max_dists_t2s = torch.norm(projections_t2s, dim=-1)
#         # take maximum over s2t and t2s
#         max_dists, max_dist_idx = torch.max(
#             torch.stack([max_dists_t2s[cleaned_max_idx_t2s], dists_s2t[cleaned_max_idx_t2s]], dim=-1), dim=-1)

#         projections[cleaned_max_idx_t2s] = projections[cleaned_max_idx_t2s] * (max_dist_idx.unsqueeze(-1)) + \
#             projections_t2s[cleaned_max_idx_t2s] * \
#             (1 - max_dist_idx.unsqueeze(-1))
    
#         mask[cleaned_max_idx_t2s] = mask[cleaned_max_idx_t2s] & (max_dist_idx==0) | \
#             ((alignment_dot < 0.95) & (max_dist_idx == 1))
#     return projections, mask, idx_s2t.flatten()

# def face_scores_face_folds(projected_points, faces, reg_sampler):
#     device = projected_points.device
#     sampled_faces = reg_sampler.sampled_faces
#     face_ids = reg_sampler.sampled_face_ids
#     vertices = reg_sampler.sampled_vertices
#     updated_vertex_normals = calculate_vertex_normals(
#         projected_points, sampled_faces)
#     fold_per_sampled_face_after = calculate_face_folds(projected_points, sampled_faces,
#                                                        vertex_normals=updated_vertex_normals)
#     _, areas_after = calculate_face_normals_and_areas(
#         projected_points, sampled_faces)
#     _, areas_before = calculate_face_normals_and_areas(vertices, sampled_faces)
#     fold_per_sampled_face_after[fold_per_sampled_face_after < 0.02] = 0
#     face_scores_after = torch.zeros(
#         faces.shape[0], device=device, dtype=torch.float)
#     fold_per_sampled_face_after = torch.nan_to_num(
#         fold_per_sampled_face_after, nan=0)
#     face_scores_after = torch.index_add(face_scores_after, dim=0, index=face_ids,
#                                         source=fold_per_sampled_face_after)
#     face_scores = face_scores_after
#     return face_scores

def project_to_triangle_soup(mesh_optimizer, vertices = None, faces=None, update_regular_sampling = False, return_debug_info = False):
    if vertices is None:
        vertices = mesh_optimizer.vertices
    if faces is None:
        faces = mesh_optimizer.faces
    rayCaster = mesh_optimizer.rayCaster
    reg_sampler = mesh_optimizer.reg_sampler
    if update_regular_sampling:
        mesh_optimizer.reg_sampler.sample_regular(vertices.unsqueeze(0), faces.unsqueeze(0))
    with torch.no_grad():
        debug_info = {}
        #vertex_normals = calculate_head_angle_weighted_vertex_normals(vertices, faces)
        __, projection_mask1, t1, projected_colors1, _ , prim_id1 = rayCaster.test_point_visibility(reg_sampler.sampled_vertices.detach()+reg_sampler.sampled_vertices_normals*1e-6, None, directions=-reg_sampler.sampled_vertices_normals)
        __, projection_mask2, t2, projected_colors2, _ , prim_id2 = rayCaster.test_point_visibility(reg_sampler.sampled_vertices.detach()-reg_sampler.sampled_vertices_normals*1e-6, None, directions=reg_sampler.sampled_vertices_normals)
        are_both_inside = torch.zeros(t1.shape[0], device = 'cuda', dtype = bool)
        if torch.any(t1!=np.inf):
            projections1 = reg_sampler.sampled_vertices_normals * (-t1.unsqueeze(-1))
            nudged_samples1 = reg_sampler.sampled_vertices.detach() + projections1*1e-1
            nudged_projections1 = nudged_samples1 + projections1  
            # todo: send all points at once
            wn1 = igl.fast_winding_number_for_meshes(vertices.detach().cpu().numpy(), faces.cpu().numpy(), nudged_samples1[t1!=np.inf].cpu().numpy())
            wn2 = igl.fast_winding_number_for_meshes(vertices.detach().cpu().numpy(), faces.cpu().numpy(), nudged_projections1[t1!=np.inf].cpu().numpy())
            wn1_self_intersection = (wn1>1.5) | (wn1<-0.5)
            wn1_inside = np.abs(wn1)>0.5
            wn1_outside = np.abs(wn1)<0.5
            wn2_inside = np.abs(wn2)>0.5
            wn2_outside = np.abs(wn2)<0.5
            are_both_inside[t1!=np.inf] = torch.tensor(np.abs(wn1)>0.5 , device='cuda', dtype=bool)
            mask = torch.ones(t1.shape[0], device = 'cuda', dtype = bool)
            mask[t1!=np.inf] = torch.tensor(~wn1_self_intersection & ((wn1_inside & wn2_inside) | (wn1_outside & wn2_outside)), device='cuda', dtype=bool)
            t1[~mask] = np.inf
            if return_debug_info:
                debug_info['projections1'] = torch.nan_to_num(projections1, posinf=0).cpu().numpy()
        if torch.any(t2!=np.inf):
            projections2 = reg_sampler.sampled_vertices_normals * (t2.unsqueeze(-1))
            nudged_samples2 = reg_sampler.sampled_vertices.detach() + projections2*1e-1
            nudged_projections2 = nudged_samples2 + projections2
            wn1 = igl.fast_winding_number_for_meshes(vertices.detach().cpu().numpy(), faces.cpu().numpy(), nudged_samples2[t2!=np.inf].cpu().numpy())
            wn2 = igl.fast_winding_number_for_meshes(vertices.detach().cpu().numpy(), faces.cpu().numpy(), nudged_projections2[t2!=np.inf].cpu().numpy())
            wn1_self_intersection = (wn1>1.5) | (wn1<-0.5)
            wn1_inside = np.abs(wn1)>0.5
            wn1_outside = np.abs(wn1)<0.5
            wn2_inside = np.abs(wn2)>0.5
            wn2_outside = np.abs(wn2)<0.5
            are_both_inside[t2 == np.inf] = False
            are_both_inside[t2!=np.inf] &= torch.tensor(np.abs(wn1)>0.5 , device='cuda', dtype=bool)
            mask = torch.ones(t2.shape[0], device = 'cuda', dtype = bool)
            mask[t2!=np.inf] = torch.tensor(~wn1_self_intersection & ((wn1_inside & wn2_inside) | (wn1_outside & wn2_outside)), device='cuda', dtype=bool)
            t2[~mask] = np.inf 
            if return_debug_info:
                debug_info['projections2'] = torch.nan_to_num(projections2, posinf=0).cpu().numpy()
        min_t =  torch.where(t1<t2, -t1, t2)
        projected_colors = projected_colors1 * (t1<t2) +  projected_colors2 * (t1>=t2)
        projection_mask = ~(projection_mask1 & projection_mask2)
        projection_mask[torch.abs(min_t) > 5e-2] = False
        projection_mask[are_both_inside] = False
        vertex_normals = calculate_vertex_normals(vertices, faces)
        face_normals = calc_face_normals(vertices, faces, normalize=True)
        edges, fe = calc_edges(faces, with_dummies = False)
        _, face_collapses = calc_face_collapses(None, faces, edges, fe, face_normals, vertex_normals )
        face_mask = torch.ones(faces.shape[0], device = 'cuda', dtype = torch.bool)
        face_mask[face_collapses] = False
        #projection_mask &= face_mask[reg_sampler.face_ids]
        projections = reg_sampler.sampled_vertices_normals * min_t.unsqueeze(-1)*projection_mask.unsqueeze(-1)
        projections = torch.nan_to_num(projections, posinf=0)
        sampled_face_normals = torch.zeros_like(reg_sampler.sampled_vertices, dtype=torch.float, device = faces.device)
        prim_id =  torch.where(t1<t2, prim_id1, prim_id2).long()#[projection_mask]
        prim_id[~projection_mask] = -1
        sampled_face_normals[projection_mask] = rayCaster.target_face_normals[prim_id[projection_mask]]
        sampled_face_normals = sampled_face_normals.to('cuda')
        normal_mask = torch.sum(sampled_face_normals * projections/(torch.norm(projections, dim=-1, keepdim=True)+1e-8), dim=-1)
        normal_mask = torch.abs(normal_mask) > 0.7
        projected_points = reg_sampler.sampled_vertices.detach() + projections * normal_mask.unsqueeze(-1)
        if return_debug_info:
            debug_info['projection_mask'] = projection_mask.cpu().numpy()
            debug_info['are_both_inside'] = are_both_inside.cpu().numpy()
            debug_info['sampled_face_normals'] = sampled_face_normals.cpu().numpy()
            debug_info['projected_colors'] = projected_colors.cpu().numpy() 
            return projected_points, projection_mask, debug_info
        return projected_points, projection_mask, projected_colors, prim_id

def calculate_double_face_areas(fv):
    fv0 = fv[:, 0]
    fv1 = fv[:, 1]
    fv2 = fv[:, 2]
    double_face_areas = torch.norm(torch.cross(fv1 - fv0, fv2 - fv0, dim=-1), dim=-1, keepdim=True)
    return double_face_areas

def calc_for_face_vertices_areas(fv, eps=1e-8):
    fv0 = fv[:, 0]
    fv1 = fv[:, 1]
    fv2 = fv[:, 2]
    face_normals = torch.cross(fv1 - fv0, fv2 - fv0, dim=-1)
    normal_mask = torch.norm(face_normals, dim=-1) < eps
    face_normals[~normal_mask] /= torch.norm(face_normals[~normal_mask], dim=-1, keepdim=True)
    face_normals[normal_mask] = 0
    return face_normals

def calc_for_face_vertices_normals(fv, eps=1e-8):
    fv0 = fv[:, 0]
    fv1 = fv[:, 1]
    fv2 = fv[:, 2]
    face_normals = torch.cross(fv1 - fv0, fv2 - fv0, dim=-1)
    normal_mask = torch.norm(face_normals, dim=-1) < eps
    face_normals[~normal_mask] /= torch.norm(face_normals[~normal_mask], dim=-1, keepdim=True)
    face_normals[normal_mask] = 0
    return face_normals

def face_scores_face_folds(projected_points, faces, reg_sampler, face_normals, projected_points_mask=None, curvature_threshold=0.02, mode = 'face split'):
    device = projected_points.device

    # todo: multiply score by face area

    # faces = reg_sampler.f[0]
    # vertices, sampled_faces, face_ids, _, _ = reg_sampler.get_sampled_points_per_face()
    sampled_faces = reg_sampler.sampled_faces
    face_ids = reg_sampler.sampled_face_ids
    if projected_points_mask is None:
        sampled_face_mask = None
    else:
        sampled_face_mask = ~(torch.any(projected_points_mask[sampled_faces],dim=-1)) #get_maximal_face_value_over_vertices(sampled_faces, 1 - projected_points_mask.int())
        sampled_face_mask = sampled_face_mask.bool()
    updated_vertex_normals = calculate_vertex_normals(projected_points, sampled_faces[sampled_face_mask])
    if projected_points_mask is not None:
        updated_vertex_normals[projected_points_mask] = 0
    fold_per_sampled_face_after = calculate_face_folds(projected_points, sampled_faces,
                                                       vertex_normals=updated_vertex_normals)
    fold_per_sampled_face_after = torch.nan_to_num(fold_per_sampled_face_after, nan=0)
    #sampled_face_mask[fold_per_sampled_face_after > 1] = False
    # flipped_face_mask = fold_per_sampled_face_after > 1
    # flipped_vertex_mask = calculate_vertex_incident_scalar(projected_points.shape[0], sampled_faces,
    #                                                 flipped_face_mask.float()) > 0
    # flipped_vertex_mask = flipped_vertex_mask.squeeze(-1)
    # flipped_face_mask = get_maximal_face_value_over_vertices(sampled_faces, flipped_vertex_mask) > 0
    # sampled_face_mask &= ~flipped_face_mask
    fold_per_sampled_face_after[fold_per_sampled_face_after > 1] = 0
    fold_per_sampled_face_after[~sampled_face_mask] = 0
    # check if normals are flipped
    face_normals_before = face_normals[face_ids]
    face_normals_after = calc_for_face_vertices_normals(projected_points[reg_sampler.sampled_faces])
    flipped_normals = torch.sum(face_normals_before * face_normals_after, dim=-1) < 0
    fold_per_sampled_face_after[flipped_normals] = 0
    # calculcate area of new triangles
    # face_areas_after = calculate_double_face_areas(projected_points[reg_sampler.sampled_faces]).flatten().sqrt()
    # face_areas_after[fold_per_sampled_face_after == 0] = 0
    # fold_per_sampled_face_after *= face_areas_after
    #fold_per_sampled_face_after /= face_areas_after.sum()

    # if projected_points_mask is not None:
    #     if mode == 'face split':
    #         projection_vec = (projected_points - reg_sampler.sampled_vertices)
    #         projection_vec /= torch.norm(projection_vec, dim=-1, keepdim=True)
    #         factor = torch.sum(reg_sampler.sampled_vertices_normals*projection_vec,dim=-1).abs()
    #         fold_per_sampled_face_after *= torch.sum(factor[sampled_faces],dim=-1)/3
            #fold_per_sampled_face_after[~sampled_face_mask] = 0 
    #     elif mode == 'edge flip':
    #         fold_per_sampled_face_after[~sampled_face_mask] = 1

    # todo: get triangle vertices and get mask 

    fold_per_sampled_face_after[fold_per_sampled_face_after < curvature_threshold] = 0
    num_of_non_zero_per_face = torch.zeros(faces.shape[0], device=device, dtype=torch.float)
    face_scores_after = torch.zeros(faces.shape[0], device=device, dtype=torch.float)
    fold_per_sampled_face_after = torch.nan_to_num(fold_per_sampled_face_after, nan=0)
    face_scores_after = torch.index_add(face_scores_after, dim=0, index=face_ids,
                                        source=fold_per_sampled_face_after)
    
    # calculate the correction for the edge flip
    # all uncertain faces are scored as 1 (instead of 0 for face split)
    # in other words, for face split, uncertainty means *dont split* and for edge flip, uncertainty means flip
    edge_flip_correction_per_face = torch.zeros_like(fold_per_sampled_face_after, device = fold_per_sampled_face_after.device)
    edge_flip_correction_per_face[~sampled_face_mask] = 1
    edge_flip_correction = torch.zeros(faces.shape[0], device=device, dtype=torch.float)
    edge_flip_correction = torch.index_add(edge_flip_correction, dim=0, index=face_ids,
                                        source=edge_flip_correction_per_face)
    edge_flip_correction /= reg_sampler.amount_of_vertices_for_each_face
    #fold_per_sampled_face_after *= face_areas_after
    # num_of_non_zero_per_face = torch.index_add(num_of_non_zero_per_face, dim=0, index=face_ids,
    #                                            source=(fold_per_sampled_face_after >= curvature_threshold).float())
    
    return face_scores_after, edge_flip_correction, fold_per_sampled_face_after,face_normals_after, updated_vertex_normals, sampled_face_mask, edge_flip_correction_per_face


@torch.no_grad()
def point_projections_using_sdf(points, sdf):
    grads = gradient(points, sdf, method='finitediff')
    dists = sdf(points)
    projection = -dists * grads / (torch.norm(grads, dim=-1, keepdim=True)+ 1e-5)
    return projection

@torch.no_grad()
def face_split_sdf_sphere_tracing(all_vertices, vertex_normals, sdf, iterations = 5, return_debug_info = False):
    with torch.no_grad():
        projected_points = all_vertices.clone()
        for _ in range(iterations):
            projection = point_projections_using_sdf(projected_points, sdf)
            projection = vertex_normals*torch.sum(projection * vertex_normals, dim=-1, keepdim=True)
            projected_points = projection + projected_points
    projected_sdf_vals = sdf(projected_points)
    original_sdf_vals = sdf(all_vertices)
    projection_mask = torch.abs(projected_sdf_vals.flatten())>torch.abs(original_sdf_vals.flatten())
    projection_mask |= torch.abs(projected_sdf_vals.flatten())>1e-6
    return projected_points, projection_mask, original_sdf_vals, projected_sdf_vals