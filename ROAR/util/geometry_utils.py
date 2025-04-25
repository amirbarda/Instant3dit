import torch
try:
    import torch_scatter
except:
    print ('Warning: torch_scatter not found!')
import numpy as np
from ROAR.util.topology_utils import normalized_laplacian, calc_edges, orient_edges
import torch.nn.functional as tfunc
import igl


def double_face_areas(vs, faces):
    """

    :param vs:
    :param faces:
    :return: 2*face_areas
    """
    v0 = vs[:, faces[0, :]][:, :, 0]
    v1 = vs[:, faces[0, :]][:, :, 1]
    v2 = vs[:, faces[0, :]][:, :, 2]

    return torch.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1, keepdim=True)


class ZeroNanGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        grad[grad != grad] = 0
        return grad


def calc_edge_length(
        vertices: torch.Tensor,  # V,3 first may be dummy
        edges: torch.Tensor,  # E,2 long, lower vertex index first, (0,0) for unused
) -> torch.Tensor:  # E

    full_vertices = vertices[edges]  # E,2,3
    a, b = full_vertices.unbind(dim=1)  # E,3
    return torch.norm(a - b, p=2, dim=-1)


def calc_face_normals(
        vertices: torch.Tensor,  # V,3 first vertex may be unreferenced
        faces: torch.Tensor,  # F,3 long, first face may be all zero   
        normalize: bool = False,
) -> torch.Tensor:  # F,3
    """
         n
         |
         c0     corners ordered counterclockwise when
        / \     looking onto surface (in neg normal direction)
      c1---c2
    """
    full_vertices = vertices[faces]  # F,C=3,3
    v0, v1, v2 = full_vertices.unbind(dim=1)  # F,3
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # F,3
    if normalize:
        face_normals = tfunc.normalize(face_normals, eps=1e-6, dim=1)  # TODO inplace?
    return face_normals  # F,3


# def calc_vertex_normals(
#         vertices: torch.Tensor,  # V,3 first vertex may be unreferenced
#         faces: torch.Tensor,  # F,3 long, first face may be all zero
#         face_normals: torch.Tensor = None,  # F,3, not normalized
# ) -> torch.Tensor:  # F,3
#
#     F = faces.shape[0]
#
#     if face_normals is None:
#         face_normals = calc_face_normals(vertices, faces)
#
#     vertex_normals = torch.zeros((vertices.shape[0], 3, 3), dtype=vertices.dtype, device=vertices.device)  # V,C=3,3
#     vertex_normals.scatter_add_(dim=0, index=faces[:, :, None].expand(F, 3, 3),
#                                 src=face_normals[:, None, :].expand(F, 3, 3))
#     vertex_normals = vertex_normals.sum(dim=1)  # V,3
#     return tfunc.normalize(vertex_normals, eps=1e-6, dim=1)


def calc_face_ref_normals(
        faces: torch.Tensor,  # F,3 long, 0 for unused
        vertex_normals: torch.Tensor,  # V,3 first unused
        normalize: bool = False,
) -> torch.Tensor:  # F,3
    """calculate reference normals for face flip detection"""
    full_normals = vertex_normals[faces]  # F,C=3,3
    ref_normals = full_normals.sum(dim=1)  # F,3
    if normalize:
        ref_normals = tfunc.normalize(ref_normals, eps=1e-12, dim=1)
    return ref_normals


def calculate_head_angles_per_vertex_per_face(v, f):
    # for all faces, calculate the head angles for each vertex
    fv0 = v[f[:, 0]]
    fv1 = v[f[:, 1]]
    fv2 = v[f[:, 2]]
    cos0 = torch.sum(tfunc.normalize(fv1 - fv0, dim=1) * tfunc.normalize(fv2 - fv0, dim=1), dim=-1, keepdim=True)
    cos0 = torch.clip(cos0, max=1, min=-1)
    cos1 = torch.sum(tfunc.normalize(fv2 - fv1, dim=1) * tfunc.normalize(fv0 - fv1, dim=1), dim=-1, keepdim=True)
    cos1 = torch.clip(cos1, max=1, min=-1)
    cos2 = torch.sum(tfunc.normalize(fv0 - fv2, dim=1) * tfunc.normalize(fv1 - fv2, dim=1), dim=-1, keepdim=True)
    cos2 = torch.clip(cos2, max=1, min=-1)
    head_angle0 = torch.acos(cos0)
    head_angle1 = torch.acos(cos1)
    head_angle2 = torch.acos(cos2)
    head_angle = torch.cat([head_angle0, head_angle1, head_angle2],dim=-1)
    head_angle = head_angle.reshape([-1,1])
    face_ids = torch.arange(f.shape[0], device=v.device).repeat(3)
    vertex_ids = f.T.flatten()
    return head_angle, vertex_ids, face_ids

def calculate_face_folds(v, f, vertex_normals=None):
    if vertex_normals is None:
        vertex_normals = calculate_vertex_normals(v, f)
    face_normals, _ = calculate_face_normals_and_areas(v, f)
    face_normals_repeated = face_normals.unsqueeze(1).repeat_interleave(repeats=3, dim=-2)
    face_vertex_normals = vertex_normals[f]
    mean_of_dot_prods = torch.mean(torch.sum(face_normals_repeated * face_vertex_normals, dim=-2), dim=-1)
    return 1 - mean_of_dot_prods


def calculate_edge_normals(v, f):
    angles, fn0, fn1, edge_vec = calculate_dihedral_angles(v, f, return_normals_and_orientation=True)
    edge_normals = rotate_vectors_by_angles_around_axis(fn0, angles / 2, edge_vec)
    return edge_normals


def get_barycentric_coordinates(p, vertices_for_faces, normalize=True):
    """
    :param vs: points to calculate barycentric coordinate for
    :param faces: target mesh faces for each point
    :param face_points: face vertices for each face
    :return:
    """
    v0 = vertices_for_faces[:, 0]
    v1 = vertices_for_faces[:, 1]
    v2 = vertices_for_faces[:, 2]
    # division by 2 not required, only ratio is used
    A0 = torch.norm(torch.cross((v1 - v0), (p - v0), dim=-1), dim=-1, keepdim=True)
    A1 = torch.norm(torch.cross((v2 - v1), (p - v1), dim=-1), dim=-1, keepdim=True)
    A2 = torch.norm(torch.cross((v0 - v2), (p - v2), dim=-1), dim=-1, keepdim=True)
    barycentric_coords = torch.cat([A0, A1, A2], dim=-1)
    if normalize:
        A = torch.norm(torch.cross((v1 - v0), (v2 - v0), dim=-1), dim=-1, keepdim=True)
        barycentric_coords /= A
    return barycentric_coords

def outward_facing_normals_for_face(vertices_for_faces):
    v0 = vertices_for_faces[:, 0]
    v1 = vertices_for_faces[:, 1]
    v2 = vertices_for_faces[:, 2]
    e0 = tfunc.normalize(v1 - v0)
    e1 = tfunc.normalize(v2 - v1)
    e2 = tfunc.normalize(v0 - v2)
    angle0 = 2*np.pi - torch.acos(torch.sum(e0*(-e2), dim=-1, keepdim=True).clip(-1,1))/2
    angle1 = 2*np.pi - torch.acos(torch.sum(e1*(-e0), dim=-1, keepdim=True).clip(-1,1))/2
    angle2 = 2*np.pi - torch.acos(torch.sum(e2*(-e1), dim=-1, keepdim=True).clip(-1,1))/2
    face_normals = torch.cross(e0, e1, dim=-1)
    outward_vector0 = rotate_vectors_by_angles_around_axis(e2, angle0, face_normals)
    outward_vector1 = rotate_vectors_by_angles_around_axis(e0, angle1, face_normals)
    outward_vector2 = rotate_vectors_by_angles_around_axis(e1, angle2, face_normals)
    return torch.cat([outward_vector0, outward_vector1, outward_vector2], dim=0)

def are_points_in_faces(p, vertices_for_faces, scale_factor=1.0):
    # barycentric_coords = get_barycentric_coordinates(p, vertices_for_faces, normalize=True)
    # return torch.sum(barycentric_coords, dim=-1) <= 1.01
    v0 = vertices_for_faces[:, 0]
    v1 = vertices_for_faces[:, 1]
    v2 = vertices_for_faces[:, 2]
    e0 = tfunc.normalize(v1 - v0)
    e1 = tfunc.normalize(v2 - v1)
    e2 = tfunc.normalize(v0 - v2)
    face_normals = torch.cross(e0, e1, dim=-1)
    turn0 = torch.sum(torch.cross(tfunc.normalize(p - v0), e0, dim=-1) * face_normals, dim=-1)
    turn1 = torch.sum(torch.cross(tfunc.normalize(p - v1), e1, dim=-1) * face_normals, dim=-1)
    turn2 = torch.sum(torch.cross(tfunc.normalize(p - v2), e2, dim=-1) * face_normals, dim=-1)
    return (turn0 <= 0) & (turn1 <= 0) & (turn2 <= 0)


def calculate_face_areas(v, f, eps=1e-10):
    """
    :param v: vertex tensor
    :param f: face tensor
    :param eps: area thresold for degenerate face
    :param default_normal: normal to assign in case of degenerate face. None assigns vector [1,0,0]
    :return: tensor of face normals
    """

    face_vertices = v[f]
    v0 = face_vertices[:, 0, :]
    v1 = face_vertices[:, 1, :]
    v2 = face_vertices[:, 2, :]
    cross = torch.cross((v1 - v0), (v2 - v0), dim=1)
    norms = torch.norm(cross, dim=1, keepdim=True)
    face_areas = norms / 2

    # add eps to avoid division by zero
    face_areas[face_areas < eps] = eps

    return face_areas


def calculate_face_normals_and_areas(v, f, eps=1e-9, default_normal=None):
    """
    :param v: vertex tensor
    :param f: face tensor
    :param eps: area thresold for degenrate face
    :param default_normal: normal to assign in case of degenerate face. None assigns vector [1,0,0]
    :return: tensor of face normals
    """
    device = v.device
    if default_normal is None:
        default_normal = torch.tensor([1, 0, 0], dtype=torch.float, device=device)
    batch_size = f.shape[0]
    batch_list = torch.arange(batch_size)
    face_normals = torch.zeros_like(f, dtype=torch.float, device=device)
    face_vertices = v[f]
    v0 = face_vertices[:, 0, :]
    v1 = face_vertices[:, 1, :]
    v2 = face_vertices[:, 2, :]
    cross = torch.cross((v1 - v0), (v2 - v0), dim=1)
    norms = torch.norm(cross, dim=1, keepdim=True)
    face_areas = norms / 2
    # add eps to avoid division by zero
    face_areas[face_areas < eps] = eps
    face_normals[norms.squeeze(-1) > eps] = (cross / norms)[norms.squeeze(-1) > eps]
    face_normals[norms.squeeze(-1) <= eps] = default_normal

    return face_normals, face_areas


def calculate_vertex_normals(v, f, normalize=True, mask=None):
    device = v.device
    face_normals, face_areas = calculate_face_normals_and_areas(v, f)
    head_angle, vertex_ids, face_ids = calculate_head_angles_per_vertex_per_face(v,f)
    if mask is not None:
        face_normals[mask] = 0
        face_areas[mask] = 0
    # todo: unroll f
    f_unrolled = f.flatten()
    f_unrolled = torch.stack([f_unrolled*3, f_unrolled*3+1, f_unrolled*3+2],dim=-1).flatten()
    # create face_index vertex
    face_indices_repeated_per_vertex = torch.arange(f.shape[0], device=v.device)
    face_indices_repeated_per_vertex = torch.repeat_interleave(face_indices_repeated_per_vertex, repeats=3)
    normals_repeated_per_face = face_normals[face_indices_repeated_per_vertex]
    face_areas_repeated_per_face = face_areas[face_indices_repeated_per_vertex]
    face_angles_repeated_per_face = head_angle#face_areas[face_indices_repeated_per_vertex]
    vertex_normals = torch.zeros_like(v.flatten(), device=device)
    # vertex_areas = torch.zeros([v.shape[0], 1], device=device)
    # vertex_normals = torch.index_add(vertex_normals, dim=0, index=f_unrolled,
    #                                  source=normals_repeated_per_face * face_areas_repeated_per_face * face_angles_repeated_per_face)
    source= (normals_repeated_per_face * face_areas_repeated_per_face * face_angles_repeated_per_face).flatten()
    vertex_normals.put_(f_unrolled, source, accumulate = True)
    vertex_normals = vertex_normals.reshape([-1,3])
    #vertex_areas = torch.index_add(vertex_areas, dim=0, index=f_unrolled, source=face_areas_repeated_per_face * face_angles_repeated_per_face)
    #vertex_normals /= vertex_areas
    if normalize:
        vertex_normals = tfunc.normalize(vertex_normals, dim=1)
    return vertex_normals


def project_points_on_plane_along_vector(v, p, n, vn=None, direction_threshold=0.01, distance_percent_limit=0.01):
    # direction threshold was 0.01
    """

    :param v: source vertices (to project)
    :param p: projected (target) points
    :param n: target normals
    :param vn: vertex normals
    :return: projection vectors along normal vector
    """

    projection = torch.zeros_like(v, device=v.device, dtype=torch.float)
    mask = torch.ones(v.shape[0], device=v.device, dtype=torch.bool)
    if vn is None:
        weight = -torch.sum(n * ((p - v) / torch.norm(p - v, dim=-1, keepdim=True)),
                            dim=-1)
        mask = weight > direction_threshold  # mask = True -> valid projection
        projection[mask] = torch.sum((p[mask] - v[mask]) * n[mask], dim=-1, keepdim=True) * n[mask] * weight.unsqueeze(-1)[
            mask]
    else:
        weight = torch.sum(vn * n, dim=-1)
        mask = weight > direction_threshold
        projection[mask] = torch.sum((p[mask] - v[mask]) * n[mask], dim=-1, keepdim=True) / \
                           torch.sum(n[mask] * vn[mask], dim=-1, keepdim=True) * vn[mask]

    # get distance between source and target to avoid blow-up of projection term
    distances_from_target = torch.norm(p - v, dim=-1)
    distance_from_projected_point = torch.norm(p - (v + projection), dim=-1)
    mask = mask & (distances_from_target * (1 + distance_percent_limit) > distance_from_projected_point)
    projection[~mask] = 0
    return projection, mask


def calculate_dihedral_angles(v, f, return_normals_and_orientation=False):
    edges, fe, ef = calc_edges(f, with_edge_to_face=True, with_dummies=False)
    oriented_edges = orient_edges(edges, f[ef[:, 0, 0]])
    face_normals = calc_face_normals(v, f, normalize=True)
    fn0 = face_normals[ef[:, 0, 0]]
    fn1 = face_normals[ef[:, 1, 0]]
    edge_vec = tfunc.normalize(v[oriented_edges[:, 1]] - v[oriented_edges[:, 0]])
    # todo: calculate angle to rotate first normal to second normal through oriented edge
    dot = torch.sum(fn0 * fn1, dim=-1, keepdim=True)
    det = torch.sum(edge_vec * torch.cross(fn0, fn1, dim=-1), dim=-1, keepdim=True)
    angles = torch.atan2(det, dot)  # from -180 to 180
    # angles[angles < 0] = angles[angles < 0] + 2 * np.pi
    if return_normals_and_orientation:
        return angles, fn0, fn1, edge_vec
    else:
        return torch.abs(angles)


def calculate_edge_normals(v, f):
    angles, fn0, fn1, edge_vec = calculate_dihedral_angles(v, f, return_normals_and_orientation=True)
    edge_normals = rotate_vectors_by_angles_around_axis(fn0, angles / 2, edge_vec)
    return edge_normals


def calculate_delta_vectors(v, f):
    edges, _ = calc_edges(f)  # E,2
    E = edges.shape[0]
    neighbor_smooth = torch.zeros_like(v)  # V,S - mean of 1-ring vertices
    e_v = v[edges]
    torch_scatter.scatter_mean(src=e_v.flip(dims=[1]).reshape(E * 2, -1), index=edges.reshape(E * 2, 1),
                               dim=0, out=neighbor_smooth)
    # add laplace smoothing to gradients
    laplace = v - neighbor_smooth[:, :3]  # laplace vector
    return laplace


def calculate_generalize_vertex_normals(v, f):
    nonnormalized_vertex_normals = calculate_vertex_normals(v, f, normalize=False)
    norm = torch.norm(nonnormalized_vertex_normals, dim=-1, keepdim=True)
    delta_vectors = calculate_delta_vectors(v, f)
    return tfunc.normalize(nonnormalized_vertex_normals * norm ** 2 + delta_vectors * (1 - norm) ** 2, eps=1e-6)


def rotation_angle_between_two_vectors_and_axis(source_vectors, target_vectors, axes):
    """
    batched operation
    :param vectors: #Vx3 float tensor
    :param angles: #Vx1 float tensor
    :param axes: #Vx3 float tensor
    :return: vectors rotated by angles around axes
    """
    dot = torch.sum(source_vectors * target_vectors, dim=-1, keepdim=True)
    det = torch.sum(axes * (torch.cross(source_vectors, target_vectors, dim=-1)), dim=-1, keepdim=True)
    return torch.atan2(det, dot)


def rotate_vectors_by_angles_around_axis(vectors, angles, axes):
    """
    batched operation, uses the Rodriguez formula
    :param vectors:
    :param angles: in radians
    :param axes:
    :return:
    """
    K2v = torch.cross(axes, torch.cross(axes, vectors, dim=-1), dim=-1)
    Kv = torch.cross(axes, vectors, dim=-1)
    return vectors + torch.sin(angles) * Kv + (1 - torch.cos(angles)) * K2v


def calculate_laplacian_vectors(vertices, faces):
    L = normalized_laplacian(vertices.shape[0], faces)
    return torch._sparse_mm(L, vertices)


def project_pc_to_mesh(points, v, f):
    """
    wrapper for igl.point_mesh_squared_distance for torch
    :param points: #Px3 point cloud points tensor
    :param v: #Vx3 mesh vertices tensor
    :param f: #Fx3 mesh faces tensor
    :return:
    """
    device = v.device
    squared_distances, parent_faces, projected_points = igl.point_mesh_squared_distance(points.cpu().numpy(),
                                                                                        v.cpu().numpy(),
                                                                                        f.cpu().numpy())
    squared_distances = torch.tensor(squared_distances, device=device)
    parent_faces = torch.tensor(parent_faces, device=device, dtype=torch.long)
    projected_points = torch.tensor(projected_points, device=device)
    return squared_distances, parent_faces, projected_points


def get_perp_vectors(vecs, eps=1e-5):
    vec1 = torch.tensor([[1, 0, 0]], device=vecs.device, dtype=torch.float).repeat(vecs.shape[0], 1)
    perp_vectors = torch.cross(vecs, vec1, dim=-1)
    vec2 = torch.tensor([[0, 1, 0]], device=vecs.device, dtype=torch.float).repeat(vecs.shape[0], 1)
    vec2 = vec2[torch.norm(perp_vectors, dim=-1) < eps]
    perp_vectors[torch.norm(perp_vectors, dim=-1) < eps] = torch.cross(
        vecs[torch.norm(perp_vectors, dim=-1) < eps], vec2, dim=-1)
    perp_vectors /= torch.norm(perp_vectors, dim=-1, keepdim=True)
    return perp_vectors


def distance_between_points_and_lines(s0, dirs, points):
    t = torch.sum((points - s0) * dirs, dim=-1, keepdim=True) / torch.sum(dirs * dirs, dim=-1, keepdim=True)
    return torch.norm(points - (s0 + dirs * t), dim=-1)
