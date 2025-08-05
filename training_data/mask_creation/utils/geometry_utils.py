import torch
import torch.nn.functional as tfunc

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
    f_unrolled = f.flatten()
    f_unrolled = torch.stack([f_unrolled*3, f_unrolled*3+1, f_unrolled*3+2],dim=-1).flatten()
    face_indices_repeated_per_vertex = torch.arange(f.shape[0], device=v.device)
    face_indices_repeated_per_vertex = torch.repeat_interleave(face_indices_repeated_per_vertex, repeats=3)
    normals_repeated_per_face = face_normals[face_indices_repeated_per_vertex]
    face_areas_repeated_per_face = face_areas[face_indices_repeated_per_vertex]
    face_angles_repeated_per_face = head_angle#face_areas[face_indices_repeated_per_vertex]
    vertex_normals = torch.zeros_like(v.flatten(), device=device)
    source= (normals_repeated_per_face * face_areas_repeated_per_face * face_angles_repeated_per_face).flatten()
    vertex_normals.put_(f_unrolled, source, accumulate = True)
    vertex_normals = vertex_normals.reshape([-1,3])
    if normalize:
        vertex_normals = tfunc.normalize(vertex_normals, dim=1)
    return vertex_normals
