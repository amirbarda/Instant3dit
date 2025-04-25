import torch
import numpy as np
try:
    import torch_scatter
except:
    print ('Warning: torch_scatter not found!')


# TODO: does this support borders?
def calc_edges(
        faces: torch.Tensor,  # F,3 long - first face may be dummy with all zeros
        with_edge_to_face: bool = False,
        with_dummies=True
):
    """
    returns tuple of
    - edges E,2 long, 0 for unused, lower vertex index first
    - face_to_edge F,3 long
    - (optional) edge_to_face shape=E,[left,right],[face,side]

    o-<-----e1     e0,e1...edge, e0<e1
    |      /A      L,R....left and right face
    |  L /  |      both triangles ordered counter clockwise
    |  / R  |      normals pointing out of screen
    V/      |
    e0---->-o
    """

    F = faces.shape[0]

    # make full edges, lower vertex index first
    face_edges = torch.stack((faces, faces.roll(-1, 1)), dim=-1)  # F*3,3,2
    full_edges = face_edges.reshape(F * 3, 2)
    sorted_edges, _ = full_edges.sort(dim=-1)  # F*3,2 TODO min/max faster?

    # make unique edges
    edges, full_to_unique = torch.unique(input=sorted_edges, sorted=True, return_inverse=True, dim=0)  # (E,2),(F*3)
    E = edges.shape[0]
    face_to_edge = full_to_unique.reshape(F, 3)  # F,3

    if not with_edge_to_face:
        return edges, face_to_edge

    is_right = full_edges[:, 0] != sorted_edges[:, 0]  # F*3
    edge_to_face = torch.zeros((E, 2, 2), dtype=torch.long, device=faces.device)  # E,LR=2,S=2
    scatter_src = torch.cartesian_prod(torch.arange(0, F, device=faces.device),
                                       torch.arange(0, 3, device=faces.device))  # F*3,2
    edge_to_face.reshape(2 * E, 2).scatter_(dim=0, index=(2 * full_to_unique + is_right)[:, None].expand(F * 3, 2),
                                            src=scatter_src)  # E,LR=2,S=2
    if with_dummies:
        edge_to_face[0] = 0
    return edges, face_to_edge, edge_to_face  # =EF


def consolidate_indices_np(f):
    """
    close gaps in index tensor
    e.g:
    [0,1,2,5,4] -> [0,1,2,4,3]
    """
    max_ele = np.max(f)
    full_indices = np.ones(max_ele + 1, dtype=np.int64) * (-1)
    full_indices[np.unique(f)] = np.arange(np.unique(f).shape[0], dtype=np.int64)

    return full_indices[f]


def consolidate_indices(f):
    """
    close gaps in index tensor
    e.g:
    [0,1,2,5,4] -> [0,1,2,4,3]
    """
    max_ele = torch.max(f)
    full_indices = torch.ones(max_ele + 1, device=f.device, dtype=torch.long) * (-1)
    full_indices[torch.unique(f)] = torch.arange(torch.unique(f).shape[0], device=f.device, dtype=torch.long)

    return full_indices[f]


# # for pytorch geometric models
# def calculate_edge_index(f):
#     edges = calculate_e(f)
#     edges_index = torch.repeat_interleave(edges, repeats=2, dim=0)
#     edges_index[::2, 0] = edges_index[1::2, 1]
#     edges_index[::2, 1] = edges_index[1::2, 0]
#
#     return edges_index.T


def calculate_ve_mat(edges):
    v0 = edges[:, 0]
    v1 = edges[:, 1]
    edge_id = torch.arange(edges.shape[0], device=edges.device, dtype=torch.long)
    indices1 = torch.stack([v0, edge_id], dim=0)
    indices2 = torch.stack([v1, edge_id], dim=0)
    indices = torch.concat([indices1, indices2], dim=-1)
    vals = torch.ones(indices.shape[-1], device=edges.device, dtype=torch.float)
    return torch.sparse_coo_tensor(indices, vals, device=edges.device)


def calculate_adj_matrix(f, normalize_rows=None):
    edges, _ = calc_edges(f)
    v0 = edges[:, 0]
    v1 = edges[:, 1]
    rows = torch.cat([v0, v1])
    cols = torch.cat([v1, v0])
    indices = torch.stack([rows, cols])
    vals = torch.ones_like(rows, device=f.device, dtype=torch.float)
    if normalize_rows is not None:
        vals /= torch.cat([normalize_rows[v0], normalize_rows[v1]]).flatten()
    adj_mat = torch.sparse_coo_tensor(indices, vals, device=f.device)
    return adj_mat


def calculate_vertex_incident_scalar(num_vertices, f, per_face_scalar_field, avg=False):
    """
    computes a scalar field per vertex by summing / averaging the incident face scalar field
    """
    device = f.device
    incident_face_areas = torch.zeros([num_vertices, 1], device=device)
    f_unrolled = f.flatten()
    face_indices_repeated_per_vertex = torch.arange(f.shape[0], device=f.device)
    face_indices_repeated_per_vertex = torch.repeat_interleave(face_indices_repeated_per_vertex, repeats=3)
    face_areas_repeated_per_face = per_face_scalar_field[face_indices_repeated_per_vertex].unsqueeze(-1)
    incident_face_areas = torch.index_add(incident_face_areas, dim=0, index=f_unrolled,
                                          source=face_areas_repeated_per_face)
    if avg:
        neighbors = torch.index_add(torch.zeros_like(incident_face_areas), dim=0, index=f_unrolled,
                                    source=torch.ones_like(face_areas_repeated_per_face))
        incident_face_areas = incident_face_areas / neighbors

    return incident_face_areas


def calculate_vertex_incident_vector(num_vertices, f, per_face_vector_field):
    device = f.device
    f_unrolled = f.flatten()
    face_indices_repeated_per_vertex = torch.arange(f.shape[0], device=f.device)
    face_indices_repeated_per_vertex = torch.repeat_interleave(face_indices_repeated_per_vertex, repeats=3)
    normals_repeated_per_face = per_face_vector_field[face_indices_repeated_per_vertex]
    incident_face_vectors = torch.zeros([num_vertices, per_face_vector_field.shape[-1]], device=device)
    incident_face_vectors = torch.index_add(incident_face_vectors, dim=0, index=f_unrolled,
                                            source=normals_repeated_per_face)
    return incident_face_vectors


def calculate_vertex_incident_vector_on_edges(num_vertices, edges, per_edge_vector_field):
    device = edges.device
    f_unrolled = edges.flatten()
    face_indices_repeated_per_vertex = torch.arange(edges.shape[0], device=edges.device)
    face_indices_repeated_per_vertex = torch.repeat_interleave(face_indices_repeated_per_vertex, repeats=2)
    normals_repeated_per_face = per_edge_vector_field[face_indices_repeated_per_vertex]
    incident_face_vectors = torch.zeros([num_vertices, per_edge_vector_field.shape[-1]], device=device)
    incident_face_vectors = torch.index_add(incident_face_vectors, dim=0, index=f_unrolled,
                                            source=normals_repeated_per_face)
    return incident_face_vectors


def calculate_vertex_valences(num_vertices, f):
    one_per_face = torch.ones(f.shape[0], device=f.device)
    return calculate_vertex_incident_scalar(num_vertices, f, one_per_face)


def calculate_adjacent_faces(f):
    """
    adjacent face ids for each vertex
    :param f: face tensor
    :return: adj_faces: tensor of adjacent face ids for each vertex (ordered by vertex id)
            num_of_incident_faces: number of adjacent vertices
            starting_idx: starting ids in a
    """
    f_flattened = f.flatten()
    sorted_order = torch.argsort(f_flattened)
    unique_vertices, num_of_incident_faces = torch.unique(f_flattened, return_counts=True)
    face_ids = torch.arange(f.shape[0], device=f.device).repeat_interleave(3)
    adj_faces = face_ids[sorted_order]
    starting_idx = torch.cumsum(num_of_incident_faces, dim=0)
    starting_idx = torch.roll(starting_idx, 1)
    starting_idx[0] = 0
    return adj_faces, num_of_incident_faces, starting_idx


def sparse_eye(n, device):
    vals = torch.ones(n, device=device)
    rows = torch.arange(n, device=device, dtype=torch.long)
    cols = torch.arange(n, device=device, dtype=torch.long)
    indices = torch.stack([rows, cols], dim=0)
    return torch.sparse_coo_tensor(indices, values=vals, device=device)


def normalized_laplacian(num_vertices, f):
    valences = calculate_vertex_valences(num_vertices, f)
    adj_mat = calculate_adj_matrix(f, normalize_rows=valences)
    return sparse_eye(num_vertices, device=f.device) - adj_mat


def laplacian_uniform(num_verts, faces, device='cuda'):
    """
    Compute the uniform laplacian

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    V = num_verts
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device=device, dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()


def calculate_incident_faces_to_edge(f, edges=None):
    """

    :param f: face tensor
    :return: faces_per_edge:
    """
    adj_faces, num_of_incident_faces, starting_idx = calculate_adjacent_faces(f)
    if edges is None:
        edges, _ = calc_edges(f)
    v_ids_for_incident_faces_per_edge = torch.repeat_interleave(edges.flatten(),
                                                                repeats=num_of_incident_faces[edges.flatten()])
    e_ids_for_incident_faces_per_edge = torch.repeat_interleave(torch.arange(edges.shape[0], device=f.device).flatten(),
                                                                repeats=torch.sum(num_of_incident_faces[edges], dim=-1))

    offset_vector = torch.arange(v_ids_for_incident_faces_per_edge.shape[0], device=f.device)
    cumsum = torch.cumsum(num_of_incident_faces[edges.flatten()], dim=0).roll(1)
    cumsum[0] = 0
    to_subtract = cumsum.repeat_interleave(repeats=num_of_incident_faces[edges.flatten()])
    offset_vector -= to_subtract
    starting_idx_for_v_ids = starting_idx[v_ids_for_incident_faces_per_edge]
    starting_idx_for_v_ids += offset_vector
    faces_per_edge = adj_faces[starting_idx_for_v_ids]
    return faces_per_edge, e_ids_for_incident_faces_per_edge, v_ids_for_incident_faces_per_edge


# todo: change to get_maximal_edge_value_over_faces
def get_maximal_face_value_over_edges(edges, f, face_values):
    faces_per_edge, e_ids_for_incident_faces_per_edge, v_ids_for_incident_faces_per_edge = \
        calculate_incident_faces_to_edge(f, edges)
    face_values = face_values[faces_per_edge]
    maximal_value_per_edge, _ = torch_scatter.scatter_max(face_values, e_ids_for_incident_faces_per_edge)
    return maximal_value_per_edge


def get_maximal_face_value_over_vertices(f, vertex_values):
    vertex_values_per_face = vertex_values[f]
    max_values = torch.max(vertex_values_per_face, dim=-1)
    return max_values[0]


def orient_edges(edges, f_for_orientation):
    ev0 = edges[:, 0]
    ev1 = edges[:, 1]
    fv0 = f_for_orientation[:, 0]
    fv1 = f_for_orientation[:, 1]
    fv2 = f_for_orientation[:, 2]
    flip_flag = ((ev1 == fv0) & (ev0 == fv1)) | ((ev1 == fv1) & (ev0 == fv2)) | ((ev1 == fv2) & (ev0 == fv0))
    oriented_edges = torch.tensor(edges, device=edges.device, dtype=torch.long)
    oriented_edges[flip_flag, 0] = edges[flip_flag, 1]
    oriented_edges[flip_flag, 1] = edges[flip_flag, 0]
    return oriented_edges
