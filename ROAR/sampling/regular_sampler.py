import torch
from ROAR.util.geometry_utils import double_face_areas, calculate_face_normals_and_areas, calculate_vertex_normals, calculate_edge_normals
from ROAR.util.topology_utils import consolidate_indices, calc_edges
import igl


class RegularSampler:

    """
    regularly supersamples faces on mesh using the scheme described in the paper. This regular sampling maintains internal topology, allowing
    quantities such as curvature to be calculated.
    """

    def __init__(self, max_sampling_n=80, delta=0.8 / 100, device="cpu", with_ignore=False):
        self.amount_of_vertices_for_each_face = None
        self.sampled_vertices = None
        self.sampled_vertices_normals = None
        self.delta = delta
        self.max_samples = max_sampling_n
        self.device = device
        self.n_tag = None
        self.face_ids = None
        self.sampled_faces = None
        self.sampled_face_ids = None
        self.face_list = None

        indexing_tensor = [torch.arange(i) for i in range(1, max_sampling_n)]
        level_tensor = [torch.ones_like(indexing_tensor[i]) * i for i in
                        range(max_sampling_n - 1)]
        indexing_tensor = torch.cat(indexing_tensor)
        indexing_tensor2 = torch.cat(level_tensor) - indexing_tensor
        self.indexing_tensor = torch.stack([indexing_tensor, indexing_tensor2], dim=-1).to(device)

    def create_sample_points(self, vs, faces, offset = 0):
        """
        uniformally sample points on the selected faces.
        self.delta is the density of the samples

        :param vs: #V x 3 vertices of target mesh
        :param faces: #F x 3 faces of target mesh
        :param face_list: 1-D Tensor of face ids to sample
        :return:
        self.vertices: #sV x 3 sampled vertices on faces
        self.coordinates: #sV x 2 barycentric coordinates of sampled faces (on parent face)
        self.face_id_tensor_masked:
        self.n_tag: #F Tensor, n_tag_ceil*(n_tag_ceil+1)//2 is the amount of sampled vertices per face
        self.local_vertex_idx: #sV Tensor of local ids of sampled vertex in parent face (ordinal numbers)
        self.face_start_idx: #F Tensor of indices in self.local_vertex_idx where a new face begins to be sampled
        """

        # calculate the amount of points needed per face to uniformly sample the mesh given the required delta.
        bounding_box_diagonal = torch.norm(torch.max(vs, dim=1)[0] - torch.min(vs, dim=1)[0])
        T = double_face_areas(vs, faces)
        n_tag = torch.sqrt(0.25 + T / bounding_box_diagonal ** 2 / self.delta ** 2) - 0.5
        # clipping maximum number of sampled vertices per face.
        n_tag[n_tag > self.max_samples - 5] = self.max_samples - 5

        #ignore faces with not enough samples for curvature (minimum 3 needed)
        faces_to_ignore = n_tag < 3

        #n_tag is a real number, and needs to be an integer. probability of a rounding up:
        probability = torch.floor(n_tag[~faces_to_ignore]) / 2 + 1 - T[
            ~faces_to_ignore] / 2 / bounding_box_diagonal ** 2 / self.delta ** 2 * (
                              1 / (torch.floor(n_tag[~faces_to_ignore]) + 1))

        # mask for rounding up num samples
        floor_mask = faces_to_ignore.flatten() * False
        floor_mask[~faces_to_ignore.flatten()] = torch.rand_like(n_tag[~faces_to_ignore]) < probability

        n_tag[faces_to_ignore] = 3

        #round down all num_samples
        n_tag = torch.floor(n_tag)

        #round up num samples where required
        n_tag[~faces_to_ignore & ~floor_mask.unsqueeze(0).unsqueeze(-1)] += 1

        #regular sampling
        face_tensor = torch.arange(self.max_samples * (self.max_samples + 1) // 2, device=vs.device).unsqueeze(0).unsqueeze(
            1).repeat([1, faces.shape[1], 1])
        face_id_tensor = torch.arange(faces.shape[1], device=vs.device).unsqueeze(-1).repeat(
            [1, self.max_samples * (self.max_samples + 1) // 2]).unsqueeze(0)

        face_first_vec = ((vs[:, faces[0, :]][:, :, 1] - vs[:, faces[0, :]][:, :,
                                                         0]) / (n_tag - 1)).unsqueeze(-2)

        face_second_vec = ((vs[:, faces[0, :]][:, :, 2] - vs[:, faces[0, :]][:, :,
                                                          0]) / (n_tag - 1)).unsqueeze(-2)
        face_first_vertex = vs[:, faces[0, :]][:, :, 0]
        
        if offset != 0:
            face_first_vec = face_first_vec + face_first_vec/torch.norm(face_first_vec, dim=-1, keepdim=True)*offset
            face_second_vec = face_second_vec + face_second_vec/torch.norm(face_second_vec, dim=-1, keepdim=True)*offset
            face_first_vertex_dir = face_first_vertex - vs[:, faces[0, :]].mean(dim=2)
            face_first_vertex = face_first_vertex + face_first_vertex_dir/torch.norm(face_first_vertex_dir, dim=-1, keepdim=True)*offset

        face_base_vector_tensor = torch.cat([face_first_vec, face_second_vec], dim=-2)
        face_base_vector_tensor = face_base_vector_tensor.unsqueeze(-3).repeat(
            [1, 1, self.max_samples * (self.max_samples + 1) // 2, 1, 1])
        face_first_vertex = face_first_vertex.unsqueeze(-2).repeat([1, 1, self.max_samples * (self.max_samples + 1) // 2, 1])
        mask = face_tensor < n_tag * (n_tag + 1) // 2
        face_base_vector_tensor = face_base_vector_tensor[mask]
        face_first_vertex = face_first_vertex[mask]
        face_tensor_masked = face_tensor[mask]  # face tensor holds the parent face idx for every sampled point
        self.face_ids = face_id_tensor[mask]
        coordinates = self.indexing_tensor[face_tensor_masked]  # convert ordinal numbers in face idx to coordinates of indexing tensor

        self.sampled_vertices = torch.sum(coordinates.unsqueeze(-1) * face_base_vector_tensor,
                                          dim=1) + face_first_vertex
        
        self.n_tag = n_tag.flatten().long()
        assert torch.all(self.n_tag>0)
        face_start_idx = torch.where(face_tensor_masked == 0)[0]
        local_vertex_idx = face_tensor_masked

        # return values instead of internal variables
        return coordinates, face_start_idx, local_vertex_idx


    def sample_regular(self, v, f, face_list=None, orient_normals_strategy = 'smooth', offset = 0.0):
        """
        samples vertices regularly on faces of mesh (v,f)
        :param v:
        :param f:
        :param face_list:
        :return:
        sampled_faces:
        sampled_face_ids:
        """
        if face_list is not None:
            f = f[:,face_list]
        self.face_list = face_list

        # uniformly sample points
        coordinates, face_start_idx, local_vertex_idx = self.create_sample_points(v, f, offset)
        self.amount_of_vertices_for_each_face = (self.n_tag * (self.n_tag + 1) // 2).flatten()
        #with torch.no_grad():
        forward_triangle_first = coordinates + torch.tensor([1, 0], device=v.device)
        forward_triangle_second = coordinates + torch.tensor([0, 1], device=v.device)

        forward_triangle_first_idx, forward_triangle_second_idx = \
            self.set_up_triangles(forward_triangle_first, forward_triangle_second)

        forward_triangle_first_mask = forward_triangle_first_idx < self.amount_of_vertices_for_each_face[self.face_ids]
        forward_triangle_second_mask = forward_triangle_second_idx < self.amount_of_vertices_for_each_face[
            self.face_ids]

        forward_triangle_mask = forward_triangle_first_mask & forward_triangle_second_mask

        # convert from coordinates to indices for lower triangle
        backward_triangle_first = coordinates - torch.tensor([1, 0], device=v.device)
        backward_triangle_second = coordinates - torch.tensor([0, 1], device=v.device)

        backward_triangle_first_idx, backward_triangle_second_idx = \
            self.set_up_triangles(backward_triangle_first, backward_triangle_second)

        backward_triangle_first_mask = torch.all(backward_triangle_first >= 0,
                                                dim=-1)  # backward_triangle_first_idx >= 0
        backward_triangle_second_mask = torch.all(backward_triangle_second >= 0,
                                                dim=-1)  # backward_triangle_second_idx >= 0

        backward_triangle_mask = backward_triangle_second_mask & backward_triangle_first_mask

        # every vertex is assigned its parent face's normal
        face_normals, face_areas = calculate_face_normals_and_areas(v.squeeze(0), f.squeeze(0))
        sampled_vertex_normals = face_normals.repeat_interleave(repeats=self.n_tag * (self.n_tag + 1) // 2, dim=0)

        # edge0_mask = coordinates[:, 1] == 0
        # n_tag_repeated_for_vertices = (self.n_tag - 1).repeat_interleave(repeats=self.n_tag * (self.n_tag + 1) // 2)
        # edge1_mask = torch.sum(coordinates, dim=-1) == n_tag_repeated_for_vertices
        # edge2_mask = coordinates[:, 0] == 0
        # self.vertex_mask0 = (coordinates[:, 0] == 0) & (coordinates[:, 1] == 0)
        # self.vertex_mask1 = edge1_mask & (coordinates[:, 1] == 0)
        # self.vertex_mask2 = edge1_mask & (coordinates[:, 0] == 0)


        # self.external_vertices_mask = self.vertex_mask0 | self.vertex_mask1 | self.vertex_mask2 | edge0_mask | edge1_mask | edge2_mask

        if orient_normals_strategy == 'seperate':
            edge0_mask = coordinates[:, 1] == 0
            edge2_mask = coordinates[:, 0] == 0
            n_tag_repeated_for_vertices = (self.n_tag - 1).repeat_interleave(repeats=self.n_tag * (self.n_tag + 1) // 2)
            edge1_mask = torch.sum(coordinates, dim=-1) == n_tag_repeated_for_vertices

            _, fe, ef = calc_edges(f.squeeze(0), with_edge_to_face=True, with_dummies=False)
            # ef1 = ef[:, 0, 0]
            # ef2 = ef[:, 1, 0]
            # todo: normalize this according to head angle too?
            # for every face calculate head angle 
            # edge_normals = (face_normals[ef1] * face_areas[ef1] + face_normals[ef2] * face_areas[ef2]) / (
            #         face_areas[ef1] + face_areas[ef2])
            fe_normals = calculate_edge_normals(v[0],f[0])[fe]
            
            fe_normals0 = fe_normals[:, 0].repeat_interleave(repeats=self.n_tag * (self.n_tag + 1) // 2, dim=0)[edge0_mask]
            fe_normals1 = fe_normals[:, 1].repeat_interleave(repeats=self.n_tag * (self.n_tag + 1) // 2, dim=0)[edge1_mask]
            fe_normals2 = fe_normals[:, 2].repeat_interleave(repeats=self.n_tag * (self.n_tag + 1) // 2, dim=0)[edge2_mask]
            sampled_vertex_normals[edge0_mask] = fe_normals0
            sampled_vertex_normals[edge1_mask] = fe_normals1
            sampled_vertex_normals[edge2_mask] = fe_normals2
            self.vertex_mask0 = (coordinates[:, 0] == 0) & (coordinates[:, 1] == 0)
            self.vertex_mask1 = edge1_mask & (coordinates[:, 1] == 0)
            self.vertex_mask2 = edge1_mask & (coordinates[:, 0] == 0)
            vertex_normals = calculate_vertex_normals(v[0], f[0])
            sampled_vertex_normals[self.vertex_mask0] = vertex_normals[f[0, :, 0]]
            sampled_vertex_normals[self.vertex_mask1] = vertex_normals[f[0, :, 1]]
            sampled_vertex_normals[self.vertex_mask2] = vertex_normals[f[0, :, 2]]

        elif orient_normals_strategy == 'smooth':
            barycentric_coords = coordinates/(self.n_tag[self.face_ids]-1).unsqueeze(-1)
            barycentric_coords = torch.cat([1-barycentric_coords.sum(dim=-1, keepdim=True), barycentric_coords], dim=-1)
            self.barycentric_coords = barycentric_coords
            vertex_normals = calculate_vertex_normals(v[0], f[0])
            face_vertex_normals_per_sampled_v = vertex_normals[f[0]][self.face_ids]
            sampled_vertex_normals = (face_vertex_normals_per_sampled_v * barycentric_coords.unsqueeze(-1)).sum(dim=1)

        elif orient_normals_strategy == 'face only':
            pass
            
        else:
            raise NotImplementedError("regular sampling normal orientation policy does not exist")

        self.sampled_vertices_normals = sampled_vertex_normals/torch.norm(sampled_vertex_normals, dim=-1, keepdim=True)

        masked_local_vertices = (local_vertex_idx + face_start_idx[self.face_ids])
        masked_forward_triangle_first_idx = (forward_triangle_first_idx + face_start_idx[self.face_ids])
        masked_forward_triangle_second_idx = (forward_triangle_second_idx + face_start_idx[self.face_ids])
        masked_backward_triangle_first_idx = (backward_triangle_first_idx + face_start_idx[self.face_ids])
        masked_backward_triangle_second_idx = (backward_triangle_second_idx + face_start_idx[self.face_ids])
        masked_forward_triangle_mask = forward_triangle_mask
        masked_backward_triangle_mask = backward_triangle_mask
        forward_faces = torch.stack(
            [masked_local_vertices.cpu(), masked_forward_triangle_first_idx.cpu(), masked_forward_triangle_second_idx.cpu()], dim=-1)
        forward_faces = forward_faces.to(device=v.device)
        forward_faces_masked = forward_faces[masked_forward_triangle_mask]
        backward_faces = torch.stack(
            [masked_local_vertices, masked_backward_triangle_first_idx, masked_backward_triangle_second_idx], dim=-1)
        backward_faces_masked = backward_faces[masked_backward_triangle_mask]
        self.sampled_faces = consolidate_indices(torch.cat([forward_faces_masked, backward_faces_masked]))

        face_ids_forward = self.face_ids[forward_triangle_first_mask & forward_triangle_second_mask]
        face_ids_backward = self.face_ids[backward_triangle_second_mask & backward_triangle_first_mask]
        self.sampled_face_ids = torch.cat([face_ids_forward, face_ids_backward])
        self.face_ids = self.face_ids
        if face_list is not None:
            self.sampled_face_ids = face_list[self.sampled_face_ids]
            self.face_ids = face_list[self.face_ids]
        
    def set_up_triangles(self, triangle_first, triangle_second):
        triangle_first_idx = torch.sum(triangle_first, dim=-1) * (
                torch.sum(triangle_first, dim=-1) + 1) // 2 + triangle_first[:, 0]
        triangle_second_idx = torch.sum(triangle_second, dim=-1) * (
                torch.sum(triangle_second, dim=-1) + 1) // 2 + triangle_second[:, 0]
        return triangle_first_idx, triangle_second_idx

    # currently not used
    def calculate_integral_sum(self, triangle_first_idx, triangle_second_idx,
                               triangle_mask, min_dists, double_sample_area_vec):
        # convert from coordinates to indices for lower triangle
        e_i_j = min_dists[:, torch.where(triangle_mask)[0]]
        e_ip1_j = min_dists[:, triangle_first_idx[triangle_mask]]
        e_i_jp1 = min_dists[:, triangle_second_idx[triangle_mask]]
        triangle_error = e_i_j * (e_i_j + e_ip1_j + e_i_jp1) + e_ip1_j * (e_ip1_j + e_i_jp1) + e_i_jp1 ** 2
        triangle_error = triangle_error * double_sample_area_vec[triangle_mask] / 12
        return torch.sum(triangle_error)

    def find_sampled_vertices_for_face_id(self, face_id):
        """
        for debugging
        :param face_id: face id from original (pre-sampling) mesh
        :return:
        the local (v,f) pair for the regular sampling of the fiven face_id
        the ids of the sampled vertices in the full regular sampling
        """
        assert self.sampled_vertices is not None, "no sampled vertices"
        relevant_sampled_face_ids = torch.where(self.sampled_face_ids == face_id)[0]
        consolidated_faces = consolidate_indices(self.sampled_faces[relevant_sampled_face_ids])
        relevant_sampled_vertices_ids = torch.unique(self.sampled_faces[relevant_sampled_face_ids].flatten())
        relevant_sampled_vertices = self.sampled_vertices[relevant_sampled_vertices_ids]
        return relevant_sampled_vertices, consolidated_faces, relevant_sampled_vertices_ids, relevant_sampled_face_ids