from copy import deepcopy
import time
import torch_scatter
from ROAR.util.remeshing_utils import calc_face_collapses, collapse_edges_v2, rowwise_in, \
    pack, prepend_dummies, remove_dummies, face_split, calculate_k_hop_average, select_faces_and_edges, \
    collapse_given_edges, get_quality
from ROAR.sampling.regular_sampler import RegularSampler
from ROAR.util.geometry_utils import calc_edge_length, calc_face_normals, calculate_vertex_normals, calculate_dihedral_angles
from ROAR.util.topology_utils import calc_edges, get_maximal_face_value_over_edges
from ROAR.mesh_optimizer.geometry_scores import *
from ROAR.mesh_optimizer.Qslim_score import calculate_QSlim_score

def merge_dicts(main_dict, dict_to_merge, prefix_to_add = ""):
    for k,v in dict_to_merge.items():
        assert not k in main_dict, "key {} already exists in dict".format(k)
        main_dict[prefix_to_add + str(k)] = v
    return main_dict


def min_edge_length_per_vertex(vertices, faces):
    edges, _ = calc_edges(faces)
    edge_lengths = calc_edge_length(
        vertices, edges).repeat_interleave(repeats=2)
    edges = edges.flatten()
    min_lengths, incident_vertex_ids = torch_scatter.scatter_min(
        edge_lengths, edges, dim=0)
    return min_lengths, incident_vertex_ids


def lerp_unbiased(a: torch.Tensor, b: torch.Tensor, weight: float, step: int):
    """lerp with adam's bias correction"""
    c_prev = 1 - weight ** (step - 1)
    c = 1 - weight ** step
    a_weight = weight * c_prev / c
    b_weight = (1 - weight) / c
    a.mul_(a_weight).add_(b, alpha=b_weight)


class MeshOptimizer:
    """Use this like a pytorch Optimizer, but after calling opt.step(), do vertices,faces = opt.remesh()."""

    def __init__(self,
                 vertices: torch.Tensor,  # V,3
                 faces: torch.Tensor,  # F,3
                 opt,
                 allowed_vertices = None,
                 extra_vertex_attributes = None,
                 face_colors: torch.Tensor = None, 
                 target_colors_map = None
                 ):
        if opt is None:
            return
        
        self.target_colors_map = target_colors_map
        self.target_colors_map = target_colors_map
        self._vertices = vertices
        self._faces = faces
     
        self.face_colors = face_colors

        self.opt = deepcopy(opt)
        self.velocity = None
        self._lr = opt.lr
        self._ramp = opt.ramp
        self._grad_lim = opt.grad_lim
        self._step = 0
        V = self._vertices.shape[0]
        F = self._faces.shape[0]
        self._vertex_ramp = torch.ones(V, device=vertices.device) * opt.ramp
        self._start = time.time()
        # prepare continuous tensor for all vertex-based data
        self._vertices_etc = torch.zeros([V, 9], device=vertices.device, dtype=torch.float)
        if extra_vertex_attributes is not None:
            self._vertices_etc = torch.cat([self._vertices_etc, extra_vertex_attributes], dim=-1)
        self._face_attributes = torch.zeros([F, 8], device=vertices.device, dtype=torch.float)
        self.k = 1
        self._split_vertices_etc()
        self._split_face_attributes()
        self._vertices.copy_(vertices)  # initialize vertices
        self._ref_len[:] = min_edge_length_per_vertex(self._vertices,
                                                      self._faces)[0]
        self._vertices.requires_grad_()
        
        if self.face_colors is not None:
            self._face_attributes[:, :3] = self.face_colors[:]
            self.face_colors.requires_grad_()

        self.reg_sampler = RegularSampler(
            device=self._vertices.device, delta=opt.delta, max_sampling_n=opt.max_samples)
        self.allowed_vertices = allowed_vertices
        
    @property
    def vertices(self):
        return self._vertices

    @property
    def faces(self):
        return self._faces

    def get_allowed_faces_and_edges(self, faces = None, mode = 'relaxed'):
        if faces is None:
            faces = self._faces
        _, _, edge_to_face = calc_edges(faces, with_edge_to_face=True, with_dummies=False)
        if mode == 'relaxed':
            allowed_faces = self.allowed_vertices[faces].any(dim=-1)
        elif mode == 'strict':
            allowed_faces = self.allowed_vertices[faces].all(dim=-1)
        else:
            raise ValueError('allowed faces mode = [relaxed | strict]')
        allowed_edges = allowed_faces[edge_to_face[:,:,0]].all(dim=-1)
        return allowed_faces, allowed_edges
    
    def get_border_vertices_ids(self):
        # border vertices are any vertices on edge which contain one allowed and one frozen vertex
        edges , _ = calc_edges(self._faces, with_dummies=False)
        border_edges = self.allowed_vertices[edges].sum(dim=-1)==1
        border_vertices = edges[border_edges].flatten()
        return border_vertices             
        
    def _split_vertices_etc(self):
        """
        update vertex list and adam parameters after remeshing
        :return:
        """
        self._vertices = self._vertices_etc[:, :3]
        self._m2 = self._vertices_etc[:, 3]
        self._nu = self._vertices_etc[:, 4]
        self._m1 = self._vertices_etc[:, 5:8]
        self._ref_len = self._vertices_etc[:, 8]
        with_gammas = any(g != 0 for g in self.opt.gammas)
        self._smooth = self._vertices_etc[:,
                                          :8] if with_gammas else self._vertices_etc[:, :3]
    
    def _split_face_attributes(self):
        """
        update vertex list and adam parameters after remeshing
        :return:
        """
        self.face_colors = self._face_attributes[:, :3].contiguous()
        self._m2_faces = self._face_attributes[:, 3]
        self._nu_faces = self._face_attributes[:, 4]
        self._m1_faces = self._face_attributes[:, 5:8]

    def zero_grad(self):
        self._vertices.grad = None
        self._vertices_etc.grad = None
        if self.face_colors is not None:
            self.face_colors.grad = None
        if self.reg_sampler.sampled_vertices is not None:
            self.reg_sampler.sampled_vertices.grad = None
            self.reg_sampler.sampled_vertices_normals.grad = None

    @torch.no_grad()
    def step(self, vertex_weights=None, output_debug_info = False):
        
        eps = 1e-8
        self._step += 1
        debug_info = {}
        # spatial smoothing
        edges, _ = calc_edges(self._faces)  # E,2
        E = edges.shape[0]
        edge_smooth = self._smooth[edges]  # E,2,S
        neighbor_smooth = torch.zeros_like(
            self._smooth)  # V,S - mean of 1-ring vertices
        torch_scatter.scatter_mean(src=edge_smooth.flip(dims=[1]).reshape(E * 2, -1), index=edges.reshape(E * 2, 1),
                                   dim=0, out=neighbor_smooth)        
        # apply optional smoothing of m1,m2,nu
        if self.opt.gammas[0]:
            self._m1.lerp_(neighbor_smooth[:, 5:8], self.opt.gammas[0])
        if self.opt.gammas[1]:
            self._m2.lerp_(neighbor_smooth[:, 3], self.opt.gammas[1])
        if self.opt.gammas[2]:
            self._nu.lerp_(neighbor_smooth[:, 4], self.opt.gammas[2])
        laplace = self._vertices - neighbor_smooth[:, :3]  # laplace vector
        # find border allowed vertices
        grad = torch.nan_to_num(self._vertices.grad)
        grad = torch.addcmul(
            grad, laplace, self._nu[:, None], value=self.opt.laplacian_weight)
        # gradient clipping
        if self._step > 1:
            grad_lim = self._m1.abs().mul_(self.opt.grad_lim)
            grad.clamp_(min=-grad_lim, max=grad_lim)
        # moment updates
        lerp_unbiased(self._m1, grad, self.opt.betas[0], self._step)
        lerp_unbiased(self._m2, (grad ** 2).sum(dim=-1),
                      self.opt.betas[1], self._step)
        velocity = self._m1 / self._m2[:, None].sqrt().add_(eps)  # V,3
        self.velocity = velocity
        speed = velocity.norm(dim=-1)  # V
        if self.opt.betas[2]:
            lerp_unbiased(self._nu, speed, self.opt.betas[2], self._step)  # V
        else:
            self._nu.copy_(speed)  # V
        min_incident_edge_lengths, incident_vertex_id = min_edge_length_per_vertex(
            self._vertices, self._faces)
        self._ref_len[:] = min_incident_edge_lengths.clamp(
            max=self.opt.edge_len_lims[1])
        # update vertices
        ramped_lr = self._lr * \
            min(1, self._step * (1 - self.opt.betas[0]) / self._ramp)
        update_v = velocity * self._ref_len[:, None] * (-ramped_lr)
        update_v[~self.allowed_vertices] = 0
        if vertex_weights is not None:
            update_v.mul_(vertex_weights[:, None])
        self._vertices += update_v
        if output_debug_info:
            debug_info['update_v'] = update_v.detach().cpu().numpy()
        if self.face_colors is not None and self.face_colors.grad is not None and self.opt.color_weight>0:
            ramped_lr_faces = self._lr * min(1, self._step * (1 - self.opt.betas[0]) / self._ramp)
            grad_faces = self.face_colors.grad
            lerp_unbiased(self._m1_faces, grad_faces, self.opt.betas[0], self._step)
            lerp_unbiased(self._m2_faces, (grad_faces ** 2).sum(dim=-1), self.opt.betas[1], self._step)
            face_velocity = self._m1_faces / self._m2_faces[:, None].sqrt().add_(eps)  # V,3
            self.face_velocity = velocity
            speed_face = face_velocity.norm(dim=-1)  # V
            self._nu_faces.copy_(speed_face)  # V
            self.face_colors += face_velocity * (-ramped_lr_faces)
            # self.face_colors -= self.face_colors.grad * self._lr * 5000
            self.face_colors.clip_(0, 1)
            self._face_attributes[:, :3] = self.face_colors

        return self._vertices, debug_info
    
    
    @torch.no_grad()
    def remove_bad_faces(
            self,
            vertices_etc: torch.Tensor,  # V,D
            faces: torch.Tensor,  # F,3 long
            output_debug_info = False
    ):
        
        debug_info = {}
        _, allowed_edges =  self.get_allowed_faces_and_edges(faces)
        debug_info['before_remove_bad_faces_mesh'] = {'v': vertices_etc[:,:3].cpu().numpy(), 
                                                                       'f': faces.cpu().numpy()}
        # dummies
        vertices_etc, faces = prepend_dummies(vertices_etc, faces)
        vertices = vertices_etc[:, :3]  # V,3
        # collapse
        edges, face_to_edge = calc_edges(faces)  # E,2 F,3
        min_edgelen = torch.ones(edges.shape[0], device=edges.device) * self.opt.edge_len_lims[0]
        edge_length = calc_edge_length(vertices, edges)  # E
        face_normals = calc_face_normals(vertices, faces, normalize=True)  # F,3 #todo: should be True?
        vertex_normals = calculate_vertex_normals(vertices, faces)  # V,3
        edge_collapse_ids, face_collapse_id = calc_face_collapses(vertices, faces, edges, face_to_edge,
                                                                  face_normals,
                                                                  vertex_normals,
                                                                  edge_length,
                                                                  min_edgelen, area_ratio=0.5)
        # shortness = (1 - edge_length / min_edgelen[edges].mean(dim=-1)).clamp_min_(0)  # e[0,1] 0...ok, 1...edgelen=0
        priority = torch.ones(edges.shape[0], device=edges.device, dtype=torch.float) * np.inf
        v_mask = torch.ones(self.allowed_vertices.shape[0] + 1, dtype=bool, device = edges.device)
        v_mask[1:] = ~self.allowed_vertices
        qslim_scores, collapse_locations, debug_info = calculate_QSlim_score(vertices, faces,
                                                                 quality_threshold=None,
                                                                 avoid_folding_faces=False,
                                                                 v_mask= v_mask,
                                                                 with_dummies=True) 
        priority[edge_collapse_ids] = edge_length[edge_collapse_ids]

        priority_mask = priority[1:]
        priority_mask[~allowed_edges] = np.inf
        priority[1:] = priority_mask
        priority[0] = np.inf
        priority[torch.isinf(qslim_scores)] = np.inf

        vertices_etc, faces, collapsed, collapsed_edge_mask, affected_faces = collapse_edges_v2(vertices_etc, faces, edges,
                                                                                   priorities=torch.exp(-priority),
                                                                                   threshold=self.opt.face_fold_threshold,
                                                                                   collapse_locations=collapse_locations)
        self._face_attributes = self._face_attributes[~collapsed[1:]]      
        # if edge included frozen vertex, new vertex is frozen.
        self.allowed_vertices[edges[collapsed_edge_mask]-1] = (self.allowed_vertices[edges[collapsed_edge_mask]-1]).all(dim=-1).unsqueeze(-1).repeat([1,2])
        vertices_etc, faces, old_to_new_face_idx_map, used_vertices = pack(vertices_etc, faces)
        self.allowed_vertices = self.allowed_vertices[used_vertices[1:]]
        vertices_etc, faces, self.face_colors = remove_dummies(vertices_etc, faces, self.face_colors)
        affected_faces = old_to_new_face_idx_map[affected_faces.long()]
        affected_faces = affected_faces[affected_faces != -1]
        edges, face_to_edge = calc_edges(faces, with_dummies=False)
        initial_candidates = torch.zeros(edges.shape[0], device=self.opt.device, dtype=bool)
        _, allowed_edges =  self.get_allowed_faces_and_edges(faces)
        initial_candidates[allowed_edges] = True
        
        if output_debug_info:
            debug_info['after_remove_bad_faces_mesh'] = {'v': vertices_etc[:,:3].cpu().numpy(), 
                                                                     'f': faces.cpu().numpy()}
            debug_info['edges_after_remove_bad_faces_arr'] = edges.cpu().numpy()
            debug_info['affected_faces_arr'] = affected_faces.cpu().numpy()
            debug_info['initial_candidates_arr'] = initial_candidates.cpu().numpy()
            debug_info['face_collapse_id_arr'] = face_collapse_id[1:].cpu().numpy()
            debug_info['edge_collapse_ids_arr'] = (edge_collapse_ids - 1).cpu().numpy()
        
        flip_edges_debug_info = self.flip_edges(vertices_etc[:, :3], faces,
                                        with_border=False,
                                        initial_candidates=initial_candidates.clone(),
                                        check_degree=True,
                                        face_score_ratio=None,
                                        number_of_edges_limit=2000,
                                        output_debug_info=output_debug_info)     
        debug_info = merge_dicts(debug_info, flip_edges_debug_info)
        
        return vertices_etc, faces, debug_info
    
    # def calculate_edge_flip_score (self,vertices,faces,face_scores):
        
    #     _, _, ef = calc_edges(faces, with_edge_to_face=True, with_dummies=False)
    #     ef_face_scores = face_scores[ef[:,:,0]]
    #     max_ef_face_scores = torch.max(ef_face_scores, dim=-1)[0]
    #     min_ef_face_scores = torch.min(ef_face_scores, dim=-1)[0]
    #     score = min_ef_face_scores*max_ef_face_scores #face_score_diff/(1+face_score_diff)
    #     score = score/(1+score)
    #     return score

    # @torch.no_grad()
    # def improve_quality(
    #         self,
    #         vertices_etc: torch.Tensor,  # V,D
    #         faces: torch.Tensor,  # F,3 long
    #         sizing_field,  # V,1
    # ):
    #     iterations = self.opt.tangential_relaxation_iterations
    #     tangential_vertex_norm_threshold = self.opt.tangential_relaxation_threshold
    #     m = torch.nn.ConstantPad1d((1, 0), 0)
    #     sizing_field = m(sizing_field)
    #     vertices_etc, faces = prepend_dummies(vertices_etc, faces)
    #     for _ in range(iterations):
    #         vertex_normals_norm = torch.norm(calculate_vertex_normals(vertices_etc[:, :3], faces, normalize=False),
    #                                          dim=-1)
    #         vertex_normals_norm[vertex_normals_norm <
    #                             tangential_vertex_norm_threshold] *= 0.001
    #         for _ in range(1):
    #             vertices_etc[:, :3] = tangential_relxation(vertices_etc[:, :3], faces, sizing_field,
    #                                                        weights=vertex_normals_norm)
    #     vertices_etc, faces = remove_dummies(vertices_etc, faces)
    #     return vertices_etc, faces
    
    
    @torch.no_grad()
    def flip_edges(self,
                   vertices: torch.Tensor,  # V,3 first unused
                   faces: torch.Tensor,  # F,3 long, first must be 0, 0 for unused

                   face_score_ratio: torch.Tensor = None,
                   max_iterations = 10,
                   with_border: bool = True,  # handle border edges (D=4 instead of D=6)
                   initial_candidates=None,
                   check_degree=True,
                   with_normal_check: bool = True,  # check face normal flips
                   number_of_edges_limit=None,
                   dihedral_angle_threshold = 0.05,
                   output_debug_info = False,
                   stable: bool = False,  # only for unit testing
                   ):

        V = vertices.shape[0]
        device = vertices.device
        flip_edge_to_face = None
        flipped = None
        updated_1_rings = None
        #total_flipped = None

        debug_info = {}
        # else:
        #     new_candidate_ids = torch.arange(initial_candidates.shape[0], device=self.opt.device)

        for i in range(max_iterations):
        
            #new_candidate_ids = torch.where(initial_candidates)[0]

            edges, fe, edge_to_face = calc_edges(faces, with_edge_to_face=True, with_dummies=False)
            # if total_flipped is None:
            #     total_flipped = torch.zeros(edges.shape[0], dtype=torch.bool, device = 'cuda')
            if updated_1_rings is not None and face_score_ratio is not None:
                ef_face_score_update, _, _ = self.calculate_face_score(vertices=vertices,
                                                                       faces=faces[
                                                                           edge_to_face[updated_1_rings.flatten(), :,
                                                                           0]].reshape(-1, 3))
                ef_face_score_update = torch.sort(ef_face_score_update.reshape(-1, 2), dim=-1)[0]
                face_score_ratio[updated_1_rings.flatten()] = ef_face_score_update[:, 0] / (
                        ef_face_score_update[:, 1] + 1e-8) + ef_face_score_update[:, 0]
                
            E = edges.shape[0]
            vertex_degree = torch.zeros(V, dtype=torch.long, device=device)  # V long
            vertex_degree.scatter_(dim=0, index=edges.reshape(E * 2), value=1, reduce='add')
            neighbor_corner = (edge_to_face[:, :, 1] + 2) % 3  # go from side to corner
            neighbors = faces[edge_to_face[:, :, 0], neighbor_corner]  # E,LR=2, the opposite vertex of the 1-ring
            ee = fe[edge_to_face[:, :, 0]]
            e0 = ee[torch.arange(edges.shape[0], device=device), 0, (edge_to_face[:, 0, 1] + 1) % 3]
            e1 = ee[torch.arange(edges.shape[0], device=device), 0, (edge_to_face[:, 0, 1] + 2) % 3]
            e2 = ee[torch.arange(edges.shape[0], device=device), 1, (edge_to_face[:, 1, 1] + 1) % 3]
            e3 = ee[torch.arange(edges.shape[0], device=device), 1, (edge_to_face[:, 1, 1] + 2) % 3]
            edge_ring = torch.stack([e0, e1, e2, e3], dim=-1)

            # todo: this only works with dummies..
            # edge_is_inside = neighbors.all(dim=-1)  # E

            # if with_border:
            #     # inside vertices should have D=6, border edges D=4, so we subtract 2 for all inside vertices
            #     # need to use float for masks in order to use scatter(reduce='multiply')
            #     vertex_is_inside = torch.ones(V, 2, dtype=torch.float32, device=vertices.device)  # V,2 float
            #     src = edge_is_inside.type(torch.float32)[:, None].expand(E, 2)  # E,2 float
            #     vertex_is_inside.scatter_(dim=0, index=edges, src=src, reduce='multiply')
            #     vertex_is_inside = vertex_is_inside.prod(dim=-1, dtype=torch.long)  # V long
            #     vertex_degree -= 2 * vertex_is_inside  # V long

            neighbor_degrees = vertex_degree[neighbors]  # E,LR=2
            edge_degrees = vertex_degree[edges]  # E,2

            loss_change = torch.zeros(edges.shape[0], device=device, dtype=torch.float)
            #candidates = torch.zeros(edges.shape[0], device=device, dtype=torch.bool)

            candidates= initial_candidates.clone()

            if output_debug_info:
                valence_loss = (-(2 + neighbor_degrees.sum(dim=-1)) + edge_degrees.sum(
                    dim=-1)) * 1 #self.opt.edge_flip_degree_weight  # E
                debug_info['{}_edge_flip_valence_loss_arr'.format(i)] = valence_loss.cpu().numpy()

            if check_degree:
                loss_change += (-(2 + neighbor_degrees.sum(dim=-1)) + edge_degrees.sum(
                    dim=-1)) * 1#self.opt.edge_flip_degree_weight  # E            
                candidates &= loss_change > 0  # torch.logical_and(loss_change < 0)#, edge_is_inside)  # & small_dihedral_angle  # E
                if number_of_edges_limit is not None and (torch.where(candidates)[0]).shape[0] > number_of_edges_limit:
                    candidates[torch.where(candidates)[0][number_of_edges_limit]:] = False
                if loss_change.shape[0] == 0:
                    return
                
            # allowed edges condition
            #allowed_edges_condition = self.allowed_vertices[neighbors].all(dim=-1) & self.allowed_vertices[edges].all(dim=-1)
            allowed_edges_condition = self.allowed_vertices[neighbors].any(dim=-1) | self.allowed_vertices[edges].all(dim=-1)
            candidates[candidates.clone()] &= allowed_edges_condition[candidates]

            if with_normal_check:
                #  cl-<-----e1     e0,e1...edge, e0<e1
                #   |      /A      L,R....left and right face
                #   |  L /  |      both triangles ordered counter clockwise
                #   |  / R  |      normals pointing out of screen
                #   V/      |
                #   e0---->-cr
                edges_neighbors = torch.concat((edges[candidates], neighbors[candidates]), dim=-1)  # E',4
                v = vertices[edges_neighbors]  # E",4,3
                v = v - v[:, 0:1]  # make relative to e0
                e1 = v[:, 1]
                cl = v[:, 2]
                cr = v[:, 3]
                n = torch.cross(e1, cl) + torch.cross(cr, e1)  # sum of old normal vectors
                candidates[candidates.clone()] &= (torch.sum(n * torch.cross(cr, cl), dim=-1) > -self.opt.eps_val) & \
                                                  (torch.sum(n * torch.cross(cl - e1, cr - e1),
                                                             dim=-1) > -self.opt.eps_val)  # first new face and second new face

            # do not flip any which already has a zero face score ratio
            if not check_degree:
                nonzero_ratio_candidates = face_score_ratio[candidates] > 0
                candidates[candidates.clone()] &= nonzero_ratio_candidates

            # todo: do what-if analysis for vertex normals, and face fold score. reject candidates with negative face fold score
            # for those that do pass, keep vertex normals for following face score calculation.

            # simulate face score after edge flip for each relevant edge
            edges_neighbors = torch.concat((edges[candidates], neighbors[candidates]), dim=-1)  # E',4
            flip_faces = edges_neighbors[:, [[0, 3, 2], [1, 2, 3]]]  # E",2,3
            if not check_degree and face_score_ratio is not None and self.rayCaster is not None and flip_faces.shape[
                0] > 0:
                flipped_ef_face_score, _, flipped_edges_face_score_debug_info = self.calculate_face_score(vertices=vertices,
                                                                        faces=flip_faces.reshape(-1, 3),
                                                                        mode='edge flip', output_debug_info=output_debug_info)

                # todo: flipped_ef_correction_score is wrong becuase of exterior normals
                #flipped_ef_face_score += flipped_ef_correction_score
                # if output_debug_info:
                #     debug_info[i]['fipped_faces_arr'] = flip_faces.reshape(-1, 3).cpu().numpy()
                #     debug_info[i]['flipped_ef_face_score_arr'] = flipped_ef_face_score.cpu().numpy()
                #     debug_info[i]['flipped_faces_sampled_points_arr'] = self.reg_sampler.sampled_vertices.cpu().numpy()
                #     debug_info[i]['flipped_faces_sampled_normals_arr'] = self.reg_sampler.sampled_vertices_normals.cpu().numpy()
                #     debug_info[i]['flipped_faces_projected_points_arr'] = flipped_edges_face_score_debug_info['projected_points_arr']
                #     debug_info[i]['flipped_faces_projection_mask_arr'] = flipped_edges_face_score_debug_info['projection_mask_arr']
                flipped_ef_face_score = torch.sort(flipped_ef_face_score.reshape(-1, 2), dim=-1)[0]
                flipped_face_score_ratio = flipped_ef_face_score[:, 0] / (
                        flipped_ef_face_score[:, 1] + 1e-8) + flipped_ef_face_score[:, 0]             

                # padded_flipped_ef_face_score = torch.zeros(candidates.shape[0], device=self.opt.device)
                # padded_flipped_ef_face_score[candidates] = flipped_face_score_ratio
                # to avoid "flip-floping" edges
                candidates[candidates.clone()] &= flipped_face_score_ratio <= face_score_ratio[candidates]

                    #loss_change[candidates] += (flipped_face_score_ratio - face_score_ratio[candidates])
                    # todo: base this on dihedral angles (change must be low)
                    #candidates[candidates.clone()] &= flipped_face_score_ratio <= face_score_ratio[candidates]
            elif check_degree:
                dihedral_angles = calculate_dihedral_angles(vertices, faces).squeeze(-1)
                for_edge_collapse = (dihedral_angles > dihedral_angle_threshold)
                candidates[candidates.clone()] &= for_edge_collapse[candidates]

            # condition to avoid creating non-manifold edges
            edges_neighbors = torch.concat((edges[candidates], neighbors[candidates]), dim=-1)  # E',4
            flip_edges_neighbors = edges_neighbors  # E",4 = 1-ring around edge (vertices)
            sorted_opposite_vertices_ids = torch.sort(flip_edges_neighbors[:, 2:], dim=-1)[0]
            candidates[candidates.clone()] &= ~rowwise_in(sorted_opposite_vertices_ids, edges)

            edges_neighbors = torch.concat((edges[candidates], neighbors[candidates]), dim=-1)  # E',4
            loss_change = loss_change[candidates]  # E'
            _, order = loss_change.sort(descending=False, stable=stable)  # E'
            rank = torch.zeros_like(order)
            rank[order] = torch.arange(0, len(rank), device=rank.device)
            vertex_rank = torch.zeros((V, 4), dtype=torch.long, device=device)  # V,4
            _, argmax = torch_scatter.scatter_max(src=rank[:, None].expand(-1, 4), index=edges_neighbors, dim=0, out=vertex_rank)
            vertex_rank, _ = vertex_rank.max(dim=-1)  # V
            neighborhood_rank, _ = vertex_rank[edges_neighbors].max(dim=-1)  # E'
            flip = rank == neighborhood_rank  # E'
            if torch.all(~flip):
                break

            flip_edges_neighbors = edges_neighbors[flip]  # E",4 = 1-ring around edge (vertices)
            flip_edge_to_face = edge_to_face[candidates, :, 0][flip]  # E",2
            flip_faces = flip_edges_neighbors[:, [[0, 3, 2], [1, 2, 3]]]  # E",2,3
            flipped = candidates.clone()
            flipped[candidates] &= flip

            if output_debug_info:
                debug_info['{}_edges_arr'.format(i)] = edges.cpu().numpy()
                debug_info['{}_candidates_arr'.format(i)] = initial_candidates.cpu().numpy()
                debug_info['{}_flipped_arr'.format(i)] = flipped.cpu().numpy()
                debug_info['{}_mesh'.format(i)] = {'v': vertices.cpu().numpy(), 'f':faces.long().cpu().numpy()}

            faces.scatter_(dim=0, index=flip_edge_to_face.reshape(-1, 1).expand(-1, 3), src=flip_faces.reshape(-1, 3))

            # todo: assert that number of allowed faces stays the same

            if output_debug_info:
                debug_info['{}_after_flip_mesh'.format(i)] = {'v': vertices.cpu().numpy(), 'f':faces.long().cpu().numpy()}

            sorted_neighbours = torch.sort(neighbors, dim=-1)[0]

            repeated_edges_per_flipped = edges.unsqueeze(1).repeat([1, sorted_neighbours[flipped].shape[0], 1])
            # edge_id encodes the position of the new (flipped) edges in the edge tensor. the edge tensor is #Ex2 and lexicographically sorted.   
            edge_id = torch.all(~( repeated_edges_per_flipped < sorted_neighbours[flipped]), dim=-1).T | (repeated_edges_per_flipped[:,:,0] > sorted_neighbours[flipped][:,0]).T
            # turn off flipped edges
            initial_candidates[flipped] = False

            # indices is the first index in edge_id where either:
            # 1. the second and first index of the flipped edge are bigger or equal to the edges' indices
            # 2. only the first index of the flipped edge is bigger than the first index of the edge's indices
            idx = torch.arange(edge_id.shape[1], 0, -1, device=device)
            tmp2 = edge_id * idx
            indices = torch.argmax(tmp2, 1, keepdim=True)  # indices may not be unique!
            indices[torch.all(~edge_id, dim=-1)] = edges.shape[0] # bug fixed! if edge_id is all false, the potential index needs to be at the end of the edge tensor

            edge_offsets_add = torch.arange(edges.shape[0], device=device).unsqueeze(0).repeat(
                [neighbors[flipped].shape[0], 1]) >= indices
            edge_offsets_add = edge_offsets_add.int()
            edge_offsets_remove = torch.arange(edges.shape[0], device=device).unsqueeze(0).repeat(
                [neighbors[flipped].shape[0], 1]) > torch.where(flipped)[0].unsqueeze(-1)
            edge_offsets_remove = edge_offsets_remove.int() * (-1)
            old_to_new_edges_map = torch.sum(edge_offsets_add, dim=0) + torch.sum(edge_offsets_remove, dim=0) + \
                                   torch.arange(edges.shape[0], device=device)
            new_candidate_ids = old_to_new_edges_map[torch.where(initial_candidates)[0]]
            initial_candidates[:] = False
            initial_candidates[new_candidate_ids] = True
            if face_score_ratio is not None and self.rayCaster is not None:
                updated_1_rings = old_to_new_edges_map[edge_ring[flipped]]
                # premute update face score ratios according to old_to_new_edges_map
                face_score_ratio_permuted = torch.zeros(face_score_ratio.shape[0] + 1, device=device)

                if output_debug_info:
                    edges_new, fe, edge_to_face = calc_edges(faces, with_edge_to_face=True, with_dummies=False)
                    debug_info['{}_edges_after_flip_arr'.format(i)] = edges_new.cpu().numpy()
                    debug_info['{}_one_rings_arr'.format(i)] = edge_ring[flipped].cpu().numpy()
                    debug_info['{}_updated_1_rings_arr'.format(i)] = updated_1_rings.cpu().numpy()

                # -1 goes to last index
                old_to_new_edges_map[torch.where(flipped)[0]] = face_score_ratio.shape[0]
                face_score_ratio_permuted = torch.scatter(face_score_ratio_permuted, 0, old_to_new_edges_map,
                                                          face_score_ratio)
                face_score_ratio = face_score_ratio_permuted[:-1]
            
            # update face attributes
            # for j in range(old_to_new_edges_map.shape[0]):
            #     if j in torch.where(flipped)[0]:
            #         continue
            #     assert torch.isclose(face_score_ratio_permuted[old_to_new_edges_map[j]],
            #                          face_score_ratio[j])
            # todo: calulcate effected edges after flip
            if new_candidate_ids.shape[0] == 0:
                break

            if output_debug_info:
                debug_info['flipped_arr'] = flipped.cpu().numpy()
        return debug_info


    @torch.no_grad()
    def update_topology(self,
                        selected_faces,
                        selected_edges,
                        edges,
                        collapse_locations,
                        face_scores=None,
                        edge_flip_correction = None,
                        output_debug_info = False,
                        ):
        
        debug_info = {}

        if face_scores is None:
            # todo: calculate face scores
            pass

        # if output_debug_info:
        #     edges, _ = calc_edges(self._faces)
        #     debug_info["before_topology_update_edges_arr"] = edges.cpu().numpy()
        #     edge_flip_score = self.calculate_edge_flip_score(self._vertices, self._faces, edge_flip_correction)
        #     debug_info["before_topology_update_edge_flip_score_arr"] = edge_flip_score.cpu().numpy()

        affected_faces = torch.empty(0, device=self.opt.device, dtype=torch.long)
        collapsed = None
        if selected_edges.shape[0] > 0:
            self._vertices_etc, self._faces = prepend_dummies(self._vertices_etc, self._faces)
            edges, _ = calc_edges(self._faces)
            selected_edges += 1
            m = torch.nn.ConstantPad1d((1, 0), 0)
            collapse_locations = m(collapse_locations)
            self._vertices_etc, self._faces, collapsed, affected_faces = collapse_given_edges(self._vertices_etc,
                                                                                            self._faces, edges,
                                                                                            selected_edges,
                                                                                            collapse_locations)
            self._face_attributes = self._face_attributes[~collapsed[1:]]  
            # if edge included forzen vertex, new vertex is frozen. todo: add this to vertex attributes in vertex_etc
            self.allowed_vertices[edges[selected_edges]-1] = (self.allowed_vertices[edges[selected_edges]-1]).all(dim=-1).unsqueeze(-1).repeat([1,2])                  
            self._vertices_etc, self._faces, old_to_new_face_idx_map, used_vertices = pack(self._vertices_etc, self._faces)
            self.allowed_vertices = self.allowed_vertices[used_vertices[1:]]
            self._vertices_etc, self._faces, self.face_colors = remove_dummies(self._vertices_etc, self._faces)
            selected_faces = old_to_new_face_idx_map[1:][selected_faces]
            affected_faces = old_to_new_face_idx_map[affected_faces]
            affected_faces = affected_faces[affected_faces != -1]

        # update face scores
        face_split_mask = torch.zeros(self._faces.shape[0], dtype=torch.bool, device=selected_faces.device)
        face_split_mask[selected_faces] = True
        self._vertices_etc, self._faces, new_face_ids, old_to_new_face_idx_map = face_split(self._vertices_etc,
                                                                                            self._faces, face_split_mask)
        repeats = torch.ones(face_split_mask.shape[0], device=self.opt.device, dtype=torch.long)
        repeats[face_split_mask] = 3
        self._face_attributes = torch.repeat_interleave(self._face_attributes, repeats=repeats, dim=0)
        amount_of_vertices_added = face_split_mask.sum()
        self.allowed_vertices = torch.cat([self.allowed_vertices, torch.ones(amount_of_vertices_added, device = self.opt.device, dtype=torch.bool)])
        # get the edges as inital candidates
        affected_faces = torch.cat([old_to_new_face_idx_map[affected_faces], new_face_ids], dim=0)

        # update face scores only in affected faces
        updated_face_scores = torch.ones(self._faces.shape[0], device=self.opt.device) * (-1)
        affected_faces_mask = torch.zeros(self._faces.shape[0], device=self.opt.device, dtype=torch.bool)
        affected_faces_mask[affected_faces] = True
        if face_scores is not None:
            if collapsed is not None:
                updated_face_scores[old_to_new_face_idx_map] = face_scores[~collapsed[1:]]
            else:
                updated_face_scores[old_to_new_face_idx_map] = face_scores
        edges, fe, edge_to_face = calc_edges(self._faces, with_edge_to_face=True, with_dummies=False)
        affected_edges = fe[affected_faces].flatten()
        initial_candidates = torch.zeros(edges.shape[0], device=edges.device, dtype=torch.bool)
        initial_candidates[affected_edges] = True
        _, allowed_edges =  self.get_allowed_faces_and_edges()
        initial_candidates[~allowed_edges] = False

        debug_info['initial_candidates_arr'] = initial_candidates.cpu().numpy()
                        
        if affected_faces.shape[0] > 0 and face_scores is not None:
            updated_face_scores[affected_faces], edge_flip_correction, _ = self.calculate_face_score(
                                                                                  vertices=self._vertices_etc[:, :3],
                                                                                  faces=self._faces[affected_faces])
        if output_debug_info:
            debug_info['after_topology_update_before_edge_flip_mesh'] = { 'v': self._vertices_etc[:,:3].detach().cpu().numpy(),
                                                                'f': self._faces.cpu().numpy()}


        edges, fe, edge_to_face = calc_edges(self._faces, with_edge_to_face=True, with_dummies=False)
        
        if output_debug_info:     
            debug_info["edges_after_topology_update_arr"] = edges.cpu().numpy()
        # calculate face score ratio for affacted faces

        # updated_face_scores, edge_flip_correction, _ = self.calculate_face_score(vertices=self._vertices_etc[:, :3],
        #                                                                           faces=self._faces)
        # ef_face_scores = torch.sort(updated_face_scores[edge_to_face[:, :, 0]], dim=-1)[0]
        # face_score_ratio = ef_face_scores[:, 0] / (ef_face_scores[:, 1] + 1e-8) + ef_face_scores[:, 0]

        # debug_info['face score ratio_arr'] = face_score_ratio.cpu().numpy()

        # flipped = torch.zeros(edges.shape[0], device = edges.device, dtype = bool)

        # flip_edges_debug_info = self.flip_edges(self._vertices_etc[:, :3], self._faces,
        #                        with_border=False,
        #                        initial_candidates=initial_candidates.clone(),
        #                        with_normal_check = True,
        #                        check_degree= False ,
        #                        face_score_ratio= face_score_ratio,
        #                        number_of_edges_limit=50,
        #                        output_debug_info = output_debug_info)
        
        # debug_info = merge_dicts(debug_info, flip_edges_debug_info)      

        affected_edges = fe[affected_faces].flatten()
        
        if output_debug_info:
            edges, _ = calc_edges(self._faces, with_edge_to_face=False, with_dummies=False)
            if flipped is None:
               flipped = torch.zeros(edges.shape[0], device = edges.device, dtype=torch.bool)
            debug_info['flipped_edges_after_topology_update_arr'] = flipped.detach().cpu().numpy()
            debug_info['after_topology_update_after_edge_split_mesh'] = {'v': self._vertices_etc[:,:3].detach().cpu().numpy(),
                                                                'f': self._faces.cpu().numpy()}
            debug_info['affected_edges_arr'] = affected_edges.cpu().numpy()
            debug_info['affected_faces_arr'] = affected_faces.cpu().numpy()
            
        return self._vertices_etc, self._faces, debug_info


    def remesh(self, time_step, output_debug_info = False):

        debug_info = {}
        debug_info['before_remesh_mesh'] = {'v': self._vertices.cpu().numpy(), 'f': self._faces.cpu().numpy()}
        if self.opt.folded_face_cleanup_interval > 0 and time_step % self.opt.folded_face_cleanup_interval == 0:
            self.reg_sampler.sample_regular(
                self._vertices.detach().unsqueeze(0), self._faces.unsqueeze(0))
            self._vertices_etc, self._faces, remove_bad_faces_debug_info = self.remove_bad_faces(self._vertices_etc, self._faces,
                                                                            output_debug_info = output_debug_info)
            
            debug_info = merge_dicts(debug_info, remove_bad_faces_debug_info, prefix_to_add='remove_folded_')
            
            self._split_face_attributes()
            self._split_vertices_etc()

            debug_info['after_remove_bad_faces_mesh'] = {'v': self._vertices.cpu().numpy(), 'f':self._faces.cpu().numpy()}
            
        if self.opt.face_split_interval > 0 and time_step % self.opt.face_split_interval == 0 and time_step > 0:
            # todo: change to meshSDF configuration
            edges, fe, ef = calc_edges(self._faces, with_edge_to_face=True, with_dummies = False)
            self.reg_sampler.sample_regular(
                self._vertices.detach().unsqueeze(0), self._faces.unsqueeze(0))
            face_scores, edge_flip_correction, face_score_debug_info = self.calculate_face_score(output_debug_info=output_debug_info)
            edge_scores, collapse_locations, edges, edge_score_debug_info = self.calculate_edge_score(
                self._vertices[:, :3], self._faces,
                face_scores, output_debug_info = output_debug_info)
            #allowed_edges = self.allowed_vertices[edges].all(dim=-1)
            allowed_faces, allowed_edges =  self.get_allowed_faces_and_edges()
            face_scores[ef[:,:,0][edge_scores < self.opt.edge_curvature_threshold]]=0
            face_scores[~allowed_faces] = 0
            edge_scores[~allowed_edges] = np.inf # allowed edges are thoses who are both incident on allowed triangles

            selected_faces, selected_edges, self.topology_loss = select_faces_and_edges(self._vertices, self._faces,
                                                                                        face_scores,
                                                                                        edge_scores,
                                                                                        zero_edge_score_threshold = self.opt.edge_curvature_threshold,
                                                                                        num_allowed_faces=self.opt.max_faces)
            
            if output_debug_info:
                debug_info = merge_dicts(debug_info, edge_score_debug_info, prefix_to_add='update_topology_')
                debug_info['before_topology_update_mesh'] =  {'v': self._vertices.cpu().numpy(), 'f': self._faces.cpu().numpy()}
                #debug_info['vertex_normals_head_angles'] = vertex_normals.cpu().detach().numpy()
                debug_info['sampled_points_arr'] = self.reg_sampler.sampled_vertices_normals.cpu().detach().numpy()
                debug_info['sampled_faces_arr'] = self.reg_sampler.sampled_faces.cpu().detach().numpy()
                debug_info['sampled_face_ids_arr'] = self.reg_sampler.sampled_face_ids.cpu().detach().numpy()
                debug_info['face_scores_arr'] = face_scores.cpu().numpy()
                debug_info['edges_arr'] =  edges.cpu().numpy()
                debug_info['selected_edges_arr'] = selected_edges.cpu().numpy()
                debug_info['selected_faces_arr'] = selected_faces.cpu().numpy()
                debug_info['edge_scores_arr'] = edge_scores.detach().cpu().numpy()
                debug_info['face_ids_arr'] =  self.reg_sampler.face_ids.cpu().detach().numpy()

            #debug_info['sampled_face_folds'] = sampled_face_scores.cpu().detach().numpy()
            #debug_info['sampled_face_normals_after'] = face_normals_after.cpu().detach().numpy()
            #debug_info['vertex_normal_normals_after'] = updated_vertex_normals.cpu().detach().numpy()
            # debug_info['folded_faces_to_collapse'] = collapse_id.cpu().detach().numpy()
            # debug_info['folded_edges_to_collapse'] = edges_to_collapse.cpu().detach().numpy()
            # debug_info['face_score_ratio'] = face_score_ratio.cpu().detach().numpy()

            self._vertices_etc, self._faces, update_topology_debug_info = self.update_topology(selected_faces, selected_edges, edges,
                                                            collapse_locations,
                                                            face_scores=face_scores,
                                                            edge_flip_correction = edge_flip_correction,
                                                            output_debug_info = output_debug_info)
            
            debug_info = merge_dicts(debug_info, update_topology_debug_info, prefix_to_add='update_topology_')
            self._split_vertices_etc()
            self._split_face_attributes()
            self.face_colors.requires_grad_()

            debug_info['after_topoloy_update_mesh'] = {'v': self._vertices.cpu().numpy(), 'f':self._faces.cpu().numpy()}

        self._vertices.requires_grad_()
        self.reg_sampler.sample_regular(
            self._vertices.unsqueeze(0), self._faces.unsqueeze(0))
        return self._vertices, self._faces, debug_info


    @torch.no_grad()
    def calculate_face_score(self, vertices=None, faces=None, mode='face split', output_debug_info = False):
        debug_info = {}

        if faces is None:
            faces = self._faces
        if vertices is None:
            vertices = self._vertices

        self.reg_sampler.sample_regular(vertices.detach().unsqueeze(0), faces.unsqueeze(0))
     
        projected_point_colors = None

        if self.opt.reconstruction_mode == 'triangle soup' or self.opt.reconstruction_mode == 'mesh':
            projected_points, projection_mask, projected_point_color_ids, projected_face_ids = project_to_triangle_soup(self, vertices, faces)
            if self.rayCaster.target_colors is not None:
                projected_point_colors = self.rayCaster.target_colors[projected_point_color_ids]
            else:
                projected_point_colors = None
        elif self.opt.reconstruction_mode in ["sdf", "csg", "point cloud"]:
            projected_points, projection_mask, sdf_vals, projected_sdf_vals = face_split_sdf_sphere_tracing(self.reg_sampler.sampled_vertices, self.reg_sampler.sampled_vertices_normals, self.rayCaster)
            debug_info['projected_points_original_arr'] = projected_points.detach().cpu().numpy()
            projected_points[projection_mask] = self.reg_sampler.sampled_vertices[projection_mask]
            debug_info['sdf_vals_arr'] = sdf_vals.detach().cpu().numpy()
            debug_info['projected_sdf_vals_arr'] = projected_sdf_vals.detach().cpu().numpy()
        else:
            raise NotImplementedError("invalid face split mode")

        if mode == 'face split':
            face_curvature_threshold = self.opt.face_curvature_threshold
        else:
            face_curvature_threshold = 0
            #projection_mask[self.reg_sampler.external_vertices_mask] = False

        face_normals = calc_face_normals(vertices, faces, normalize=True)
        face_scores, edge_flip_face_score_correction, sampled_face_scores, face_normals_after, updated_vertex_normals, sampled_faces_mask, sampled_face_corrections = face_scores_face_folds(projected_points, faces, self.reg_sampler,
                                                                  face_normals = face_normals,
                                                                  curvature_threshold=face_curvature_threshold,
                                                                  projected_points_mask=~projection_mask,
                                                                  mode=mode)
        
        #debug_info['masked_faces.txt'] = masked_faces.cpu().numpy()

        if (self.opt.reconstruction_mode == 'triangle soup' or self.opt.reconstruction_mode == 'mesh') and self.opt.color_weight > 0 and self.rayCaster.face_colors is not None:
            face_scores += self.opt.color_weight * self.calculate_color_score(projected_point_color_ids, projected_face_ids, vertices,
                                                                              faces, mode=mode)
            
        if (self.opt.reconstruction_mode == 'triangle soup' or self.opt.reconstruction_mode == 'mesh') and self.opt.quality_weight > 0:
            face_scores += self.opt.quality_weight * self.calculate_face_quality_score(self.opt.k, vertices, faces)

        # required to keep sliver faces from recieving high face scores due to color score
        # face_areas = double_face_areas(vertices.unsqueeze(0), faces.unsqueeze(0)).flatten()
        # if torch.any(face_scores>1e-5):
        #     face_areas /= torch.max(face_areas)
        #     face_scores *= face_areas
        #face_scores *= torch.exp(-get_quality(vertices, faces, max_val=5) + 1)

        # face_scores[face_scores < 1e-6] = 0

        face_fold_score = calculate_face_folds(vertices, faces)
        # self intersecting faces
        face_scores[face_fold_score > 1] = 0

        if mode == 'face split':
            #face_scores[(self.reg_sampler.n_tag.flatten() < 3)] = 0
            face_scores[(face_scores < 0)] = 0
            edges, fe = calc_edges(faces, with_dummies=False)
            edges_lengths = calc_edge_length(vertices, edges)
            face_edge_lengths = edges_lengths[fe]
            face_scores[torch.any(face_edge_lengths < self.opt.edge_len_lims[0]*0.1, dim=-1)] = 0


        if output_debug_info:
            debug_info['projected_points_arr'] = projected_points.cpu().numpy()
            debug_info['projection_mask_arr'] = projection_mask.cpu().numpy()
            debug_info['sampled_face_scores_arr'] = sampled_face_scores.cpu().numpy() 
            debug_info['face_normals_after_arr'] = face_normals.cpu().numpy() 
            debug_info['updated_vertex_normals_arr'] = updated_vertex_normals.cpu().numpy()
            debug_info['sampled_faces_mask_arr'] = sampled_faces_mask.cpu().numpy()
            debug_info['sampled_face_corrections_arr'] = sampled_face_corrections.cpu().numpy()
            #debug_info['projected_point_colors_arr'] = projected_point_colors.cpu().numpy()
        
        return face_scores, edge_flip_face_score_correction, debug_info

        # if mode == 'face split':
        #     face_scores[(face_scores < 0) | (self.reg_sampler.n_tag.flatten() < 3)] = 0
        #     edges, fe = calc_edges(faces, with_dummies=False)
        #     edges_lengths = calc_edge_length(vertices, edges)
        #     face_edge_lengths = edges_lengths[fe]
        #     face_scores[torch.any(face_edge_lengths < self.opt.edge_len_lims[0]*0.1, dim=-1)] = 0

    def calculate_color_score(self, projected_colors_id, vertices=None, faces=None):
        if faces is None:
            faces = self._faces
        if vertices is None:
            vertices = self._vertices
        amount_of_colors = self.rayCaster.target_colors.shape[0]
        # todo: change this to simple projection
        face_ids = self.reg_sampler.face_ids
        idx = projected_colors_id
        # idx, face_ids, mask = self.get_colors_per_face(v_target, n_target, sampled_colors_id, faces=faces,
        #                                                vertices=vertices)
        # for each sampled vertex, get color
        colors_per_face = idx  # sampled_colors_id[idx]
        sum_colors_per_face = torch.zeros(faces.shape[0] * amount_of_colors, device=colors_per_face.device,
                                          dtype=torch.int32)
        amount_per_face = torch.zeros(faces.shape[0], device=colors_per_face.device, dtype=torch.int32)
        amount_per_face = torch.scatter_add(amount_per_face, 0, face_ids,
                                            torch.ones_like(face_ids, device=colors_per_face.device, dtype=torch.int32))
        offset = torch.arange(faces.shape[0], device=colors_per_face.device) * amount_of_colors
        offset = offset.repeat_interleave(amount_per_face)
        colors_per_face += offset
        per_sampled_point_vote = torch.ones_like(colors_per_face, device=colors_per_face.device,
                                                 dtype=torch.int32)
        #per_sampled_point_vote[projected_face_ids] = 0
        sum_colors_per_face = torch.scatter_add(sum_colors_per_face, 0, colors_per_face, per_sampled_point_vote)
        sum_colors_per_face = sum_colors_per_face.reshape([-1, amount_of_colors])
        # if face split mode - add masked for each face. todo: still required?
        # if mode == 'face split':
        #     sum_colors_per_face.index_add_(dim=0, index=face_ids[mask],
        #                                    source=torch.ones([face_ids[mask].shape[0], amount_of_colors],
        #                                                      device=v_target.device, dtype=torch.int32))
        sum_colors_per_face = torch.nan_to_num(sum_colors_per_face, nan=1) / amount_per_face.unsqueeze(-1)
        color_scores = 1 - torch.max(sum_colors_per_face, dim=-1)[0]
        return color_scores 
    
    def calculate_edge_quality_score(self, k, vertices=None, faces=None):
        if faces is None:
            faces = self._faces
        if vertices is None:
            vertices = self._vertices
        k_avg = calculate_k_hop_average(vertices, faces, k).squeeze(-1)
        edges, fe = calc_edges(faces, with_dummies=False)
        lengths = calc_edge_length(vertices, edges)
        # length < 3/5*k_mean
        edge_k_avg = torch.mean(k_avg[edges], dim=-1)
        edge_quality_scores = (3 * edge_k_avg / 5.0 - lengths).clip(min=0) / edge_k_avg
        return edge_quality_scores

    def calculate_edge_score(self, vertices, faces, face_scores, with_dummies = False, output_debug_info = False):

        """
        larger score = more redundant
        :param vertices: #V x 3 mesh vertices
        :param faces: #F x 3 mesh faces
        :param face_scores: #F calculated face scores
        :return: edge score for clearing redundant edges.
        """

        edges, _, _ = calc_edges(faces, with_edge_to_face=True, with_dummies=with_dummies)  # E,2 F,3
        potential_score = get_maximal_face_value_over_edges(edges, faces, face_scores)
        edge_score = potential_score  # * self.opt.potential_score_coeff
        # todo: project self.colors to nearest pallete color, get ids
        # if self.target_colors_map is not None:
        #     color_dists = torch.norm(self.face_colors.unsqueeze(1) - self.target_colors_map.unsqueeze(0), dim=-1)
        #     color_ids = torch.argmin(color_dists, dim=-1)
        #     one_hot_encoding = torch.eye(self.target_colors_map.shape[0], device=vertices.device)
        #     color_one_hot_encoding = one_hot_encoding[color_ids]
        # else:
        #     color_one_hot_encoding = None

        debug_info = {}

        if self.opt.color_weight > 0:
            face_colors = self.face_colors
        else:
            face_colors = None
        qslim_score, collapse_locations, qslim_debug_info = calculate_QSlim_score(vertices.detach(), faces, face_colors=face_colors,
                                                                quality_threshold=self.opt.collapse_quality_threshold,
                                                                avoid_folding_faces=False, v_mask=~self.allowed_vertices, output_debug_info = output_debug_info)
        
        assert torch.all(~torch.isnan(qslim_score))
        
        debug_info = merge_dicts(debug_info, qslim_debug_info)

        edge_score += qslim_score
        assert torch.all(~torch.isnan(edge_score))
        if self.opt.quality_weight > 0:
            edge_score += self.opt.quality_weight*self.calculate_edge_quality_score(self.opt.k, vertices ,faces)
        # if return_score_dict:
        #     edge_score_dict = {"potential_score": potential_score,
        #                        "qslim_score": qslim_score}
        #     return edge_score, collapse_locations, edges, edge_score_dict

        return edge_score, collapse_locations, edges, debug_info
