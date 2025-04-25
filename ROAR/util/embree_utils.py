import torch
import embree
from ROAR.sampling.random_sampler import sample_points_from_meshes
from ROAR.util.geometry_utils import get_perp_vectors, rotate_vectors_by_angles_around_axis, \
    distance_between_points_and_lines, calc_face_normals
import numpy as np


class RayRatioVisibility:

    def __init__(self, minimal_horizon=75 * np.pi / 180, density_in_plane=0.3, num_rays_elevation=8, num_samples=50000,
                 min_ciruclar_rays=6):
        self.verts = None
        self.num_verts = None

        self.faces = None
        self.num_faces = None

        self.centroids = None

        self.device = None
        self.scene = None
        self.geometry = None

        self.density_in_plane = density_in_plane
        self.num_rays_elevation = num_rays_elevation
        self.minimal_horizon = minimal_horizon
        self.num_samples = num_samples
        self.min_circular_rays = min_ciruclar_rays

    def set_up(self, verts, faces, target_color_ids = None, target_colors = None, texture_info = None):
        """

        :param verts: in numpy format
        :param faces: in numpy format
        :param target_color_ids: for each face, record its color 1-hot encoding
        :param target_colors: all colors for target_mesh
        :return:
        """
        self.verts = verts.astype(np.float32)
        self.num_verts = verts.shape[0]

        self.faces = faces.astype(np.uint32)
        self.num_faces = faces.shape[0]

        self.target_color_ids = target_color_ids
        self.target_colors = target_colors

        if target_color_ids is not None:
            self.face_colors = target_colors[target_color_ids]
        else:
            self.face_colors = None

        self.texture_info = texture_info

        v = torch.tensor(verts, device = 'cuda', dtype=torch.float)
        f = torch.tensor(faces, device = 'cuda', dtype=torch.int64)
        self.target_face_normals = calc_face_normals(v, f, normalize=True)

        if target_color_ids is not None:
            fv = v[f].reshape(-1,3)
            ff = torch.arange(fv.shape[0], device = 'cuda', dtype=torch.long).reshape([-1,3])

            fv += self.target_face_normals.repeat_interleave(3, dim=0)*1e-5

            self.verts = fv.cpu().numpy().astype(np.float32)
            self.num_verts = fv.shape[0]

            self.faces = ff.cpu().numpy().astype(np.uint32)
            self.num_faces = ff.shape[0]

        self.centroids = self.verts[self.faces].mean(1)

        self.device = embree.Device()
        self.scene = self.device.make_scene()
        self.geometry = self.device.make_geometry(embree.GeometryType.Triangle)
        vertex_buffer = self.geometry.set_new_buffer(
            embree.BufferType.Vertex,  # buf_type
            0,  # slot
            embree.Format.Float3,  # fmt
            3 * np.dtype('float32').itemsize,  # byte_stride
            self.num_verts)  # item_count
        vertex_buffer[:] = self.verts[:]

        index_buffer = self.geometry.set_new_buffer(
            embree.BufferType.Index,  # buf_type
            0,  # slot
            embree.Format.Uint3,  # fmt
            3 * np.dtype('uint32').itemsize,  # byte_stride,
            self.num_faces)  # item count
        index_buffer[:] = self.faces[:]

        self.geometry.commit()
        self.scene.attach_geometry(self.geometry)
        self.geometry.release()
        self.scene.commit()

        self.verts = torch.tensor(self.verts, device = 'cuda', dtype=torch.float)
        self.faces = torch.tensor(self.faces.astype(np.int32), device = 'cuda', dtype=torch.int64)


    def create_ray_umbrella(self, points, normals):
        # The first ray is always the normal
        ray_umbrella = torch.tensor(normals, device=points.device)
        sample_point_ids = torch.arange(points.shape[0], device=points.device)
        perp_vectors = get_perp_vectors(normals)
        # find vector that is min_horizon away
        for i in range(self.num_rays_elevation - 1):
            d_angle = self.minimal_horizon / (self.num_rays_elevation - 1) * (i + 1)
            d_angles_vec = torch.ones([normals.shape[0], 1], device=points.device, dtype=torch.float) * d_angle
            rotated_vectors = \
                rotate_vectors_by_angles_around_axis(normals, d_angles_vec, perp_vectors)
            # calculate the amount of required circular vectors
            radii = distance_between_points_and_lines(points, normals, rotated_vectors + points)
            num_circle_vectors = torch.round(2 * np.pi * radii) / self.density_in_plane
            num_circle_vectors = torch.max(num_circle_vectors,
                                           torch.ones_like(num_circle_vectors,
                                                           device=points.device) * self.min_circular_rays)
            d_circle_angle = 2 * np.pi / num_circle_vectors
            curr_angle = torch.zeros_like(d_circle_angle)
            for j in range(int(num_circle_vectors[0].item())):
                rotated_in_circle = rotate_vectors_by_angles_around_axis(rotated_vectors, curr_angle.unsqueeze(-1),
                                                                         normals)
                ray_umbrella = torch.cat([ray_umbrella, rotated_in_circle], dim=0)
                sample_point_ids = torch.cat([sample_point_ids, torch.arange(points.shape[0], device=points.device)],
                                             dim=0)
                curr_angle += d_circle_angle
        return ray_umbrella, sample_point_ids
    


    def ray_ratio_face_visibilty(self, v, f, debug_info=False):
        # always sample each faces centeroid
        centroids = torch.mean(v[f], dim=-2)
        face_normals = calc_face_normals(v, f, normalize=True)
        sampled_points, sampled_normals, face_idx = sample_points_from_meshes(v.unsqueeze(0), f.unsqueeze(0),
                                                                              self.num_samples,
                                                                              return_normals=True)
        sampled_points = sampled_points.squeeze(0)
        sampled_normals = sampled_normals.squeeze(0)
        sampled_points = torch.cat([centroids, sampled_points], dim=0)
        sampled_normals = torch.cat([face_normals, sampled_normals], dim=0)
        face_idx = torch.cat([torch.arange(f.shape[0], device=v.device), face_idx], dim=0)
        # for each point, sample a ray umbrella
        # return number of vectors for each circle
        # return the sample vertex id for each ray
        sample_point_score, no_hit, ray_umbrella, sampled_point_ids = self.test_point_visibility(sampled_points,
                                                                                                 sampled_normals)
        sampled_face_score = torch.zeros(f.shape[0], device=v.device)
        sampled_face_score = torch.scatter_add(sampled_face_score, 0, face_idx, sample_point_score)
        # calculate the amount of samples in each face
        number_of_samples_per_faces = torch.zeros(f.shape[0], device=v.device)
        ones = torch.ones_like(face_idx, dtype=torch.float)
        number_of_samples_per_faces = torch.scatter_add(number_of_samples_per_faces, 0, face_idx, ones)
        if debug_info:
            return no_hit, ray_umbrella, sampled_point_ids, sampled_points, sampled_normals, sampled_face_score / number_of_samples_per_faces
        # calculate ratio for each face.
        return sampled_face_score / number_of_samples_per_faces


    def test_point_visibility(self, points, point_normals, directions = None, target_distances_sqrd = None, return_color = False):
        # return color for each hit face
        if directions is None:
            ray_umbrella, sampled_point_ids = self.create_ray_umbrella(points, point_normals)
        else:
            ray_umbrella = directions
            sampled_point_ids = torch.arange(points.shape[0], device=points.device)
        sample_point_score = torch.zeros(points.shape[0], device=points.device)
        no_hit = torch.zeros(points.shape[0], device=points.device, dtype=bool)
        prim_id = torch.zeros(points.shape[0], device=points.device, dtype=torch.long)
        t = torch.zeros(points.shape[0], device=points.device, dtype=torch.float)
        if target_distances_sqrd is not None:
            sampled_target_distances_sqrd = target_distances_sqrd[sampled_point_ids]
        # use embree
        for i in range(ray_umbrella.shape[0]//1000000+1):
            tmp_ray_umbrella = ray_umbrella[i*1000000:(i+1)*1000000]
            rayhit = embree.RayHit1M(tmp_ray_umbrella.shape[0])
            rayhit.org[:] = points[sampled_point_ids][i*1000000:(i+1)*1000000].cpu().numpy()
            rayhit.dir[:] = tmp_ray_umbrella.cpu().numpy()
            rayhit.tnear[:] = 1e-6
            rayhit.tfar[:] = np.inf
            rayhit.flags[:] = 0
            rayhit.geom_id[:] = embree.INVALID_GEOMETRY_ID
            # todo: intersection filter can be used to cull backfaces
            context = embree.IntersectContext()
            self.scene.intersect1M(context, rayhit) #need to keep below 1M?
            geom_id = rayhit.geom_id
            prim_id[i*1000000:(i+1)*1000000] = torch.tensor(rayhit.prim_id.astype(np.int64), dtype=torch.long, device = points.device)
            t[i*1000000:(i+1)*1000000] = torch.tensor(rayhit.tfar, dtype=torch.float, device = points.device)
            #t = torch.tensor(t, device=points.device)
            # keep track of which rays hit something.
            # scatter hits back to sample points and then back to faces
            if target_distances_sqrd is None:
                no_hit_tmp = geom_id == embree.INVALID_GEOMETRY_ID
                no_hit[i*1000000:(i+1)*1000000] = torch.tensor(no_hit_tmp, device=points.device, dtype = torch.bool)
                sample_point_score = torch.scatter_add(sample_point_score, 0, sampled_point_ids[i*1000000:(i+1)*1000000], no_hit[i*1000000:(i+1)*1000000].float())
            else:
                no_hit[i*1000000:(i+1)*1000000] = sampled_target_distances_sqrd[i*1000000:(i+1)*1000000] > t**2
                sample_point_score = torch.scatter_add(sample_point_score, 0, sampled_point_ids[i*1000000:(i+1)*1000000], no_hit[i*1000000:(i+1)*1000000].float() )
        
        #prim_id = torch.tensor(rayhit.prim_id.astype(np.int64), dtype=torch.long, device = points.device)
        # clamp sample point score to 1
        sample_point_score[sample_point_score <= ray_umbrella.shape[0]/points.shape[0] //2] = 0

        projected_colors = torch.zeros(points.shape[0], device = points.device, dtype = torch.long)
        if self.face_colors is not None:
            projected_colors[~no_hit] = self.target_color_ids[prim_id[~no_hit]]
        sample_point_score = torch.clamp(sample_point_score, max=1)
        return 1-sample_point_score, no_hit, t, projected_colors, sampled_point_ids, prim_id