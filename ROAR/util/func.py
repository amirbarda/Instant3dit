
import torch
import io
import numpy as np
from pathlib import Path
import trimesh
import imageio
#import open3d as o3d
#from pytorch3d.ops.knn import knn_points
import torch.nn.functional as tfunc
from ROAR.sampling.triangle_soup_to_pointcloud import triangle_soup_to_point_cloud_random
from ROAR.util.geometry_utils import calculate_vertex_normals, rotation_angle_between_two_vectors_and_axis
from subprocess import check_call
import math
import torch.nn.functional as F
#import requests
import signal
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import os
import gzip
import pickle
import sys
import igl

def read_mtlfile(fname):
    materials = {}
    with open(fname) as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        if line:
            prefix, data = line.split(' ', 1)
            if 'newmtl' in prefix:
                material = {}
                materials[data] = material
            elif materials:
                if len(data.split(' ')) > 1:
                    material[prefix] = tuple(float(d) for d in data.split(' '))
                else:
                    if data.isdigit():
                        try:
                            material[prefix] = int(data)
                        except ValueError:
                            material[prefix] = float(data)
                    else:
                        material[prefix] = data

    return materials
    

def calculate_elevation_and_azimuth(points):
    x_vecs = torch.tensor([1,0,0], dtype=torch.float, device=points.device)
    x_vecs = x_vecs.expand(points.shape[0],-1)
    z_vecs = torch.tensor([0,0,1], dtype=torch.float, device=points.device)
    z_vecs = z_vecs.expand(points.shape[0],-1)
    # todo: get vector from origin to points
    vectors = points
    # project from point to xy plane - handle edge case of zero vector
    xy_vectors = torch.tensor(vectors, device=points.device)
    xy_vectors[:,2] = 0
    xy_vectors = tfunc.normalize(xy_vectors)
    # calculate the clockwise rotation around z needed to get it to be in the xz axis - this is the azimuth
    cos_elevation = torch.sum(tfunc.normalize(points)*xy_vectors,dim=-1)
    azimuth = rotation_angle_between_two_vectors_and_axis(x_vecs, xy_vectors, z_vecs)
    # rotate vector and calculate the rotation around the y axis to get the vector in the z direction
    return azimuth.flatten(), torch.acos(cos_elevation).flatten()*torch.sign(points[:,2])

def reasd_mtlfile(fname):
    materials = {}
    with open(fname) as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        if line:
            prefix, data = line.split(' ', 1)
            if 'newmtl' in prefix:
                material = {}
                materials[data] = material
            elif materials:
                if len(data.split(' ')) > 1:
                    material[prefix] = tuple(float(d) for d in data.split(' '))
                else:
                    if data.isdigit():
                        try:
                            material[prefix] = int(data)
                        except ValueError:
                            material[prefix] = float(data)
                    else:
                        material[prefix] = data

    return materials


def slice_mesh_above_plane(v, f, plane):
    """slices a mesh above a plane, returns the new mesh"""
    mask = plane[:3] @ v.T > -plane[-1]
    mask = mask[f].all(axis=1)
    new_v, new_f, _, _ = igl.remove_unreferenced(v, f[mask])
    return new_v, new_f


def decimate_mesh(v, f, budget=5000):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(v, f)
    ms.add_mesh(m, "my_mesh")
    ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=budget,
                    preserveboundary=True, preservenormal=True, preservetopology=True, planarquadric=True)
    # ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=budget,
    #                 preserveboundary=True, preservenormal=True, preservetopology=True)
    new_v = ms.current_mesh().vertex_matrix()
    new_f = ms.current_mesh().face_matrix()
    ms.clear()
    return new_v, new_f


def get_non_manifold(v, f):
    import pymeshlab
    import pymeshlab
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(v, f)
    ms.add_mesh(m, "my_mesh")
    non_manifold_vertices = ms.select_non_manifold_vertices()
    non_manifold_edges = ms.select_non_manifold_edges()
    return non_manifold_vertices, non_manifold_edges


def resolve_non_manifold_vertices(v, f, save_dest=None):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    dest = './tmp.obj'
    if save_dest is not None:
        dest = save_dest
    igl.write_obj(dest, v, f)
    ms.load_new_mesh(dest)
    ms.repair_non_manifold_vertices_by_splitting()
    ms.save_current_mesh(dest)
    v, _, _, f, _, _ = igl.read_obj(dest)
    if save_dest is None:
        os.remove(dest)
    return v, f


def remove_duplicate_faces(
        faces: torch.Tensor  # F,3
):
    """remove duplicate faces"""
    # todo: return map of removed faces
    if type(faces) != np.ndarray:
        new_faces = faces.detach().cpu().numpy()
    else:
        new_faces = faces
    new_faces = np.sort(new_faces, axis=1)
    new_faces, indices = np.unique(new_faces, axis=0, return_index=True)
    if type(faces) != np.ndarray:
        new_faces = torch.tensor(new_faces, dtype=faces.dtype, device=faces.device)
    return new_faces, indices


def duplicate_faces(
        faces,  # F,3
):
    """creates a twin face for each face in mesh, with flipped orientation"""
    if type(faces) != np.ndarray:
        new_faces = faces.detach().cpu().numpy()
    else:
        new_faces = faces
    swapped_f = new_faces.copy()
    swapped_f[:, [1, 2]] = swapped_f[:, [2, 1]]
    new_f = np.concatenate([new_faces, swapped_f])
    if type(faces) != np.ndarray:
        new_f = torch.tensor(new_f, dtype=faces.dtype, device=faces.device)
    return new_f


def remove_small_disconnected_components(v, f, thres, sampled_point_cloud=None):
    adj_mat = igl.adjacency_matrix(f)
    # remove all small connected components and collapse small triangles
    cc_amount, cc_verts, cc_amounts = igl.connected_components(adj_mat)
    if cc_amount > 1:
        keep_cc = np.zeros_like(cc_amounts, dtype=np.bool)
        keep_cc[np.argmax(cc_amounts)] = True
        if sampled_point_cloud is None:
            cc_amount_normalized = cc_amounts / np.max(cc_amounts)
            keep_cc = cc_amount_normalized > thres
        else:
            # todo: keep largest connected components
            # todo: calculate loss with sampled point cloud
            for i in range(keep_cc.shape[0]):
                if i == np.argmax(cc_amounts):
                    continue
                num_samples = 50000
                v_torch = torch.tensor(v[cc_verts == i], device=sampled_point_cloud.device, dtype=torch.float)
                # #todo: get faces that share these vertices
                # f_torch = torch.tensor(f, device=sampled_point_cloud.device, dtype=torch.long)
                # sampled_cc_points, _ = sample_points_from_meshes(v_torch.unsqueeze(0), f_torch.unsqueeze(0),
                #                                                  num_samples=num_samples)
                loss = torch.norm(knn_points(v_torch.unsqueeze(0), sampled_point_cloud.unsqueeze(0)).dists) / \
                       v_torch.shape[0]
                if loss.item() < 0.03:
                    keep_cc[i] = True
            print('here')
            pass
        for i in range(keep_cc.shape[0]):
            if keep_cc[i]:
                continue
            v[cc_verts == i] = 2
        f = igl.collapse_small_triangles(v, f, eps=1e-10)
        v, f, _, _ = igl.remove_unreferenced(v, f)
    return v, f


def to_numpy(*args):
    def convert(a):
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        assert a is None or isinstance(a, np.ndarray)
        return a

    return convert(args[0]) if len(args) == 1 else tuple(convert(a) for a in args)


def save_mesh_properly(
        vertices: torch.Tensor,
        faces: torch.Tensor,
        filename
):
    filename = Path(filename)
    if filename.suffix not in [".obj", ".ply"]:
        raise ValueError("Only .obj and .ply are supported")
    if (faces < 0).any().item():
        raise ValueError("Faces must be positive")
    if torch.isnan(vertices).any().item():
        raise ValueError("Vertices must be finite")
    if torch.isnan(faces).any().item():
        raise ValueError("Faces must be finite")
    if vertices.dtype != torch.float32:
        raise ValueError("Vertices must be of type float32")
    if faces.dtype != torch.int64:
        raise ValueError("Faces must be of type int64")
    igl.write_obj(str(filename), vertices.detach().cpu().numpy(), faces.detach().cpu().numpy())


def save_mesh(
        vertices: torch.Tensor,
        faces: torch.Tensor,
        filename: Path,
        indexing=1
):
    filename = Path(filename)

    bytes_io = io.BytesIO()
    np.savetxt(bytes_io, vertices.detach().cpu().numpy(), 'v %.4f %.4f %.4f')
    # indexing = 1 - 1-based indexing
    # indexing = 0 - 0-based indexing
    np.savetxt(bytes_io, faces.cpu().numpy() + indexing, 'f %d %d %d')

    obj_path = filename.with_suffix('.obj')
    with open(obj_path, 'w') as file:
        file.write(bytes_io.getvalue().decode('UTF-8'))


def average_image_color(image):
    h = image.histogram()

    # split into red, green, blue
    r = h[0:256]
    g = h[256:256 * 2]
    b = h[256 * 2: 256 * 3]

    # perform the weighted average of each channel:
    # the *index* is the channel value, and the *value* is its weight
    return (
        sum(i * w for i, w in enumerate(r)) / sum(r),
        sum(i * w for i, w in enumerate(g)) / sum(g),
        sum(i * w for i, w in enumerate(b)) / sum(b)
    )


def load_obj_with_colors( # problematic - may change vertex order
        # todo: add support for texture rendering
        filename: Path,
        device='cuda'):
    v = []
    f = []
    c_real = []
    c_id = []
    uv = []
    uv_id = []
    tex = []
    tex_id = []
    uv_face_ids = []
    meshes = trimesh.load_mesh(Path(filename))
    try:
        meshes = trimesh.load_mesh(Path(filename)).dump()
    except:
        meshes = [meshes]
    total_vertex_num = 0
    total_face_num = 0
    current_tex_id = 0
    # todo: find the amount of colors
    np.arange(len(meshes)+1)
    for i, mesh in enumerate(meshes):
        # if i!=6:
        #     continue
        v.append(mesh.vertices)
        f.append(mesh.faces + total_vertex_num)
        try:
            if mesh.visual.material.image is not None:
                #c.append(np.array([average_image_color(mesh.visual.material.image)]) / 255.0)
                c_real.append(np.array([average_image_color(mesh.visual.material.image)]) / 255.0)
                tex.append(np.array(mesh.visual.material.image) / 255.0)
                tex_id.append(np.ones(mesh.vertices.shape[0]) * current_tex_id)
                uv_face_ids.append(np.arange(mesh.faces.shape[0]) + total_face_num)
                uv_id.append(mesh.faces)
                uv.append(mesh.visual.uv)
                current_tex_id += 1
            else:
                c_real.append(np.ones([1, 3]) * mesh.visual.material.diffuse[:3] / 255.0)
                uv_id.append(None)
                uv.append(None)
                uv_face_ids.append(None)
                tex.append(None)
        except:
            c_real.append(np.ones([1, 3]))
            tex.append(None)
            uv_face_ids.append(None)
            uv.append(None)
        c_id.append(np.ones(mesh.faces.shape[0]) * (i+1))
        total_vertex_num += mesh.vertices.shape[0]
        total_face_num += mesh.faces.shape[0]
    c = (np.expand_dims((np.arange(len(meshes)+1)) / len(meshes), axis=-1) * np.ones([len(meshes)+1, 3]))
    v = np.concatenate(v, axis=0)
    f = np.concatenate(f, axis=0)
    # uv = np.concatenate(uv, axis=0)
    # uv_id = np.concatenate(uv_id, axis=0)
    # tex_id = np.concatenate(tex_id, axis=0)
    # SV, SVI, SVJ, SF = igl.remove_duplicate_vertices(v, f, 1e-7)
    # SV = torch.tensor(SV, device=device, dtype=torch.float)
    # map old vertex ids to new ones, after duplicate removal
    # SF = SVJ[f]
    # SF = torch.tensor(SF, device=device, dtype=torch.long)
    v = torch.tensor(v, device=device, dtype=torch.float)
    f = torch.tensor(f, device=device, dtype=torch.long)
    c_real = torch.tensor(np.concatenate(c_real, axis=0), device=device, dtype=torch.float)
    c = torch.tensor(c, device=device, dtype=torch.float)
    c_id = torch.tensor(np.concatenate(c_id, axis=0), device=device, dtype=torch.long)
    # todo: need to convert uv_id to face based ids
    return v, f, c, c_id, uv_id, uv, tex_id, uv_face_ids, tex, c_real


def load_mesh(
        filename: Path,
        device='cuda',
        repair_dup_faces=False,
        with_face_colors=False
):
    with redirect_stderr(os.devnull):
        # vertices, faces = igl.read_triangle_mesh(str(filename))
        if filename.suffix == '.obj':
            if with_face_colors:
                vertices, faces, face_colors, c_id, uv_id, uv, tex_id, uv_face_ids, tex, c_real = \
                    load_obj_with_colors(Path(filename), device=device)
            else:
                vertices,_, _, faces, _, _ = igl.read_obj(str(filename))
        elif filename.suffix == '.ply':
            with open(filename, 'rb') as f:
                mesh = trimesh.exchange.ply.load_ply(f, prefer_color='face')
                vertices = torch.tensor(mesh['vertices'], device=device)
                faces = torch.tensor(mesh['faces'].copy(), device=device, dtype=torch.long)
                face_colors = torch.tensor(mesh['face_colors'][:, :3], device=device, dtype=torch.float) / 255.0
                c_id = None
        else:
            raise ValueError('ERROR: Invalid mesh suffix <{}> (.obj or .ply currently supported)'.format(filename.suffix))
    valid = faces.size != 0
    if repair_dup_faces:
        faces, indices = remove_duplicate_faces(faces)
        faces = duplicate_faces(faces)
        if with_face_colors:
            c_id = c_id[indices]
            c_id = c_id.repeat(repeats=[2])
    vertices = torch.tensor(vertices, dtype=torch.float, device=device)
    if valid:
        faces = torch.tensor(faces, dtype=torch.long, device=device)
    else:
        faces = None
    if with_face_colors:
        return vertices, faces, face_colors, c_id, valid
    else:
        return vertices, faces, None, None, valid


def load_mesh_o3d(path, to_torch=False, device='cuda:1'):
    mesh = o3d.io.read_triangle_mesh(str(path))
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    if to_torch:
        v = torch.tensor(v, dtype=torch.float32, device=device)
        f = torch.tensor(f, dtype=torch.int64, device=device)
    return v, f


def save_pc(
        filename: Path,
        points: torch.Tensor,
        normals: torch.Tensor = None,
        features: torch.Tensor = None,
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals.cpu().numpy())
    if features is not None:
        pcd.colors = o3d.utility.Vector3dVector(features.cpu().numpy())
    o3d.io.write_point_cloud(str(filename), pcd)
    # pc = Pointclouds(points=[points], normals=[normals], features=features)
    # IO().save_pointcloud(pc, filename, binary=False)


def load_pc(
        filename: Path,
        device='cuda',
        to_torch=True
):
    pcd = o3d.io.read_point_cloud(str(filename))  # Read the point cloud
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    features = np.asarray(pcd.colors)
    if to_torch:
        points = torch.tensor(points, dtype=torch.float32, device=device)
        normals = torch.tensor(normals, dtype=torch.float32, device=device)
        if features.size != 0:
            features = torch.tensor(features, dtype=torch.float32, device=device)
        else:
            features = None
    return points, normals, features, True


def save_ply(
        filename: Path,
        vertices: torch.Tensor,  # V,3
        faces: torch.Tensor,  # F,3
        vertex_colors: torch.Tensor = None,  # V,3
        vertex_normals: torch.Tensor = None,  # V,3
        face_colors: torch.Tensor = None,  # F,1
):
    filename = Path(filename).with_suffix('.ply')
    vertices, faces, vertex_colors, vertex_normals, face_colors = to_numpy(vertices, faces, vertex_colors,
                                                                           vertex_normals, face_colors)
    assert np.all(np.isfinite(vertices)) and faces.min() == 0 and faces.max() == vertices.shape[0] - 1

    header = 'ply\nformat ascii 1.0\n'

    header += 'element vertex ' + str(vertices.shape[0]) + '\n'
    header += 'property double x\n'
    header += 'property double y\n'
    header += 'property double z\n'

    if vertex_normals is not None:
        header += 'property double nx\n'
        header += 'property double ny\n'
        header += 'property double nz\n'

    if vertex_colors is not None:
        assert vertex_colors.shape[0] == vertices.shape[0]
        color = (vertex_colors * 255).astype(np.uint8)
        header += 'property uchar red\n'
        header += 'property uchar green\n'
        header += 'property uchar blue\n'

    header += 'element face ' + str(faces.shape[0]) + '\n'
    header += 'property list int int vertex_indices\n'
    if face_colors is not None:
        assert face_colors.shape[0] == faces.shape[0]
        fcolor = (face_colors * 255).astype(np.uint8)
        header += 'property uchar red\n'
        header += 'property uchar green\n'
        header += 'property uchar blue\n'
    header += 'end_header\n'

    with open(filename, 'w') as file:
        file.write(header)

        for i in range(vertices.shape[0]):
            s = f"{vertices[i, 0]} {vertices[i, 1]} {vertices[i, 2]}"
            if vertex_normals is not None:
                s += f" {vertex_normals[i, 0]} {vertex_normals[i, 1]} {vertex_normals[i, 2]}"
            if vertex_colors is not None:
                s += f" {color[i, 0]:03d} {color[i, 1]:03d} {color[i, 2]:03d}"
            file.write(s + '\n')

        for i in range(faces.shape[0]):
            s = f"3 {faces[i, 0]} {faces[i, 1]} {faces[i, 2]}"
            if face_colors is not None:
                s += f" {fcolor[i, 0]:03d} {fcolor[i, 1]:03d} {fcolor[i, 2]:03d}"
            file.write(s + '\n')
    full_verts = vertices[faces]  # F,3,3


def load_images(
        dir: Path,
        device
):

    def get_key(fp):
        filename = os.path.splitext(os.path.basename(fp))[0]
        int_part = filename.split('_')[0]
        return int(int_part)

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    imgs = []
    img_names = sorted(dir.glob('*[0-9]*.png'), key=get_key)
    for img in img_names: #dir.glob('*.png'):
        imgs.append(torch.tensor(imageio.imread(img), device=device, dtype=torch.float)/255)
    return torch.stack(imgs)


def save_images(
        images: torch.Tensor,  # B,H,W,CH
        dir: Path,
):
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    for i in range(images.shape[0]):
        imageio.imwrite(dir / (f'{i:06d}.png'),
                        (images[i, :, :, :] * 255).clamp(max=255).type(torch.uint8).detach().cpu().numpy())


def normalize_vertices(
        vertices: torch.Tensor,  # V,3
        scale_by = 'max'
):  
    """shift and resize mesh to fit into a unit sphere"""
    if type(vertices) is np.ndarray:
        vertices = torch.tensor(vertices, dtype=torch.float32)
    vertices -= (vertices.min(dim=0)[0] + vertices.max(dim=0)[0]) / 2
    if scale_by == 'max':
        vertices /= torch.norm(vertices, dim=-1).max()
    if scale_by == 'norm':
        vertices /= torch.norm(vertices, dim=-1).norm()
    return vertices


def laplacian(
        num_verts: int,
        edges: torch.Tensor  # E,2
) -> torch.Tensor:  # sparse V,V
    """create sparse Laplacian matrix"""
    V = num_verts
    E = edges.shape[0]

    # adjacency matrix,
    idx = torch.cat([edges, edges.fliplr()], dim=0).type(torch.long).T  # (2, 2*E)
    ones = torch.ones(2 * E, dtype=torch.float32, device=edges.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # degree matrix
    deg = torch.sparse.sum(A, dim=1).to_dense()
    idx = torch.arange(V, device=edges.device)
    idx = torch.stack([idx, idx], dim=0)
    D = torch.sparse.FloatTensor(idx, deg, (V, V))

    return D - A


def _translation(x, y, z, device):
    return torch.tensor([[1., 0, 0, x],
                         [0, 1, 0, y],
                         [0, 0, 1, z],
                         [0, 0, 0, 1]], device=device)  # 4,4


def perspective_projection(fovx=45, ar=1.0, n=0.1, f=100, device='cpu'):
    fov_rad = np.deg2rad(fovx)
    s = 1.0 / np.tan(fov_rad / 2.0)
    sx, sy = s, s / ar
    zz = (f + n) / (n - f)
    zw = 2 * f * n / (n - f)
    p = np.array([[sx, 0, 0, 0],
                  [0, sy, 0, 0],
                  [0, 0, zz, zw],
                  [0, 0, -1, 0]])
    return torch.tensor(p, dtype=torch.float32, device=device)


def persp_proj(fovx=45, ar=1, near=0.1, far=100, device='cuda'):
    """
    Build a perspective projection matrix.
    Parameters
    ----------
    fov_x : float
        Horizontal field of view (in degrees).
    ar : float
        Aspect ratio (w/h).
    near : float
        Depth of the near plane relative to the camera.
    far : float
        Depth of the far plane relative to the camera.
    """
    fov_rad = np.deg2rad(fovx)
    proj_mat = np.array([[-1.0 / np.tan(fov_rad / 2.0), 0, 0, 0],
                         [0, np.float32(ar) / np.tan(fov_rad / 2.0), 0, 0],
                         [0, 0, -(near + far) / (near - far), 2 * far * near / (near - far)],
                         [0, 0, 1, 0]])
    proj = torch.tensor(proj_mat, device=device, dtype=torch.float32)
    proj[:, :-1] *= -1
    return proj


def orthographic_projection(r, device, l=None, t=None, b=None, n=0.1, f=50.0):
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    dx = r - l
    dy = t - b
    dz = f - n
    rx = -(r + l) / (r - l)
    ry = -(t + b) / (t - b)
    rz = -(f + n) / (f - n)
    proj = np.array([[2.0 / dx, 0, 0, rx],
                     [0, 2.0 / dy, 0, ry],
                     [0, 0, -2.0 / dz, rz],
                     [0, 0, 0, 1]])
    return torch.tensor(proj, device=device, dtype=torch.float32)


def _projection(r, device, l=None, t=None, b=None, n=1.0, f=50.0, flip_y=True):
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    p = torch.zeros([4, 4], device=device)
    p[0, 0] = 2 * n / (r - l)
    p[0, 2] = (r + l) / (r - l)
    p[1, 1] = 2 * n / (t - b) * (-1 if flip_y else 1)
    p[1, 2] = (t + b) / (t - b)
    p[2, 2] = -(f + n) / (f - n)
    p[2, 3] = -(2 * f * n) / (f - n)
    p[3, 2] = -1
    # np.array([[2*n/(r-l), 0, (r+l)/(r-l), 0],
    #           [0, 2*n/(t-b), (t+b)/(t-b), 0],
    #           [0, 0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
    #           [0, 0, -1, 0]])
    return p  # 4,4


def make_grid_on_sphere(n, m, r, device):
    """create a grid of points on a sphere"""
    theta = torch.linspace(0, 2 * np.pi, m + 1, device=device)[:-1]
    phi = torch.linspace(np.pi / 8, +np.pi - np.pi / 8, n, device=device)
    theta, phi = torch.meshgrid(theta, phi)
    theta = theta.flatten()
    phi = phi.flatten()
    x = r * torch.cos(theta) * torch.sin(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(phi)
    return torch.stack([x, y, z], dim=1)


def lookat_transform(
        eye: torch.Tensor,  # 3
        at: torch.Tensor,  # 3
        up: torch.Tensor,  # 3
        device: torch.device
) -> torch.Tensor:  # 4,4
    """create a lookat transform matrix"""
    z = -(at - eye).type(torch.float32).to(device)
    z /= torch.norm(z)
    x = torch.cross(up, z).type(torch.float32).to(device)
    x /= torch.norm(x)
    y = torch.cross(z, x).type(torch.float32).to(device)
    y /= torch.norm(y)
    T = torch.eye(4, device=device)
    T[:3, :3] = torch.stack([x, y, z], dim=1)
    T[:3, 3] = eye
    return T


def make_random_cameras(n, r, at=None, upper_hemi=False, device="cuda"):
    if at is None:
        at = torch.zeros(3, dtype=torch.float32, device=device)
    locs = torch.randn((n, 3), device=device)
    if upper_hemi:
        lower_hemi_mask = locs[:, -1] < 0
        locs[lower_hemi_mask, -1] *= -1
    locs = torch.nn.functional.normalize(locs, dim=1, eps=1e-6)
    locs = locs * r
    matrices = torch.empty((n, 4, 4), dtype=torch.float32, device=device)
    for i in range(len(locs)):
        matrices[i] = lookat_transform(locs[i],
                                       at,
                                       torch.tensor([0., 0., 1.], device=device),
                                       device=device)
    c2w = matrices  # c2w
    w2c = torch.inverse(c2w)
    return w2c, persp_proj(device=device)


def make_star_cameras_pulsar(n, m, device="cuda"):
    r = 5
    matrices = torch.empty((n * m, 4, 4), dtype=torch.float32, device=device)
    locs = make_grid_on_sphere(n=n, m=m, r=r, device=device)
    for i in range(len(locs)):
        matrices[i] = lookat_transform(locs[i],
                                       torch.zeros(3, dtype=torch.float32, device=device),
                                       torch.tensor([0., 0., 1.], device=device),
                                       device=device)
    c2w = matrices  # c2w
    w2c = torch.inverse(c2w)
    return w2c


def make_star_cameras(az_count, pol_count, distance: float = 10., r=None, image_size=[512, 512], device='cuda',
                      beam_angle = 2*torch.pi, return_elevation_and_azimuth = False):
    
    # TODO: 
    if r is None:
        r = 1 / distance
    A = az_count
    P = pol_count
    C = A * P

    # phi is azimuth angle - rotate around z
    rotate_z = torch.eye(3, device=device)
    rotate_x = torch.eye(3, device=device)
    rotations = []
    for pol_id in range(pol_count):
        for az_id in range(az_count):
            az = az_id/az_count * (2*torch.pi)
            rotate_z[0,0] = np.cos(az)
            rotate_z[0,1] = -np.sin(az)
            rotate_z[1,0] = np.sin(az)
            rotate_z[1,1] = np.cos(az)

            pol = (pol_id)/(pol_count-1) * (torch.pi-40*torch.pi/180) + (1-pol_id/(pol_count-1))*(40*torch.pi/180)
            rotate_x[1,1] = np.cos(-pol)
            rotate_x[1,2] = -np.sin(-pol)
            rotate_x[2,1] = np.sin(-pol)
            rotate_x[2,2] = np.cos(-pol)

            rot = rotate_x @ rotate_z
            rotations.append(rot)
        
    # mv transforms from world space to camera space -> opencv cameras?
    mv = torch.empty((C, 4, 4), device=device)
    mv[:] = torch.eye(4, device=device)
    mv[:, :3, :3] = torch.stack(rotations, dim=0)
    mv =   _translation(0,0,-distance, device) @ mv

    if return_elevation_and_azimuth:
        angles_cartesian_prod = torch.cartesian_prod(phi, theta)
        azimuth = angles_cartesian_prod[:,0]
        elevation = angles_cartesian_prod[:,1]
        return mv, _projection(r, device), elevation, azimuth
    return mv, _projection(r, device)


# todo: function that mvp from given directions 
# todo: turn directions in to azimuth and polar matrices
def make_cameras_from_dirs(camera_locs, distance: float = 10., r=None, image_size=[512, 512], device='cuda', return_elevation_and_azimuth = False):
    if r is None:
        r = 1 / distance
    N = camera_locs.shape[0]

    phi, theta = calculate_elevation_and_azimuth(camera_locs)

    # phi is azimuth angles
    phi_rot = torch.eye(3, device=device)[None].expand(N, 3, 3).clone()
    # phi_rot[:, 0, 1, 1] = phi.cos()
    # phi_rot[:, 0, 1, 0] = -phi.sin()
    # phi_rot[:, 0, 0, 1] = phi.sin()
    # phi_rot[:, 0, 0, 0] = phi.cos()
    phi_rot[:, 2, 2] = phi.cos()
    phi_rot[:, 2, 0] = -phi.sin()
    phi_rot[:, 0, 2] = phi.sin()
    phi_rot[:, 0, 0] = phi.cos()
    # theta is polar angle
    theta_rot = torch.eye(3, device=device)[None].expand(N, 3, 3).clone()
    # theta_rot[0, :, 0, 0] = theta.sin()
    # theta_rot[0, :, 0, 2] = -theta.cos()
    # theta_rot[0, :, 2, 0] = theta.cos()
    # theta_rot[0, :, 2, 2] = theta.sin()
    theta_rot[:, 1, 1] = theta.cos()
    theta_rot[:, 1, 2] = -theta.sin()
    theta_rot[:, 2, 1] = theta.sin()
    theta_rot[:, 2, 2] = theta.cos()

    mv = torch.empty((N, 4, 4), device=device)
    mv[:] = torch.eye(4, device=device)
    mv[:, :3, :3] = (theta_rot @ phi_rot).reshape(N, 3, 3)
    mv = _translation(0, 0, -distance, device) @ mv

    if return_elevation_and_azimuth:
        angles_cartesian_prod = torch.cartesian_prod(phi, theta)
        azimuth = angles_cartesian_prod[:,0]
        elevation = angles_cartesian_prod[:,1]
        return mv, _projection(r, device), elevation, azimuth
    return mv, _projection(r, device)


def make_zoom_cameras(n, device='cuda'):
    r = 0.0008
    projections = []
    for i in range(n):
        projections.append(_projection(r[i], device))
    return torch.stack(projections)


def make_sphere(level: int = 2, radius=1., device='cuda'):
    sphere = trimesh.creation.icosphere(subdivisions=level, radius=1.0, color=None)
    vertices = torch.tensor(sphere.vertices, device=device, dtype=torch.float32) * radius
    faces = torch.tensor(sphere.faces, device=device, dtype=torch.long)
    return vertices, faces


def merge_dicts(main_dict: dict, dict_to_merge: dict, prefix: str = ""):
    for k, v in dict_to_merge.items():
        main_dict[prefix + str(k)] = v


def download_file(file_id, output_dir):
    file_path1 = Path(output_dir, str(file_id) + ".stl")
    file_path2 = Path(output_dir, str(file_id) + ".obj")
    if file_path1.is_file() or file_path2.is_file():
        print("Skipping {} because it already exists.".format(file_id))
        return
    url = "https://www.thingiverse.com/download:{}".format(file_id)
    r = requests.head(url)
    link = r.headers.get("Location", None)
    if link is None:
        print("File {} is no longer available on Thingiverse.".format(file_id))
        return
    if ".stl" in link or ".STL" in link:
        suffix = ".stl"
    else:
        suffix = ".obj"
    file_path = Path(output_dir, str(file_id) + suffix)
    command = "wget -q -O {} --tries=10 {}".format(str(file_path), link)
    check_call(command.split())
    print("Downloaded file to {}".format(str(file_path)))


def rotate_using_quaternion(q, v):
    q = q / torch.norm(q)
    q = torch.cat([q[:, 3:], q[:, :3]], dim=1)
    return torch.einsum("ij,ajk->aik", q, torch.einsum("ajk,ak->aj", v, q))


def q_mult(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.stack([w, x, y, z], axis=1)


def qv_mult(q1, v1):
    q2 = np.concatenate((np.zeros((v1.shape[0], 1)), v1), axis=-1)
    q1 = q1[None, :]
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]


def q_conjugate(q):
    q_conj = np.concatenate((q[:, :1], -q[:, 1:]), axis=-1)
    return q_conj


def merge_meshes(v1, f1, v2, f2):
    """merge two meshes into one"""
    if v2 is None or f2 is None:
        return v1, f1
    v = np.concatenate([v1, v2], axis=0)
    f = np.concatenate([f1, f2 + v1.shape[0]], axis=0)
    return v, f


def quaternion_rotation_matrix(Q):
    # Extract the values from Q
    q0 = Q[3]
    q1 = Q[0]
    q2 = Q[1]
    q3 = Q[2]

    # calculate unit quarternion
    magnitude = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)

    q0 = q0 / magnitude
    q1 = q1 / magnitude
    q2 = q2 / magnitude
    q3 = q3 / magnitude

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    rot_matrix = np.transpose(rot_matrix)

    return rot_matrix


def get_transformations_from_file(file_path):
    with open(file_path) as f:
        filenames = []
        translations = []
        quaternions = []
        for line in f:
            info = line.strip().split()
            if info:
                if info[0] == "bmesh":
                    filenames.append(info[1])
                    translations.append(np.array(info[2:5]).astype(np.float32).reshape(3, 1))
                    quaternions.append(np.array(info[5:]).astype(np.float32).reshape(4, 1))
                elif info[0] == "camera":
                    filenames.append(info[0])
                    translations.append(np.array(info[1:4]).astype(np.float32).reshape(3, 1))
                    quaternions.append(np.array(info[4:]).astype(np.float32).reshape(4, 1))
    return filenames, translations, quaternions


def remove_triangles_with_large_edges(v, f, threshold=0.1):
    """
    remove triangles with edges larger than 0.1
    """
    v1 = v[f[:, 0]]
    v2 = v[f[:, 1]]
    v3 = v[f[:, 2]]
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v3 - v1, axis=1)
    e3 = np.linalg.norm(v3 - v2, axis=1)
    mask = (e1 < threshold) & (e2 < threshold) & (e3 < threshold)
    new_v, new_f, _, _ = igl.remove_unreferenced(v, f[mask])
    return new_v, new_f


def stanford_preprocess(root_folder, output_folder):
    import pymeshlab
    """
    """
    # root_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)
    paths = []
    w2cs = []
    names = []
    renders = []
    device = "cuda:0"
    required_names = ["armadillo"]  # "bunny", "drill", "armadillo"]
    for folder in sorted(root_folder.glob("*")):
        if not folder.is_dir():
            continue
        if folder.name not in required_names:
            continue
        names.append(folder.name)
        data_folder = Path(folder, "data")
        if not data_folder.exists():
            data_folder = Path(folder)
        exp_folder = Path(output_folder, folder.name)
        result_path = Path(exp_folder, "target_images")
        result_path.mkdir(parents=True, exist_ok=True)
        conf_file = sorted(data_folder.glob("*.conf"))[0]
        filenames, translations, quaternions = get_transformations_from_file(conf_file)
        cam_t = translations[0].squeeze()
        # cam_t = np.array([0, 0, -0.7])
        cam_q = quaternions[0]
        # cam_R = quaternion_rotation_matrix(cam_q).squeeze()
        cam_R = np.eye(3)
        cam_Rt = np.concatenate((cam_R, cam_t[:, None] * 75 / 100), axis=-1)
        c2o = np.concatenate((cam_Rt, np.array([0, 0, 0, 1])[None, :]), axis=0)
        translations = translations[1:]
        quaternions = quaternions[1:]
        filenames = filenames[1:]
        for i, patch_file in enumerate(filenames):
            src = Path(data_folder, patch_file)
            dst = Path(data_folder, Path(patch_file).stem + ".obj")
            if not dst.is_file():
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(str(src))
                ms.save_current_mesh(str(dst))
                ms.clear()
        o2w = np.zeros([3, 4])
        vertices = list()
        faces = list()
        clouds = list()
        curr_c2w = []
        for curr_filename, curr_quaternion, curr_translation in zip(filenames, quaternions,
                                                                    translations):  # go through input files
            file_path = Path(data_folder, curr_filename)
            curr_cloud = o3d.io.read_point_cloud(str(file_path))
            curr_v, curr_f = igl.read_triangle_mesh(str(file_path).replace(".ply", ".obj"))
            curr_v, curr_f = remove_triangles_with_large_edges(curr_v, curr_f, 0.003)
            # convert cloud to numpy
            curr_cloud = np.asarray(curr_cloud.points)

            # compute rotation matrix from quaternions
            curr_rotation_matr = quaternion_rotation_matrix(curr_quaternion)
            curr_rotation_matr = np.squeeze(curr_rotation_matr)
            curr_translation = np.squeeze(curr_translation)
            # create transformation matrix
            o2w[:, 0:3] = curr_rotation_matr
            o2w[:, 3] = curr_translation
            faces.append(curr_f)
            new_cloud = np.concatenate((curr_cloud, np.ones((curr_cloud.shape[0], 1))), axis=-1)
            clouds.append((o2w @ new_cloud.T).T)
            # vertices.append(curr_v)
            new_v = np.concatenate((curr_v, np.ones((curr_v.shape[0], 1))), axis=-1)
            vertices.append((o2w @ new_v.T).T)
            o2w_44 = np.concatenate((o2w, np.array([[0, 0, 0, 1]])), axis=0)
            curr_c2w.append(o2w_44 @ np.linalg.inv(c2o))
        final_cloud = torch.tensor(np.concatenate(clouds, axis=0), dtype=torch.float32, device=device)
        centroid = final_cloud.mean(dim=0).cpu().numpy()
        final_cloud = final_cloud - torch.tensor(centroid, device=device)

        total_v = None
        total_f = None
        for v, f in zip(vertices, faces):
            total_v, total_f = merge_meshes(v, f, total_v, total_f)
        total_v, total_f, _, _ = igl.remove_unreferenced(total_v, total_f)
        total_v -= total_v.mean(axis=0, keepdims=True)
        megamesh_path = Path(data_folder, "megamesh.obj")
        igl.write_obj(str(megamesh_path), total_v, total_f)
        paths.append(megamesh_path)
    return names, paths


def thingi10k_preprocess(root_folder, output_folder, renderer, n_points=200000):
    """
    preprocess thingi10k meshes into output folder.
    mesh_class must be valid shapenetcore class
    n is number of meshes to extract from the class (randomly chosen)
    """
    root_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)
    # old query:
    # #be = 0, #c = 1, genus = 0, #f < 50000, with self-intersection
    # new query:
    # is not oriented, with degeneracy, tag=architecture
    file_ids = [41089, 1313538, 41091, 41092, 129685, 68254, 79136, 67891, 67893, 78533, 229960, 41308, 1037028,
                1038440, 41705, 67050, 206317, 206318, 206319, 47089, 41077, 47094, 41079, 41080, 41082, 41083, 41085]
    for i, file_id in enumerate(file_ids):
        print("downloading: {} / {}".format(i, len(file_ids)))
        download_file(file_id, root_folder)
    instances = np.array([x for x in sorted(root_folder.glob("*.stl"))])
    for i, instance in enumerate(instances):
        instance_name = instance.stem
        dst_pc = Path(output_folder, "{}.ply".format(instance_name))
        dst_mesh = Path(output_folder, "{}.obj".format(instance_name))
        if dst_pc.is_file() and dst_mesh.is_file():
            print("skipping {}, {} / {}".format(instance.stem, i, len(instances)))
            continue
        print("cleaning {}, {} / {}".format(instance.stem, i, len(instances)))
        mesh_vertices, mesh_faces, valid_mesh = load_mesh(instance, repair_dup_faces=True)
        assert valid_mesh
        mesh_vertices = normalize_vertices(mesh_vertices)
        mesh_vertices, mesh_faces = fix_mesh(renderer, mesh_vertices, mesh_faces)
        mesh_vertices = normalize_vertices(mesh_vertices)
        pc_vertices, pc_normals = triangle_soup_to_point_cloud_random(mesh_vertices, mesh_faces, n_points)
        save_pc(dst_pc, pc_vertices, pc_normals)
        save_mesh(mesh_vertices, mesh_faces, dst_mesh)


def fix_mesh(renderer, v, f, raycaster=None, c_id=None):
    """
    fixes a mesh using multi-view elimination. assumes mesh has every face has also its flipped included.
    """
    n = calculate_vertex_normals(v, f)
    _, extra = renderer.render(v, n, f, get_visible_faces=True, cull_bf=True)
    visible_faces_ids, raw_counts = extra["vis_faces"], extra["counts"]
    # find which of each twin face appears more
    # FF, IA, IC = igl.unique_simplices(f.detach().cpu().numpy())
    # face_pairs = np.argsort(IA).reshape(-1, 2)
    # face_pairs = np.arange(f.shape[0]).reshape(-1, 2)
    face_pairs = np.stack([np.arange(f.shape[0] // 2), np.arange(f.shape[0] // 2) + f.shape[0] // 2], axis=-1)
    # np.concatenate(np.arange(f.shape[0]), np.arange(f.shape[0])+f.shape[0])
    visbility_counter = np.zeros(len(f), dtype=np.int64)
    visbility_counter[visible_faces_ids.detach().cpu().numpy()] = raw_counts.detach().cpu().numpy()
    pair_vis_counts = visbility_counter[face_pairs]
    mask1 = pair_vis_counts == pair_vis_counts.max(axis=-1,
                                                   keepdims=True)  # take twin with more views (if same, take both)

    mask4 = ((pair_vis_counts.max(axis=-1) / (pair_vis_counts.min(axis=-1) + 1)) < 80)[:, None]
    # keep undecided faces
    final_mask = mask1 | np.repeat(mask4, 2, axis=-1)

    if raycaster is not None:
        undecided_faces = face_pairs[np.repeat(mask4, 2, axis=-1) | ~mask1]
        # apply embree raycasting decision criterion
        final_mask = mask1 & ~np.repeat(mask4, 2, axis=-1)
        undecided_face_ray_ratio = raycaster.ray_ratio_face_visibilty(v, f[undecided_faces])
        final_mask[np.repeat(mask4, 2, axis=-1) | ~mask1] = (undecided_face_ray_ratio > 0.2).cpu().numpy()

    fixed_visible_faces_ids = face_pairs[final_mask]
    visible_faces = f[fixed_visible_faces_ids]
    if c_id is not None:
        c_id = c_id[fixed_visible_faces_ids]
    new_v, new_f, _, _ = igl.remove_unreferenced(v.cpu().numpy(), visible_faces.cpu().numpy())
    # todo: fix c_id after remove_unreferenced
    new_v = torch.tensor(new_v, device=v.device, dtype=v.dtype)
    new_f = torch.tensor(new_f, device=f.device, dtype=f.dtype)
    return new_v, new_f, c_id


# a context manager that can launch a function with a timelimit (throws TimeoutException if it times out)
class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def save_zipped_pickle(obj, filename, protocol=-1):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)


def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object


def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

