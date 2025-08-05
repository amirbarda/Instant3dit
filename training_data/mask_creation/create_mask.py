import torch
import random
from scipy.spatial import ConvexHull
import numpy as np

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



def sample_random_points_and_normals(initial_point, sx=1,sy=1,sz=1, N = 10):
    # TODO: add twist angle (norm != 1)?
    # Create random polyhedron from random points
    points = (torch.rand([N,3], device='cuda')-0.5)
    points[:,0] *= sx
    points[:,1] *= sy
    points[:,2] *= sz
    points += initial_point
    normals = torch.rand([N,3], device = 'cuda')
    normals /= torch.norm(normals,dim=-1,keepdim=True)
    return points,normals

def normalize_scene(v):
    bbox_max= v.max(dim=0)[0]
    bbox_min =v.min(dim=0)[0]
    scale = 1.8/(bbox_max - bbox_min).max()
    offset = -(bbox_max + bbox_min) / 2
    v += offset
    v *= scale
    return v

# uses ellipsoid coordinates from https://gssc.esa.int/navipedia/index.php/Ellipsoidal_and_Cartesian_Coordinates_Conversion#:~:text=The%20(%20x%20%2C%20y%20%2C%20z,between%20Cartesian%20and%20ellipsoidal%20coordinates.
# same as spherical: just add a scaling parameter to each of the coordinates (x,y,z)
# fast vectorized implmenetation as this needs to be done after loading the data
def create_ellipsoid_meshes(a = 0.2,b = 0.2,c = 0.2, n_polar = 20, n_az = 20): # works

    # todo: create elispoied mesh using ellipsoid coordinates
    phi = (torch.arange(-n_polar//2+1, n_polar//2+1, device='cuda') * torch.pi/(n_polar+1)).unsqueeze(-1)

    # theta is polar angle - rotate around z
    theta = (torch.arange(0, n_az, device='cuda') * 2*torch.pi/(n_az)).unsqueeze(0)
    x = a*phi.cos()*theta.cos()
    y = b*phi.cos()*theta.sin()
    z = c*phi.sin().expand([-1,n_az])
    ids = torch.arange(n_polar*n_az)
    v0 = ids[:-n_az]
    v1 = v0+n_az
    v2 = (v0+n_az).reshape([n_az-1, n_polar]).roll(1, dims=-1).flatten()
    v3 = (v0).reshape([n_az-1, n_polar]).roll(1, dims=-1).flatten()
    
    v =  torch.stack([x,y,z], dim=-1).reshape([-1,3])
    f = torch.cat([torch.stack([v0,v1,v2], dim=-1), torch.stack([v2,v3,v0], dim=-1)], dim=0)
    # todo: get end cap faces 
    return v,f

def create_cylinderical_meshes(n_polar = 20): # works

    # todo: create elispoied mesh using ellipsoid coordinates
    phi = (torch.arange(n_polar, device='cuda') * 2*torch.pi/(n_polar)).unsqueeze(-1)
    # create circle with n_polar vertices
    x = phi.cos()
    y = phi.sin()
    z = torch.zeros_like(x)
    v_lower  = torch.stack([x,y,z-1], dim=-1).reshape([-1,3])
    v_upper  = torch.stack([x,y,z+1], dim=-1).reshape([-1,3])
    v = torch.cat([v_lower,v_upper],dim=0)
    ids = torch.arange(n_polar)
    wall_faces = torch.cat([torch.stack([ids+n_polar,ids, (ids+1) % n_polar],dim=1),torch.stack([(ids+1) % n_polar, (ids+1) % n_polar + n_polar, ids+n_polar], dim=-1)],dim=0)
    cap_ids = ids[2:]
    cap_faces = torch.cat([torch.stack([cap_ids*0, cap_ids-1, cap_ids],dim=1), torch.stack([cap_ids*0+n_polar, cap_ids-1+n_polar, cap_ids+n_polar],dim=1)], dim=0)
    f = torch.cat([cap_faces, wall_faces], dim=0)
    return v,f

def create_3d_blob(N = 6,a = 0.2,b = 0.2,c = 0.2,sx = 1,sy = 1,sz =1, initial_point = None, return_blobs_as_list = False):
    if initial_point is None:
        initial_point = torch.zeros([1,3], dtype=torch.float, device = 'cuda')
    v_all = []
    f_all = []
    f_blobs = []
    up = torch.tensor([0,0,1], dtype=torch.float, device= 'cuda').unsqueeze(0)
    locs, normals = sample_random_points_and_normals(initial_point, sx,sy,sz,N=N)
    rotation_axis = up.cross(normals, dim=-1)
    rotation_axis /= torch.norm(rotation_axis, dim=-1, keepdim=True)
    angles = rotation_angle_between_two_vectors_and_axis(up, normals, rotation_axis)
    #v,f = create_ellipsoid_meshes(a,b,c)
    cylinders_info = {}
    cylinders_info['scale'] = []
    for i in range(N):    
        v,f = create_cylinderical_meshes()
        multiplier = torch.rand([1,3], device='cuda').clip(min = 0.01)
        v *= multiplier
        v[:,0] *=a
        v[:,1] *= b
        v[:,2] *= c
        v_rotated = rotate_vectors_by_angles_around_axis(v, angles[[i]], rotation_axis[[i]])
        v_rotated += locs[[i]]
        cylinders_info['scale'].append([a*multiplier[0,0],b*multiplier[0,1],c*multiplier[0,2]])
        v_all.append(v_rotated)
        f_blobs.append(f)
        f_all.append(f+i*v.shape[0])
    v_blob = torch.cat(v_all,dim=0).to('cuda')
    f_blob = torch.cat(f_all, dim=0).to('cuda')
    cylinders_info['locations'] = locs
    cylinders_info['normals'] = normals
    if return_blobs_as_list:
        return v_blob, f_blob, cylinders_info, v_all,f_blobs
    else:
        return v_blob, f_blob, cylinders_info 

def is_point_inside_cylinder(points, cylinders_info):
    # tranform all vertices so that z axis of cylinder is up, and center is at origin
    # mask all coordinates which are inside the cylinder
    normals = cylinders_info['normals']
    locations = cylinders_info['locations']
    is_inside = torch.zeros(points.shape[0], dtype=bool, device = points.device)
    up = torch.tensor([0,0,1], dtype=torch.float, device= 'cuda').unsqueeze(0)
    rotation_axis = up.cross(normals, dim=-1)
    rotation_axis /= torch.norm(rotation_axis, dim=-1, keepdim=True)
    angles = rotation_angle_between_two_vectors_and_axis(normals, up, rotation_axis)
    # for all cylinders
    for i in range(normals.shape[0]):
        a,b,c = cylinders_info['scale'][i]
        points_translated = points - locations[i]
        points_rotated = rotate_vectors_by_angles_around_axis(points_translated, angles[[i]], rotation_axis[[i]])
        points_rotated[:,0] /= a
        points_rotated[:,1] /= b
        points_rotated[:,2] /= c
        is_inside |= (points_rotated[:,:2].norm(dim=-1) < 1) & (points_rotated[:,2].abs()<1)
        # align with normal with upper z axis
        # scale back to unit axis
        # get points inside unit cylinder
    return is_inside


# generate random planes uniform normals 
def generate_random_planes(initial_point, N, sx=0, sy=0,sz=0):
    points = (torch.rand([N,3], device='cuda')-0.5)
    points[:,0] *= sx
    points[:,1] *= sy
    points[:,2] *= sz
    normals = torch.zeros([N,3], device = 'cuda')
    # Randomally Sample on a unit sphere to eliminate bias -> rejection sampling from a unit cube
    for i in range(N):
        rand_vec = (torch.rand([1,3],device = 'cuda') - 0.5)*2
        while torch.norm(rand_vec) > 1:
            rand_vec = (torch.rand([1,3],device = 'cuda') - 0.5)*2
        rand_vec /= torch.norm(rand_vec)
        normals[i] = rand_vec
    #loc, normals = sample_random_points_and_normals(initial_point, sx,sy,sz, N)
    return points, normals

def find_extreme_point_in_dir(points, dir):
    id = (torch.sum(points * dir.unsqueeze(0),dim=-1)).argmax()
    return points[id]

def create_convex_hull(initial_points, locs, normals, N=10):
    # find most extreme points in normals direction
    points = torch.zeros([N,3], device = 'cuda')
    N = min(N,initial_points.shape[0])
    for i in range(normals.shape[0]):
        points[N-1] = find_extreme_point_in_dir(initial_points, normals[i])
        points[0] = locs[0]
        points[1:N-1] = initial_points[torch.randperm(initial_points.shape[0])[:N-2]]
    ch = ConvexHull(points.cpu().numpy())
    return torch.tensor(ch.points, device = 'cuda'), torch.tensor(ch.simplices, dtype = torch.int32, device = 'cuda')

def is_point_outside_of_plane(normals, locs, points):
    is_outside = torch.zeros(points.shape[0], dtype=bool, device = points.device)
    for i in range(normals.shape[0]):
        outside_i = torch.sum((points - locs[i].unsqueeze(0))*normals[i], dim=-1) > 0
        is_outside |= outside_i
    return is_outside
        