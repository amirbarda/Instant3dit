import torch
import torch.nn.functional as tfunc
from ROAR.util.geometry_utils import rotation_angle_between_two_vectors_and_axis
from utils.typing_utils import *
import torch.nn.functional as F
import numpy as np

def c2w_to_w2c(c2w):
    w2c = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0
    return w2c

def projection_from_intrinsics(fx,fy,cx,cy,img_size, n=1, f=50):
    projection = torch.zeros([fx.shape[0], 4,4], device=fx.device, dtype=torch.float)
    projection[:,0,0] = fx / (0.5*img_size[0])
    projection[:,1,1] = fy / (0.5*img_size[1])
    projection[:,0,2] = 1 - cx/(0.5*img_size[0])
    projection[:,1,2] = cy/(0.5*img_size[1]) - 1
    projection[:,2, 2] = -(f + n) / (f - n)
    projection[:,2, 3] = -(2 * f * n) / (f - n)
    projection[:,3, 2] = -1
    return projection

def convert_c2w_between_opengl_opencv(c2w):
    right = c2w[:,:,0]
    up = c2w[:,:,1]
    forward = c2w[:,:,2]
    loc = c2w[:,:,3]
    return torch.stack([right,-up,-forward, loc],dim=-1)


def get_projection_matrix(
    fovy: Float[Tensor, "B"], aspect_wh: float, near: float, far: float
) -> Float[Tensor, "B 4 4"]:
    batch_size = fovy.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
    proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[:, 1, 1] = -1.0 / torch.tan(
        fovy / 2.0
    )  # add a negative sign here as the y axis is flipped in nvdiffrast output
    proj_mtx[:, 2, 2] = -(far + near) / (far - near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[:, 3, 2] = -1.0
    return proj_mtx

def proj_to_K(projection, img_size):
    K = torch.eye(4, device=projection.device)
    K[0, 0] = abs(0.5 * projection[0, 0] * img_size[0]) #fx
    K[1, 1] = abs(0.5 * projection[1, 1] * img_size[1]) #fy
    K[0, 1] = 0
    K[0, 2] = 0.5 * (1 - projection[0, 2]) * img_size[0] #cx
    K[1, 2] = 0.5 * (projection[1, 2] + 1) * img_size[1] #cy
    return K


def calculate_elevation_and_azimuth(points):
    x_vecs = torch.tensor([1,0,0], dtype=torch.float, device=points.device)
    x_vecs = x_vecs.expand(points.shape[0],-1)
    z_vecs = torch.tensor([0,0,1], dtype=torch.float, device=points.device)
    z_vecs = z_vecs.expand(points.shape[0],-1)
    vectors = points
    xy_vectors = torch.tensor(vectors, device=points.device)
    xy_vectors[:,2] = 0
    xy_vectors = tfunc.normalize(xy_vectors)
    # calculate the clockwise rotation around z needed to get it to be in the xz axis - this is the azimuth
    cos_elevation = torch.sum(tfunc.normalize(points)*xy_vectors,dim=-1)
    azimuth = rotation_angle_between_two_vectors_and_axis(x_vecs, xy_vectors, z_vecs)
    # rotate vector and calculate the rotation around the y axis to get the vector in the z direction
    return azimuth.flatten(), torch.acos(cos_elevation).flatten()*torch.sign(points[:,2])

def get_mvp_matrix(
    c2w: Float[Tensor, "B 4 4"], proj_mtx: Float[Tensor, "B 4 4"]
) -> Float[Tensor, "B 4 4"]:
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx


def get_rays(
    directions: Float[Tensor, "... 3"],
    c2w: Float[Tensor, "... 4 4"],
    keepdim=False,
    noise_scale=0.0,
    normalize=True,
) -> Tuple[Float[Tensor, "... 3"], Float[Tensor, "... 3"]]:
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        if c2w.ndim == 2:  # (4, 4)
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
        rays_o = c2w[:, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 3:  # (H, W, 3)
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                -1
            )  # (H, W, 3)
            rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                -1
            )  # (B, H, W, 3)
            rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 4:  # (B, H, W, 3)
        assert c2w.ndim == 3  # (B, 4, 4)
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
            -1
        )  # (B, H, W, 3)
        rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)

    # add camera noise to avoid grid-like artifect
    # https://github.com/ashawkey/stable-dreamfusion/blob/49c3d4fa01d68a4f027755acf94e1ff6020458cc/nerf/utils.py#L373
    if noise_scale > 0:
        rays_o = rays_o + torch.randn(3, device=rays_o.device) * noise_scale
        rays_d = rays_d + torch.randn(3, device=rays_d.device) * noise_scale

    if normalize:
        rays_d = F.normalize(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d

def convert_c2w_between_opengl_opencv(c2w):
    right = c2w[:,:,0]
    up = c2w[:,:,1]
    forward = c2w[:,:,2]
    loc = c2w[:,:,3]
    return torch.stack([right,-up,-forward, loc],dim=-1)

def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
    device='cuda'
) -> Float[Tensor, "H W 3"]:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systemsxw

    Inputs:
        H, W, focal, principal, use_pixel_centers: image height, width, focal length, principal point and whether use pixel centers
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0

    if isinstance(focal, float):
        fx, fy = focal, focal
        cx, cy = W / 2, H / 2
    else:
        fx, fy = focal
        assert principal is not None
        cx, cy = principal

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device) + pixel_center,
        torch.arange(H, dtype=torch.float32, device=device) + pixel_center,
        indexing="xy",
    )

    directions: Float[Tensor, "H W 3"] = torch.stack(
        [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i, device=device)], -1
    )

    return directions


def get_c2w_from_Rt(Rt):
    c2w: Float[Tensor, "B 4 4"] = torch.zeros(Rt.shape[0], 4, 4).to(Rt)
    c2w[:, :3, :3] = Rt[:, :3, :3].permute(0, 2, 1)
    c2w[:, :3, 3:] = -Rt[:, :3, :3].permute(0, 2, 1) @ Rt[:, :3, 3:]
    c2w[:, 3, 3] = 1.0
    return c2w

def mvp_to_KRt(mv, projection, img_size):
    K = proj_to_K(projection, img_size)
    R = mv[:, :3, :3]
    C = -torch.norm(mv[:, :3, 3], dim=-1, keepdim=True).unsqueeze(-1) * \
        torch.bmm(torch.transpose(R, dim0=1, dim1=2),
                  torch.tensor([[[0], [0], [1.0]]], device=mv.device, dtype=torch.float).repeat([mv.shape[0], 1, 1]))
    Rt = torch.eye(4, device=mv.device).unsqueeze(0).repeat([mv.shape[0], 1, 1])
    Rt[:, :3, :3] = -torch.transpose(R, dim0=1, dim1=2)
    Rt[:, :3, [3]] = -C
    return K, Rt


def c2w_to_Rt(c2w):
    Rt = torch.inverse(c2w)
    return Rt

def Rt_to_loc_dir_model(Rt):
    cam_loc = Rt[:, :3, 3]
    R = Rt[:, :3, :3]
    cam_dir = R[:, :3, 2]  # z axis
    # todo: model matrix
    modelMatrix = R
    modelMatrix[:, :3, 0] *= -1
    modelMatrix = -torch.transpose(modelMatrix, dim0=1, dim1=2)
    return cam_loc, cam_dir, modelMatrix

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


def normalized_grid(width, height):
    """Returns grid[x,y] -> coordinates for a normalized window.
    
    Args:
        width, height (int): grid resolution
    """

    # These are normalized coordinates
    # i.e. equivalent to 2.0 * (fragCoord / iResolution.xy) - 1.0
    window_x = np.linspace(-1, 1, num=width) * (width / height)
    window_x += np.random.rand(*window_x.shape) * (1. / width)
    window_y = np.linspace(1, -1, num=height)
    window_y += np.random.rand(*window_y.shape) * (1. / height)
    coord = np.array(np.meshgrid(window_x, window_y, indexing='xy')).transpose(2,1,0)

    return coord


def look_at(f, t, width, height, mode='ortho', fov=90.0, device='cuda'):
    """Vectorized look-at function, returns an array of ray origins and directions
    URL: https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
    """

    camera_origin = f
    look_at_dir = t
    y_axis = torch.FloatTensor([0,1,0]).to(device)
    if not torch.is_tensor(camera_origin):
        camera_origin = torch.FloatTensor(f).to(device)
    if not torch.is_tensor(look_at_dir):
        look_at_dir = torch.FloatTensor(t).to(device)
    camera_view = F.normalize(look_at_dir - camera_origin, dim=-1)
    if len(camera_origin.shape) != 1:
        y_axis = torch.tensor([[0, 1.0, 0]], dtype=torch.float, device=device).repeat(repeats=[camera_origin.shape[0], 1])
        dim = -1
        camera_right = F.normalize(torch.cross(camera_view, y_axis, dim=dim), dim=dim)
        camera_up = F.normalize(torch.cross(camera_right, camera_view, dim=dim), dim=dim)

        coord = torch.tensor(normalized_grid(width, height), device=device)
        ray_origin = camera_right[:, np.newaxis, np.newaxis] * coord[np.newaxis, ..., 0, np.newaxis] * np.tan(
            np.radians(fov / 2)) + \
                     camera_up[:, np.newaxis, np.newaxis] * coord[np.newaxis, ..., 1, np.newaxis] * np.tan(
            np.radians(fov / 2)) + \
                     camera_origin[:, np.newaxis, np.newaxis] + camera_view[:, np.newaxis, np.newaxis]
        # ray_origin = ray_origin.reshape(-1,1, 3)
        ray_offset = camera_view[:, np.newaxis, np.newaxis].repeat(1, ray_origin.shape[1], ray_origin.shape[2], 1)

        if mode == 'ortho':  # Orthographic camera
            ray_dir = F.normalize(ray_offset, dim=-1)
        elif mode == 'persp':  # Perspective camera
            ray_dir = ray_origin - camera_origin[:, np.newaxis, np.newaxis]
            ray_dir = F.normalize(ray_dir, dim=-1)
            ray_origin = camera_origin[:, np.newaxis, np.newaxis].repeat(1, ray_origin.shape[1], ray_origin.shape[2], 1)
        else:
            raise ValueError('Invalid camera mode!')
    else:
        y_axis = torch.tensor([0, 1, 0], dtype=torch.float, device=device)
        dim = 0
        camera_right = F.normalize(torch.cross(camera_view, torch.FloatTensor([0, 1, 0]).to(device)), dim=0)
        camera_up = F.normalize(torch.cross(camera_right, camera_view), dim=0)
        coord = normalized_grid(width, height, device=device)
        ray_origin = camera_right * coord[..., 0, np.newaxis] * np.tan(np.radians(fov / 2)) + \
                     camera_up * coord[..., 1, np.newaxis] * np.tan(np.radians(fov / 2)) + \
                     camera_origin + camera_view
        ray_origin = ray_origin.reshape(-1, 3)
        ray_offset = camera_view.unsqueeze(0).repeat(ray_origin.shape[0], 1)
        if mode == 'ortho':  # Orthographic camera
            ray_dir = F.normalize(ray_offset, dim=-1)
        elif mode == 'persp':  # Perspective camera
            ray_dir = F.normalize(ray_origin - camera_origin, dim=-1)
            ray_origin = camera_origin.repeat(ray_dir.shape[0], 1)
        else:
            raise ValueError('Invalid camera mode!')


    return ray_origin, ray_dir


def get_K_from_intrinsics(intrinsics):
    fx = intrinsics[:,0]
    fy = intrinsics[:,1]
    cx = intrinsics[:,2]
    cy = intrinsics[:,3]
    K = torch.zeros([intrinsics.shape[0],3,3])
    K[:,0,0] = fx
    K[:,1,1] = fy
    K[:,0,2] = cx
    K[:,1,2] = cy
    K[:,2,2] = 1
    return K

def FOV_to_intrinsics(fov, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """
    focal_length = 0.5 / np.tan(np.deg2rad(fov) * 0.5)
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics

def pad_camera_extrinsics_4x4(extrinsics):
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    extrinsics = torch.cat([extrinsics, padding], dim=-2)
    return extrinsics

def center_looking_at_camera_pose(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    """
    Create OpenGL camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4) or (3, 4)
    """
    # by default, looking at the origin and world up is z-axis
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], device=camera_position.device, dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], device=camera_position.device, dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # OpenGL camera: z-backward, x-right, y-up
    z_axis = camera_position - look_at
    z_axis = F.normalize(z_axis, dim=-1).float()
    x_axis = torch.linalg.cross(up_world, z_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1).float()
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    y_axis = F.normalize(y_axis, dim=-1).float()

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics

def get_circular_camera_poses(M=120, radius=2.5, elevation=30.0):
    # M: number of circular views
    # radius: camera dist to center
    # elevation: elevation degrees of the camera
    # return: (M, 4, 4)
    assert M > 0 and radius > 0

    elevation = np.deg2rad(elevation)

    camera_positions = []
    for i in range(M):
        azimuth = 2 * np.pi * i / M
        x = radius * np.cos(elevation) * np.cos(azimuth)
        y = radius * np.cos(elevation) * np.sin(azimuth)
        z = radius * np.sin(elevation)
        camera_positions.append([x, y, z])
    camera_positions = np.array(camera_positions)
    camera_positions = torch.from_numpy(camera_positions).float()
    extrinsics = center_looking_at_camera_pose(camera_positions)
    return extrinsics

def spherical_camera_pose(azimuths: np.ndarray, elevations: np.ndarray, radius=2.5):
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    xs = radius * np.cos(elevations) * np.cos(azimuths)
    ys = radius * np.cos(elevations) * np.sin(azimuths)
    zs = radius * np.sin(elevations)

    cam_locations = np.stack([xs, ys, zs], axis=-1)
    cam_locations = torch.from_numpy(cam_locations).float()

    c2ws = center_looking_at_camera_pose(cam_locations)
    return c2ws

def get_zero123plus_input_cameras(batch_size=1, radius=4.0, fov=30.0, azimuth_offset = 0):
    """
    Get the input camera parameters.
    """
    #azimuths = np.array([30, 90, 150, 210, 270, 330]).astype(float)
    azimuths = np.array([0+azimuth_offset, 90, 90+azimuth_offset, 210, 180+azimuth_offset, 270+azimuth_offset]).astype(float)
    elevations = np.array([20, -10, 20, -10, 20, 20]).astype(float) #-10]).astype(float)
    
    c2ws = spherical_camera_pose(azimuths, elevations, radius)
    c2ws = c2ws.float()#.flatten(-2)

    Ks = FOV_to_intrinsics(fov).unsqueeze(0).repeat(6, 1, 1).float().flatten(-2)

    #extrinsics = c2ws[:, :12]
    intrinsics = torch.stack([Ks[:, 0], Ks[:, 4], Ks[:, 2], Ks[:, 5]], dim=-1)
    #cameras = torch.cat([extrinsics, intrinsics], dim=-1)

    #return cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    c2ws = c2ws[[0,2,4,5]]
    intrinsics = intrinsics[[0,2,4,5]]
    return c2ws, intrinsics*512

