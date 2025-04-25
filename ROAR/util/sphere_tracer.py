import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
# from continous_remeshing import csg
import torch
import torch.nn as nn
from tqdm import tqdm
import imageio

def sphere_tracing(
        signed_distance_function,
        ray_positions,
        ray_directions,
        num_iterations,
        convergence_threshold,
        safety_margin=1.,
        foreground_masks=None,
        bounding_radius=None,
        count_access=False,
):
    """
    Sphere traces signed distance functions
    """
    counter = 0
    if foreground_masks is None:
        foreground_masks = torch.all(torch.isfinite(ray_positions), dim=-1, keepdim=True)
    if bounding_radius:
        a = torch.sum(ray_directions * ray_directions, dim=-1, keepdim=True)
        b = 2 * torch.sum(ray_directions * ray_positions, dim=-1, keepdim=True)
        c = torch.sum(ray_positions * ray_positions, dim=-1, keepdim=True) - bounding_radius ** 2
        d = b ** 2 - 4 * a * c
        t = (-b - torch.sqrt(d)) / (2 * a)
        bounded = d >= 0
        ray_positions = torch.where(bounded, ray_positions + ray_directions * t, ray_positions)
        foreground_masks = foreground_masks & bounded
    foreground_masks = foreground_masks[:, 0]
    with torch.no_grad():
        loop = tqdm(total=num_iterations, position=0)
        converged = torch.zeros((ray_positions.shape[:-1]), device=ray_positions.device, dtype=torch.bool)
        for i in range(num_iterations):
            loop.set_description('sphere tracing up to {} iters'.format(num_iterations))
            loop.update(1)
            mask = foreground_masks & ~converged
            cur_ray_positions = ray_positions.view(-1, 3)[mask]
            cur_ray_directions = ray_directions.view(-1, 3)[mask]
            signed_distances = signed_distance_function(cur_ray_positions).view(-1, 1)
            if count_access:
                counter += cur_ray_positions.shape[0]
            cur_ray_positions = cur_ray_positions + cur_ray_directions * signed_distances * safety_margin
            ray_positions[mask] = cur_ray_positions
            if bounding_radius:
                bounded = torch.norm(ray_positions, dim=-1) < bounding_radius
                foreground_masks = foreground_masks & bounded
            converged[mask] |= torch.abs(signed_distances[:, 0]) < convergence_threshold
            if torch.all(~foreground_masks | converged):
                break
        loop.close()
    return ray_positions, converged[:, None], counter


def compute_shadows(
        signed_distance_function,
        surface_positions,
        surface_normals,
        light_directions,
        num_iterations,
        convergence_threshold,
        foreground_masks=None,
        bounding_radius=None,
):
    surface_positions, converged = sphere_tracing(
        signed_distance_function=signed_distance_function,
        ray_positions=surface_positions + surface_normals * 1e-3,
        ray_directions=light_directions,
        num_iterations=num_iterations,
        convergence_threshold=convergence_threshold,
        foreground_masks=foreground_masks,
        bounding_radius=bounding_radius,
    )
    return foreground_masks & converged


def compute_normal(
        signed_distance_function,
        surface_positions,
        finite_difference_epsilon,
        grad=False,
        count_access=False,
):
    counter = 0
    if grad:
        surface_positions.requires_grad = True
        if count_access:
            counter += surface_positions.shape[0]
        raw = signed_distance_function(surface_positions)
        d_output = torch.ones_like(raw, requires_grad=False, device=raw.device)
        surface_normals = torch.autograd.grad(
                outputs=raw,
                inputs=surface_positions,
                grad_outputs=d_output,
                create_graph=False,
                retain_graph=False)[0]
            # normals = normals.detach()
        # surface_normals = -1 * surface_normals
    else:
        finite_difference_epsilon_x = surface_positions.new_tensor([finite_difference_epsilon, 0.0, 0.0])
        finite_difference_epsilon_y = surface_positions.new_tensor([0.0, finite_difference_epsilon, 0.0])
        finite_difference_epsilon_z = surface_positions.new_tensor([0.0, 0.0, finite_difference_epsilon])
        if count_access:
            counter += 6 * surface_positions.shape[0]
        surface_normals_x = signed_distance_function(
            surface_positions + finite_difference_epsilon_x) - signed_distance_function(
            surface_positions - finite_difference_epsilon_x)
        surface_normals_y = signed_distance_function(
            surface_positions + finite_difference_epsilon_y) - signed_distance_function(
            surface_positions - finite_difference_epsilon_y)
        surface_normals_z = signed_distance_function(
            surface_positions + finite_difference_epsilon_z) - signed_distance_function(
            surface_positions - finite_difference_epsilon_z)
        surface_normals = torch.cat((surface_normals_x, surface_normals_y, surface_normals_z), dim=-1)
    surface_normals = nn.functional.normalize(surface_normals, dim=-1, eps=1e-5)
    return surface_normals, counter


def generate_rays(w2v=None, v2c=None, resx=512, resy=512, device="cuda:0"):
    """
    generates batch of rays for a batch of camera matrices nx4x4
    """
    if w2v is None or v2c is None:
        #---------------- camera matrix ---------------- #
        fx = fy = resx
        cx = cy = resx / 2
        camera_matrix = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device)
        # ---------------- camera position ---------------- #
        distance = 2.0
        azimuth = np.pi / 4.0
        elevation = np.pi / 4.0

        camera_position = torch.tensor([
            +np.cos(elevation) * np.sin(azimuth),
            -np.sin(elevation),
            -np.cos(elevation) * np.cos(azimuth)
        ], device=device, dtype=torch.float32) * distance
        # ---------------- camera rotation ---------------- #
        target_position = torch.tensor([0.0, 0.0, 0.0], device=device)
        up_direction = torch.tensor([0.0, 1.0, 0.0], device=device)

        camera_z_axis = target_position - camera_position
        camera_x_axis = torch.cross(up_direction, camera_z_axis, dim=-1)
        camera_y_axis = torch.cross(camera_z_axis, camera_x_axis, dim=-1)
        camera_rotation = torch.stack((camera_x_axis, camera_y_axis, camera_z_axis), dim=-1)
        camera_rotation = nn.functional.normalize(camera_rotation, dim=-2)
        camera_rotation = camera_rotation
        y_positions = torch.arange(resy, dtype=torch.float32, device=device)
        x_positions = torch.arange(resx, dtype=torch.float32, device=device)
        y_positions, x_positions = torch.meshgrid(y_positions, x_positions)
        z_positions = torch.ones_like(y_positions)
        ray_positions = torch.stack((x_positions, y_positions, z_positions), dim=-1)
        ray_positions = torch.einsum("mn,...n->...m", torch.inverse(camera_matrix), ray_positions)
        ray_positions = torch.einsum("mn,...n->...m", camera_rotation, ray_positions) + camera_position
        ray_directions = nn.functional.normalize(ray_positions - camera_position, dim=-1)
        return ray_positions, ray_directions

    y_clip = (torch.arange(resy, dtype=torch.float32, device=device) / resy) * 2 - 1
    x_clip = (torch.arange(resx, dtype=torch.float32, device=device) / resx) * 2 - 1
    xy_clip = torch.stack(torch.meshgrid((x_clip, y_clip), indexing='xy'), dim=-1)
    xy_clip_near = torch.cat((xy_clip, -1*torch.ones_like(xy_clip[:, :, :1]), torch.ones_like(xy_clip[:, :, :1])), dim=-1)
    xy_clip_far = torch.cat((xy_clip, torch.ones_like(xy_clip[:, :, :1]), torch.ones_like(xy_clip[:, :, :1])), dim=-1)
    rays_d = []
    rays_o = []
    for i in range(len(w2v)):
        if v2c.ndim == 3:
            v2c_transform = v2c[i]
        else:
            v2c_transform = v2c
        w2v_transform = w2v[i]
        xy_view_near = (torch.inverse(v2c_transform) @ xy_clip_near.view(-1, 4).T).T.view(resy, resx, 4)
        xy_view_near = xy_view_near / xy_view_near[:, :, 3:]
        xy_world_near = (torch.inverse(w2v_transform) @ xy_view_near.view(-1, 4).T).T.view(resy, resx, 4)
        ray_o = xy_world_near[:, :, :3]
        camera_position = torch.inverse(w2v_transform)[:3 ,3]
        ray_d = nn.functional.normalize(xy_world_near[:, :, :3] - camera_position, dim=-1)
        rays_d.append(ray_d)
        rays_o.append(ray_o)
    return torch.stack(rays_o), torch.stack(rays_d)


def render(p_sdf, ray_positions, ray_directions, num_iterations=2000, convergence_threshold=1e-5, grad=False, count_access=False, xyz_mode=False):
    counter = 0
    surface_positions, converged, trace_counter = sphere_tracing(
        signed_distance_function=p_sdf,
        ray_positions=ray_positions,
        ray_directions=ray_directions,
        num_iterations=num_iterations,
        convergence_threshold=convergence_threshold,
        safety_margin=1.0,
        bounding_radius=1.0,
        count_access=count_access
    )
    counter += trace_counter
    surface_positions = torch.where(converged, surface_positions, torch.zeros_like(surface_positions))
    if xyz_mode:
        surface_color = surface_positions / (surface_positions.norm(dim=-1, keepdim=True) + 1e-6)
        image = (surface_color + 1.0) / 2.0
    else:
        surface_normals, normal_counter = compute_normal(
            signed_distance_function=p_sdf,
            surface_positions=surface_positions.view(-1, 3),
            finite_difference_epsilon=1e-3,
            grad=grad,
            count_access=count_access
        )
        counter += normal_counter
        surface_normals = surface_normals.view(ray_positions.shape)
        surface_normals = torch.where(converged, surface_normals, torch.zeros_like(surface_normals))
        image = (surface_normals + 1.0) / 2.0
    image = torch.where(converged, image, torch.zeros_like(image))
    image = torch.concat((image, converged), dim=-1)
    if count_access:
        return image, counter
    else:
        return image


if __name__ == "__main__":
    w2v, v2c = make_star_cameras(2, 2, device="cuda:0")
    ray_origins, ray_directions = generate_rays(w2v, v2c, 512, 512, "cuda:0")
    # sdf = csg.sphere(0.5)  # a torch function returning the signed distance function for any batch of locations (n x 3)
    images = []
    for o, d in zip(ray_origins, ray_directions):
        images.append(render(sdf, o, d))
    images = torch.stack(images)
    dst = Path("output", "sphere_tracer")
    dst.mkdir(parents=True, exist_ok=True)
    for i in range(len(images)):
        imageio.imwrite(str(Path(dst, "sphere_{}.png".format(i))),(images[i,:,:,:3]*255).clamp(min=0, max=255).type(torch.uint8).detach().cpu().numpy())
    # torchvision.utils.save_image(images, "sphere.png")
