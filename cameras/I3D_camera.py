import torch
import random
import math
import json
from pathlib import Path
from utils.camera_utils import *
from utils.typing_utils import *
import torch.nn.functional as F
from easydict import EasyDict as edict


def load_fixed_camera_data(path,w = 512 ,h = 512):
    camera_path = Path(path)
    num_cameras = 16
    with open(camera_path, 'r') as f:
        camera_data = json.load(f)
    w2cs = [camera_data['frames'][i]['w2c'] for i in range(0,num_cameras,4)]
    w2c_opencv= torch.tensor(w2cs, device='cuda')
    # w2c_opengl[:,1,:] *= -1
    # w2c_opengl[:,2,:] *= -1
    fxs = [camera_data['frames'][i]['fx'] for i in range(0,num_cameras,4)]
    fx = torch.tensor(fxs, device='cuda')
    fys = [camera_data['frames'][i]['fy'] for i in range(0,num_cameras,4)]
    fy = torch.tensor(fys, device='cuda')
    cxs = [camera_data['frames'][i]['cx'] for i in range(0,num_cameras,4)]
    cx = torch.tensor(cxs, device='cuda')
    cys = [camera_data['frames'][i]['cy'] for i in range(0,num_cameras,4)]
    cy = torch.tensor(cys, device='cuda')
    intrinsic_vector = torch.tensor([fxs,fys,cxs,cys], device='cuda').T
    #w2c_opencv[:,:,1] *= -1
    #w2c_opencv[:,:,2] *= -1
    c2w_opencv = c2w_to_w2c(w2c_opencv)
    c2w_opengl = convert_c2w_between_opengl_opencv(c2w_opencv)
    #c2w_opengl[:,:,1] *= -1
    #c2w_opengl[:,:,2] *= -1
    mv = c2w_to_w2c(c2w_opengl).to('cuda')
    #mv[:,:3,3,] *= 3/2.7
    proj = projection_from_intrinsics(fx, fy, cx, cy, [w,h], n=0.1, f=1000.0)
    mvp = proj @ mv
    camera_positions = c2w_opengl[:,:3,3]
    return {'mvp_mtx': mvp, 'c2w': c2w_opengl, 'camera_positions': camera_positions, "intrinsic" : intrinsic_vector, 'width': w, 'height': h},  edict({'batch_size': num_cameras, 'width': w, 'height': h, 'elevation_range' : [-10, 90], 'azimuth_range' : [-10, 90], 'camera_distance_range' : [0,0], 'fovy_range' : [40, 70]})



class I3D_cameras():
    def __init__(self, fixed_camera_data_path) -> None:
        super().__init__()
        
        self.fixed_cameras_data, cfg = load_fixed_camera_data(fixed_camera_data_path)

        self.batch_size: int = 4
        self.height: int = cfg.height
        self.width: int = cfg.width

        self.directions_unit_focals = get_ray_directions(H=self.height, W=self.width, focal=1.0).unsqueeze(0)
        

        self.directions_unit_focal = self.directions_unit_focals[0]
        self.elevation_range = cfg.elevation_range
        self.azimuth_range = cfg.azimuth_range
        self.camera_distance_range = cfg.camera_distance_range
        self.fovy_range = cfg.fovy_range
        self.light_distance_range: Tuple[float, float] = (3.0, 3.5) 
        self.light_sample_strategy = 'dreamfusion'
        self.light_position_perturb: float = 0 #1.0

    def __call__(self, batch) -> Dict[str, Any]:
        
        if 'c2w' in batch:
            c2w = batch['c2w']
        else:
            c2w = self.fixed_cameras_data['c2w']
        right = c2w[:,:,0]
        forward = c2w[:,:,2]
        up = c2w[:,:,1]
        loc = c2w[:,:,3]

        elevation_deg = torch.rand(1) / 4 * (self.elevation_range[1] - self.elevation_range[0])*0 #+ self.cfg.azimuth_range[0]
        elevation_rad = elevation_deg*torch.pi/180
        rotation_matrix = torch.tensor([[1,0,0,0],
                                        [0,torch.cos(elevation_rad),-torch.sin(elevation_rad),0],
                                        [0,torch.sin(elevation_rad),torch.cos(elevation_rad),0],
                                        [0,0,0,0]], device = c2w.device, dtype=torch.float).unsqueeze((0))

        right = torch.bmm(rotation_matrix.expand([right.shape[0],-1,-1]), right.unsqueeze(-1))
        right = right.squeeze(-1)

        forward = torch.bmm(rotation_matrix.expand([forward.shape[0],-1,-1]), forward.unsqueeze(-1))
        forward = forward.squeeze(-1)

        up = torch.bmm(rotation_matrix.expand([up.shape[0],-1,-1]), up.unsqueeze(-1))
        up = up.squeeze(-1)

        loc = torch.bmm(rotation_matrix.expand([loc.shape[0],-1,-1]), loc.unsqueeze(-1))
        loc = loc.squeeze(-1)
        loc[:,3] =  1
        if not 'azimuth' in batch: 
            azimuth_deg = (torch.rand(1) - 0.5)/8 * (self.azimuth_range[1] - self.azimuth_range[0]) #+ self.cfg.azimuth_range[0]
        else:
            azimuth_deg= batch['azimuth']
        azimuth_rad = azimuth_deg*torch.pi/180

        rotation_matrix = torch.tensor([[torch.cos(azimuth_rad),-torch.sin(azimuth_rad),0,0],
                                        [torch.sin(azimuth_rad),torch.cos(azimuth_rad),0,0],
                                        [0,0,1,0],
                                        [0,0,0,0]], device = c2w.device, dtype=torch.float).unsqueeze((0))

        right = torch.bmm(rotation_matrix.expand([right.shape[0],-1,-1]), right.unsqueeze(-1))
        right = right.squeeze(-1)

        forward = torch.bmm(rotation_matrix.expand([forward.shape[0],-1,-1]), forward.unsqueeze(-1))
        forward = forward.squeeze(-1)

        up = torch.bmm(rotation_matrix.expand([up.shape[0],-1,-1]), up.unsqueeze(-1))
        up = up.squeeze(-1)

        loc = torch.bmm(rotation_matrix.expand([loc.shape[0],-1,-1]), loc.unsqueeze(-1))
        loc = loc.squeeze(-1)
        loc[:,3] =  1

        # if 'c2w' in batch:
        #     c2w = batch['c2w']
        #     loc = c2w[:,:,3]
        # else:
        batch['c2w'] = torch.stack([-right,-up,forward,loc], dim=-1)

        batch['camera_positions'] = loc[:,:3]

        azimuth, elevation = calculate_elevation_and_azimuth(loc[:,:3])
        
        batch['azimuth'] = azimuth
        batch['elevation'] = elevation
        batch['camera_distances'] = loc[:,:3].norm(dim=-1)

        mv = c2w_to_w2c(batch['c2w']).to(c2w.device)

        if 'intrinsic' not in batch:
            batch['intrinsic'] = self.fixed_cameras_data['intrinsic']
            

        fx,fy,cx,cy = batch['intrinsic'][0].chunk(4)

        proj = projection_from_intrinsics(fx, fy, cx, cy, [512,512], n=0.1, f=1000.0)
        mvp = torch.bmm(proj.repeat([4,1,1]),mv) #proj @ mv
        batch["mvp_mtx"] = mvp
        #batch['c2w'] = torch.stack([right,-up,forward,loc], dim=-1)


        # rotate 
        
        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        if batch is None or not "elevation" in batch:
            if random.random() < 0.5:
                # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
                elevation_deg = (
                    torch.rand(self.batch_size)
                    * (self.elevation_range[1] - self.elevation_range[0])
                    + self.elevation_range[0]
                )
                elevation = elevation_deg * math.pi / 180
            else:
                # otherwise sample uniformly on sphere
                elevation_range_percent = [
                    (self.elevation_range[0] + 90.0) / 180.0,
                    (self.elevation_range[1] + 90.0) / 180.0,
                ]
                # inverse transform sampling
                elevation = torch.asin(
                    2
                    * (
                        torch.rand(self.batch_size)
                        * (elevation_range_percent[1] - elevation_range_percent[0])
                        + elevation_range_percent[0]
                    )
                    - 1.0
                )
                elevation_deg = elevation / math.pi * 180.0
        else:
            elevation_deg = batch["elevation"]
            elevation = elevation_deg *  math.pi / 180.0
        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        if batch is None or not "azimuth" in batch:
            # simple random sampling
            azimuth_deg = (
                torch.rand(self.batch_size)
                * (self.azimuth_range[1] - self.azimuth_range[0])
                + self.azimuth_range[0]
            )
            azimuth = azimuth_deg * math.pi / 180
        else:
            azimuth_deg = batch["azimuth"]
            azimuth = azimuth_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        if batch is None or not "camera_distances" in batch:
            camera_distances: Float[Tensor, "B"] = (
                torch.rand(self.batch_size)
                * (self.camera_distance_range[1] - self.cfg.camera_distance_range[0])
                + self.camera_distance_range[0]
            )
        else:
            camera_distances = batch["camera_distances"]
        
        if batch is None or not "camera_positions" in batch:
            # convert spherical coordinates to cartesian coordinates
            # right hand coordinate system, x back, y right, z up
            # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
            camera_positions: Float[Tensor, "B 3"] = torch.stack(
                [
                    camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                    camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distances * torch.sin(elevation),
                ],
                dim=-1,
            )
        else:
            camera_positions = batch["camera_positions"]
        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        if batch is None or not "camera_positions" in batch:
            # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
            camera_perturb: Float[Tensor, "B 3"] = (
                torch.rand(self.batch_size, 3) * 2 * self.camera_perturb
                - self.cfg.camera_perturb
            )
            camera_positions = camera_positions + camera_perturb
            # sample center perturbations from a normal distribution with mean 0 and std center_perturb
            center_perturb: Float[Tensor, "B 3"] = (
                torch.randn(self.batch_size, 3) * self.center_perturb
            )
            center = center + center_perturb
            # sample up perturbations from a normal distribution with mean 0 and std up_perturb
            up_perturb: Float[Tensor, "B 3"] = (
                torch.randn(self.batch_size, 3) * self.up_perturb
            )
            up = up + up_perturb

        if batch is None or not "fovy" in batch:
        # sample fovs from a uniform distribution bounded by fov_range
            fovy_deg: Float[Tensor, "B"] = (
                torch.rand(self.batch_size)
                * (self.fovy_range[1] - self.fovy_range[0])
                + self.fovy_range[0]
            )
            fovy = fovy_deg * math.pi / 180
        else:
            fovy = batch["fovy"]

        if batch is None or not "light_positions" in batch:

            # sample light distance from a uniform distribution bounded by light_distance_range
            light_distances: Float[Tensor, "B"] = (
                torch.rand(self.batch_size, device=c2w.device)
                * (self.light_distance_range[1] - self.light_distance_range[0])
                + self.light_distance_range[0]
        )
            if self.light_sample_strategy == "dreamfusion":
                # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
                light_direction: Float[Tensor, "B 3"] = F.normalize(
                    camera_positions
                    + torch.randn(self.batch_size, 3, device=c2w.device) * self.light_position_perturb,
                    dim=-1,
                )
                # get light position by scaling light direction by light distance
                light_positions: Float[Tensor, "B 3"] = (
                    light_direction * light_distances[:, None]
                )
            elif self.light_sample_strategy == "magic3d":
                # sample light direction within restricted angle range (pi/3)
                local_z = F.normalize(camera_positions, dim=-1)
                local_x = F.normalize(
                    torch.stack(
                        [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                        dim=-1,
                    ),
                    dim=-1,
                )
                local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
                rot = torch.stack([local_x, local_y, local_z], dim=-1)
                light_azimuth = (
                    torch.rand(self.batch_size) * math.pi - 2 * math.pi
                )  # [-pi, pi]
                light_elevation = (
                    torch.rand(self.batch_size) * math.pi / 3 + math.pi / 6
                )  # [pi/6, pi/2]
                light_positions_local = torch.stack(
                    [
                        light_distances
                        * torch.cos(light_elevation)
                        * torch.cos(light_azimuth),
                        light_distances
                        * torch.cos(light_elevation)
                        * torch.sin(light_azimuth),
                        light_distances * torch.sin(light_elevation),
                    ],
                    dim=-1,
                )
                light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
            else:
                raise ValueError(
                    f"Unknown light sample strategy: {self.light_sample_strategy}"
                )
        else:
            light_positions = batch["light_positions"]

        if batch is None or not "c2w" in batch:
            lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
            right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
            up = F.normalize(torch.cross(right, lookat), dim=-1)
            c2w3x4: Float[Tensor, "B 3 4"] = torch.cat( #opengl cameras!
                [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
                dim=-1,
            )
            c2w: Float[Tensor, "B 4 4"] = torch.cat(
                [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
            )
            c2w[:, 3, 3] = 1.0
        else:
            c2w = batch['c2w']

        # get directions by dividing directions_unit_focal by focal length
        fovy = (-2.0*torch.atan( 1.0/proj[0,1,1]))[None]
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!

        if batch is None or not "rays_o" in batch:
            rays_o, rays_d = get_rays(directions, c2w.to(device=directions.device), keepdim=True)
        else:
            rays_o = batch["rays_o"]
            rays_d = batch["rays_d"]

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.01, 100.0
        )  # FIXME: hard-coded near and far

        if batch is None or not "mvp_mtx" in batch:
            mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)
        else:
            mvp_mtx = batch["mvp_mtx"]

        if batch is None or not "intrinsic" in batch:
            intrinsic = proj_to_K(proj_mtx[0], img_size=[self.width,self.height])
        else:
            intrinsic = batch["intrinsic"]   


        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "intrinsic": intrinsic,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "fovy" : fovy
        }
