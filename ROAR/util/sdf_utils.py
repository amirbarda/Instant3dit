import torch
from util.func import load_mesh, load_mesh_o3d, save_pc, save_mesh_properly, \
    slice_mesh_above_plane, decimate_mesh
from pathlib import Path
from configs.config import Config
import numpy as np
import cv2
from pyhocon import ConfigFactory
from sampling.random_sampler import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
import gsoup

###### settings #######
names = []
out_folder_name = "output"
sdf_name = "volSDF_bull_05" #"volSDF_dog_07" #"volSDF_buddah_03" #"volSDF_bull_05" #"volSDF_bread_08" # "volSDF_robot_06"  # idr_fruits_63  # idr_buddah_114
scan_number = 5
remove_views = []
###### initialize #######
options = Config().parse()
torch.manual_seed(options.seed)
# res = options.image_resolution
experiment_folder = Path(out_folder_name, options.experiment_name)
experiment_folder.mkdir(exist_ok=True, parents=True)
name = sdf_name
result_path = Path(experiment_folder, name)
result_path.mkdir(exist_ok=True, parents=True)
Config.save_to_file(result_path, options)
Path(result_path, 'polyscope_data').mkdir(exist_ok=True)
print(name)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def get_IDR_model(scan_number):
    conf = ConfigFactory.parse_file("/data1/yotam/idr/code/confs/dtu_fixed_cameras.conf")
    expdir = "/data1/yotam/idr/trained_models/dtu_fixed_cameras_{}".format(scan_number)
    model = get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
    model = model.to(options.device)
    old_checkpnts_dir = Path(expdir, "2020", 'checkpoints')
    saved_model_state = torch.load(Path(old_checkpnts_dir, 'ModelParameters', str("2000.pth")))
    model.load_state_dict(saved_model_state["model_state_dict"])
    model.eval()
    # n_views = len([x for x in Path("/data1/yotam/idr/data/DTU/scan{}/image".format(scan_number)).glob("*.png")])
    # bbox_v, bbox_f, _ = load_mesh("/data1/yotam/idr/evals/dtu_fixed_cameras_{}/bbox.obj".format(scan_number))
    poses = []
    # ks = []
    cameras = np.load(Path("/data1/yotam/idr/data/DTU/scan{}/cameras.npz".format(scan_number)))
    for i in range(64):
        P0=cameras['world_mat_%d'%i][:3,:]
        # K=cv2.decomposeProjectionMatrix(P0)[0]
        R=cv2.decomposeProjectionMatrix(P0)[1]
        c=cv2.decomposeProjectionMatrix(P0)[2]
        c=c/c[3]
        M=np.eye(4)
        M[:3,:]= np.concatenate((R, -R @ c[:3]), axis=1)
        poses.append(M)
        # ks.append(K)
    # for file in sorted(Path("/data1/yotam/idr/data/DTU/poses").glob("*.txt")):
    #     with open(file, 'r') as filor:
    #         lines = filor.readlines()
    #     cur_pos = []
    #     for line in lines:
    #         pose = np.array([float(x) for x in line.split(" ")])
    #         cur_pos.append(pose)
    #     cur_pos = np.array(cur_pos)
    #     out = cv2.decomposeProjectionMatrix(cur_pos)
    #     K = out[0]
    #     R = out[1]
    #     t = out[2]
    #     pose = np.eye(4, dtype=np.float32)
    #     pose[:3, :3] = R.T
    #     t = t/t[3]
    #     pose[:3,-1] = (-R @ t[:3])[:, 0]
    #     poses.append(pose)
    poses = np.stack(poses)
    poses = np.linalg.inv(poses)
    coa = gsoup.get_center_of_attention(poses)
    poses[:, :3, 3] -= coa
    poses, factor = gsoup.scale_poses(poses, n=7.0)
    poses = gsoup.opengl_c2w_to_opencv_c2w(poses)
    view_matrix = np.linalg.inv(poses)
    mvs = view_matrix[::2]
    return model, mvs

def chamfer_with_gt(ours_v, ours_f, gt_pc, plane_normalized, options):
    ours_v = post_process_transform @ torch.cat((ours_v, torch.ones_like(ours_v[:, 0:1])), dim=-1).T
    ours_v = ours_v[:3].T
    ours_v = ours_v.detach().cpu().numpy()
    ours_f = ours_f.cpu().numpy()
    ours_v, ours_f = slice_mesh_above_plane(ours_v, ours_f, plane_normalized.cpu().numpy())
    ours_v = torch.tensor(ours_v, dtype=torch.float32, device=options.device).unsqueeze(0)
    ours_f = torch.tensor(ours_f, dtype=torch.int64, device=options.device).unsqueeze(0)
    ours_sampled, _ = sample_points_from_meshes(ours_v, ours_f, 200000)
    ours_sampled = ours_sampled.squeeze(0)
    ours_chamfer, _ = chamfer_distance(ours_sampled.unsqueeze(0), gt_pc.unsqueeze(0))
    return ours_chamfer.item()

def prep_gt_pc(gt_mesh_path, plane_normalized, options):
    gtv, gtf = load_mesh_o3d(gt_mesh_path)
    gtv, gtf = slice_mesh_above_plane(gtv, gtf, plane_normalized.cpu().numpy())
    gtv = torch.tensor(gtv, dtype=torch.float32, device=options.device).unsqueeze(0)
    gtf = torch.tensor(gtf, dtype=torch.int64, device=options.device).unsqueeze(0)
    gt_mesh_sampled, _ = sample_points_from_meshes(gtv, gtf, 500000)
    gt_mesh_sampled = gt_mesh_sampled.squeeze(0)
    # save_pc(Path(result_path, "gt_mesh_sampled.ply"), gt_mesh_sampled)
    mask_gt = plane_normalized[:3] @ gt_mesh_sampled.T > -plane_normalized[-1]
    gt_sampled_masked = gt_mesh_sampled[mask_gt]
    return gt_sampled_masked

def compare(volSDF_scan_number, result_path, post_process_transform, options):
    volSDF_2_blendedMVS = {1: 1, 2: 7, 3: 8, 4: 12, 5:13, 6:15, 7:16, 8:17, 9:19}
    blendedMVS_scan_number = volSDF_2_blendedMVS[volSDF_scan_number]
    ours = Path(result_path, sdf_name+".obj")
    if not ours.is_file():
        return
    plane_path = Path("/data1/yotam/volsdf/data/BlendedMVS/planes/_{}.npy".format(blendedMVS_scan_number))
    plane = torch.tensor(np.load(plane_path), dtype=torch.float32, device=options.device)
    plane_normalized = plane / torch.norm(plane[:3])
    theirs_decimated = Path(result_path, "decimated.obj")
    # if theirs_decimated.exists():
    #     dec_v, dec_f, _ = load_mesh(theirs_decimated)
    #     dec_v = dec_v.unsqueeze(0)
    #     dec_f = dec_f.unsqueeze(0)
    # else:
    theirs_path = next(Path("/data1/yotam/volsdf/evals/bmvs_{}".format(volSDF_scan_number)).glob("**/*.ply"))
    theirs_v, theirs_f = load_mesh_o3d(theirs_path)
    dec_v, dec_f = slice_mesh_above_plane(theirs_v, theirs_f, plane_normalized.cpu().numpy())
    dec_v, dec_f = decimate_mesh(dec_v, dec_f, options.max_faces)
    dec_v = torch.tensor(dec_v, dtype=torch.float32, device=options.device).unsqueeze(0)
    dec_f = torch.tensor(dec_f, dtype=torch.int64, device=options.device).unsqueeze(0)
    save_mesh_properly(dec_v.squeeze(), dec_f.squeeze(), theirs_decimated)
    theirs_sampled, _ = sample_points_from_meshes(dec_v, dec_f, 500000)
    theirs_sampled = theirs_sampled.squeeze(0)
    save_pc(Path(result_path, "theirs_sampled.ply"), theirs_sampled)
    gt_mesh_path = Path("/data1/yotam/datasets/BlendedMVS/blendedMVS/scan{}/mesh.ply".format(blendedMVS_scan_number))
    gt_pc = prep_gt_pc(gt_mesh_path, plane_normalized, options)
    save_pc(Path(result_path, "gt_mesh_masked.ply"), gt_pc)
    ours_v, ours_f, _ = load_mesh(ours, device=options.device)
    # test = chamfer_with_gt(ours_v, ours_f, gt_pc, plane_normalized, options)
    ours_v = post_process_transform @ torch.cat((ours_v, torch.ones_like(ours_v[:, 0:1])), dim=-1).T
    ours_v = ours_v[:3].T
    ours_v = ours_v.cpu().numpy()
    ours_f = ours_f.cpu().numpy()
    ours_v, ours_f = slice_mesh_above_plane(ours_v, ours_f, plane_normalized.cpu().numpy())
    ours_v = torch.tensor(ours_v, dtype=torch.float32, device=options.device).unsqueeze(0)
    ours_f = torch.tensor(ours_f, dtype=torch.int64, device=options.device).unsqueeze(0)
    ours_sampled, _ = sample_points_from_meshes(ours_v, ours_f, 500000)
    ours_sampled = ours_sampled.squeeze(0)
    save_pc(Path(result_path, "ours_sampled.ply"), ours_sampled)
    mask_theirs = plane_normalized[:3] @ theirs_sampled.T > -plane_normalized[-1]
    theirs_sampled_masked = theirs_sampled[mask_theirs]
    save_pc(Path(result_path, "theirs_sampled_masked.ply"), theirs_sampled_masked)
    theirs_chamfer, theirs_chamfer_normals = chamfer_distance(theirs_sampled_masked.unsqueeze(0), gt_pc.unsqueeze(0))
    ours_chamfer, ours_chamfer_normals = chamfer_distance(ours_sampled.unsqueeze(0), gt_pc.unsqueeze(0))
    print("theirs: {}, ours: {}".format(theirs_chamfer.item(), ours_chamfer.item()))

def get_volSDF_model(volSDF_scan_number, device):
    volSDF_2_blendedMVS = {1: 1, 2: 7, 3: 8, 4: 12, 5:13, 6:15, 7:16, 8:17, 9:19}
    # doll, egg, buddah_head, angel, bull, robot, dog, bread, camera
    blendedMVS_scan_number = volSDF_2_blendedMVS[volSDF_scan_number]
    conf = ConfigFactory.parse_file("/data1/yotam/volsdf/code/confs/bmvs.conf")
    model = get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
    model = model.to(options.device)
    model_dir = "/data1/yotam/volsdf/exps/bmvs_{}/2022_12_02_13_08_42/checkpoints/ModelParameters".format(volSDF_scan_number)  # bull 5, robot 6, bread 8 
    model_path = Path(model_dir, "latest.pth")
    
    plane_path = Path("/data1/yotam/volsdf/data/BlendedMVS/planes/_{}.npy".format(blendedMVS_scan_number))
    plane = torch.tensor(np.load(plane_path), dtype=torch.float32, device=device)
    plane_normalized = plane / torch.norm(plane[:3])
    gt_mesh_path = Path("/data1/yotam/datasets/BlendedMVS/blendedMVS/scan{}/mesh.ply".format(blendedMVS_scan_number))
    gt_pc = prep_gt_pc(gt_mesh_path, plane_normalized, options)
    saved_model_state = torch.load(str(model_path))
    model.load_state_dict(saved_model_state["model_state_dict"])
    model.eval()
    poses = []
    # ks = []
    cameras = np.load(Path("/data1/yotam/volsdf/data/BlendedMVS/scan{}/cameras.npz".format(scan_number)))
    post_process_transform = torch.tensor(cameras['scale_mat_0'], device=device)
    for i in range(len(cameras) // 2):
        P0=cameras['world_mat_%d'%i][:3,:]
        # K=cv2.decomposeProjectionMatrix(P0)[0]
        R=cv2.decomposeProjectionMatrix(P0)[1]
        c=cv2.decomposeProjectionMatrix(P0)[2]
        c=c/c[3]
        M=np.eye(4)
        M[:3,:]= np.concatenate((R, -R @ c[:3]), axis=1)
        poses.append(M)
        # ks.append(K)
    # for file in sorted(Path("/data1/yotam/idr/data/DTU/poses").glob("*.txt")):
    #     with open(file, 'r') as filor:
    #         lines = filor.readlines()
    #     cur_pos = []
    #     for line in lines:
    #         pose = np.array([float(x) for x in line.split(" ")])
    #         cur_pos.append(pose)
    #     cur_pos = np.array(cur_pos)
    #     out = cv2.decomposeProjectionMatrix(cur_pos)
    #     K = out[0]
    #     R = out[1]
    #     t = out[2]
    #     pose = np.eye(4, dtype=np.float32)
    #     pose[:3, :3] = R.T
    #     t = t/t[3]
    #     pose[:3,-1] = (-R @ t[:3])[:, 0]
    #     poses.append(pose)
    poses = np.stack(poses)
    poses = np.linalg.inv(poses)
    # scan_proj = gsoup.to_torch(gsoup.opengl_project_from_opencv_intrinsics(np.stack(ks).mean(axis=0), 512, 512), device=options.device, dtype=torch.float32)
    # from gsoup import viewer
    coa = gsoup.get_center_of_attention(poses)
    poses[:, :3, 3] -= coa
    poses, factor = gsoup.scale_poses(poses, n=6.0)
    # v, f = gsoup.load_mesh("scan6.obj")
    # viewer.view(camera_poses=poses, meshes=[[v, f]])
    # exit()
    poses = gsoup.opencv_c2w_to_opengl_c2w(poses)
    view_matrix = np.linalg.inv(poses)
    mvs = view_matrix

    return model, mvs, plane_normalized, post_process_transform, gt_pc



# options.sdf_is_network = True
# if options.sdf_is_network:
#     sphere_tracer_res = [options.image_resolution, options.image_resolution]
#     batch_size = sphere_tracer_res[0] * sphere_tracer_res[1]
#     # model, mvs = get_IDR_model(scan_number)
#     model, mvs, plane_normalized, post_process_transform, gt_pc = get_volSDF_model(scan_number, options.device)
#     compare(scan_number, result_path, post_process_transform, options)
#     plane_transformed = post_process_transform.T @ plane_normalized
#     ########## remove floor ##############
#     # note: open a single normal image, and find the floor color, then replace the value below.
#     # floor_norm_color = np.array([126, 0, 126])  #buddah: [126, 0, 126]# robot: [207, 50, 64], bread: [128, 0, 123], dog: [126, 0, 126]
#     # floor_xyz = np.array([45, 0.0, 0.0])  # robot: [-1.0, 0.0, 0.0], bread: [-45.0, 0.0, 0.0], buddah: [45, 0.0, 0.0]
#     # avg_norm = ((floor_norm_color / 255)*2) - 1
#     # avg_norm /= np.linalg.norm(avg_norm)
#     # plane = sdf.plane(avg_norm, floor_xyz)
#     plane_transformed_normalized = plane_transformed / plane_transformed[:3].norm()
#     plane_normal = plane_transformed_normalized[:3]
#     plane_x = -plane_transformed_normalized[-1] / plane_transformed_normalized[0]
#     plane_sdf = sdf.plane(plane_normal.cpu(), torch.tensor([plane_x, 0, 0], dtype=torch.float32))
#     f = lambda x: torch.maximum(model.implicit_network(x)[:, 0:1], -plane_sdf(x))
#     # f = lambda x: model.implicit_network(x)[:, 0:1]
#     # f2 = lambda x: torch.maximum(model.implicit_network(x)[:, 0:1], plane_sdf(x))
#     ######## remove floor ###############
#     # f = lambda x: model.implicit_network(x)[:, 0:1]
#     # focus_on_center = False
#     init_mesh_path = None
#     more_cams = False
#     _, proj = make_star_cameras(6, 6, distance=8.0, device=options.device)
#     mv = torch.tensor(mvs, dtype=torch.float32, device=options.device)
# else:
#     mvs = None
#     more_cams = True
#     # init_mesh_path = None
#     init_mesh_path = Path(result_path, "march.obj")
#     sphere_tracer_res = [options.image_resolution, options.image_resolution]
#     batch_size = sphere_tracer_res[0] * sphere_tracer_res[1]
#     ### sdf
#     # f = sdf.box(np.array([0.5, 0.8, 1.0]))
#     # f = sdf.cog()
#     # f = sdf.cylinder(0.25)
#     # f = sdf.pyramid(1.0)
#     # f = sdf.slab(x0=-1.0/2, x1=1.0/2, y0=-1.5/2, y1=1.5/2, z0=-0.75/2, z1=0.75/2)#.k(5.0)
#     f, height = sdf.cogs(5, return_height=True)
#     # c = sdf.cylinder(0.25)
#     # f -= c.orient(sdf.X) | c.orient(sdf.Y) | c.orient(sdf.Z)
#     # f = sdf.cylinder3()
#     # f = sdf.bobby()
#     # f = sdf.knurling()
#     # f, height = sdf.gears(9)
#
#     f.save(init_mesh_path, samples=2**28)
#     # center_of_interest = torch.tensor([0.5, 0.5, height*0.75], device=options.device)
#     # r = 10.0
#     # mv3, _ = make_random_cameras(30, r=r, at=center_of_interest, upper_hemi=True, device=options.device)
#     # outside_sdf = torch.isclose(f(torch.inverse(mv3)[:, :3, -1]),torch.tensor([r-1], device=options.device)).squeeze()
#     # mv3 = mv3[outside_sdf][:10]
#     # f.save(Path(result_path, "init_mesh_raw.obj"), samples=2**28)
#     mv, proj = make_star_cameras(6, 6, distance=11.0, device=options.device)
#
# if more_cams:
#     cam_file_path = Path(result_path, "cameras.npz")
#     if cam_file_path.exists():
#         data = np.load(cam_file_path)
#         mv2, proj2 = data["mv"], data["proj"]
#         mv2 = torch.tensor(mv2, dtype=torch.float32, device=options.device)
#         proj2 = torch.tensor(proj2, dtype=torch.float32, device=options.device)
#     else:
#         center_of_interest = torch.tensor([0.7, 0.7, height], device=options.device)
#         r = 2.0
#         mv2, _ = make_random_cameras(30, r=r, at=center_of_interest, upper_hemi=True, device=options.device)
#         outside_sdf = torch.isclose(f(torch.inverse(mv2)[:, :3, -1]),torch.tensor([r-1.0], device=options.device)).squeeze()
#         mv2 = mv2[outside_sdf][:15]
#         proj2 = _projection(0.1, device=options.device)
#         np.savez(str(cam_file_path), proj=proj2.cpu().numpy(), mv=mv2.cpu().numpy())
#     proj = torch.cat((proj2[None, :, :].repeat(len(mv2), 1, 1),
#                     proj[None, :, :].repeat(len(mv), 1, 1)), dim=0)
#     mv = torch.cat((mv2, mv), dim=0)
#
#     # proj = scan_proj
#     # proj = torch.eye(4, dtype=torch.float32, device=options.device)[None, :, :]
#     # proj = _projection(r=1/3, device=options.device)
# # mv = mv[view_filter:]
# ###### main loop over files #######
# # current_time = datetime.now()
# # format_data = "%d-%m-%y-%H-%M-%S.%f"  # avoids double colon or spaces in the name
# # current_time = current_time.strftime(format_data)
#
# losses = []
# verts_n = []
# faces_n = []
#
# # if not Path(result_path, "marching_cubes_target.obj").exists():
# #     f.save(Path(result_path, "marching_cubes_target.obj"))
# # pool = MedianPool2d()
# mysdf = f
# # target_vertices = mysdf
# # target_faces = None
# image_dst = Path(result_path, "target_images")
# image_dst.mkdir(exist_ok=True, parents=True)
# ray_origins, ray_directions = sphere_tracer.generate_rays(mv, proj, sphere_tracer_res[0], sphere_tracer_res[1], options.device)
# # mysdf = csg.box(0.5)
# # mysdf = csg.sphere(0.5)
# rays = torch.cat([ray_origins, ray_directions], dim=-1)
# # images = []
# counts = [0]
# for i, ray in enumerate(rays):
#     print("view: {} / {}, counts: {}".format(i, len(rays), np.sum(counts)))
#     dst = Path(image_dst, "{:02d}.png".format(i))
#     dst_lossless = Path(image_dst, "{:02d}.npy".format(i))
#     if dst.exists():
#         continue
#     res = []
#     for j, batch in enumerate(torch.split(ray.view(-1, 6), batch_size, dim=0)):
#         print("batch {} / {}".format(j, ray.view(-1, 6).shape[0] // batch_size))
#         o = batch[:, :3]
#         d = batch[:, 3:]
#         traced, count = sphere_tracer.render(mysdf, o, d, grad=options.sdf_is_network, convergence_threshold=1e-4, count_access=True)
#         res.append(traced.detach().cpu().numpy())
#         counts.append(count)
#     image = np.concatenate(res, axis=0).reshape(sphere_tracer_res[0], sphere_tracer_res[1], 4)
#     # image = median_filter(image, size=(3, 3, 1))
#     # image = pool(torch.tensor(image, device=options.device).permute(2, 0, 1)).permute(1, 2, 0).cpu().numpy()
#     np.save(str(dst_lossless),image)
#     imageio.imwrite(str(dst), np.clip(image*255, 0, 255).astype(np.uint8))
#     # images.append(image)
# # target_images = np.stack(images)
# # target_images = torch.tensor(target_images, device=options.device)
# # target_images = load_images(Path(result_path, "target_images"), options.device)
# target_images = np.stack([np.load(x) for x in Path(result_path, "target_images").glob("*.npy")])
# target_images = torch.tensor(target_images, device=options.device)
# preproc_path = Path(result_path, "preprocess")
# preproc_path.mkdir(exist_ok=True)
# # if options.sdf_is_network:
# #     preprocessed = []
# #     for i, image in enumerate(target_images):
# #         np_alpha_channel = (image[..., -1].cpu().numpy()*255).astype(np.uint8)
# #         cnts, _ = cv2.findContours(np_alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# #         cnt = max(cnts, key=cv2.contourArea)
# #         out = np.zeros(np_alpha_channel.shape, np.uint8)
# #         cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
# #         preprocessed_alpha_channel = cv2.bitwise_and(np_alpha_channel, out)
# #         imageio.imwrite(str(Path(preproc_path, "{:02d}.png".format(i))), preprocessed_alpha_channel)
# #         preprocessed_alpha_channel = torch.tensor(preprocessed_alpha_channel.astype(np.float32)/255, device=options.device)
# #         image[..., -1] = preprocessed_alpha_channel
#
# if target_images.shape[-1] > 3:
#     save_images(target_images[..., 3:], str(Path(result_path, "target_alpha")))
#
# view_filter = np.array(remove_views, dtype=np.int32)
# mask = np.ones(mv.shape[0], dtype=np.bool)
# mask[view_filter] = False
# mv = mv[mask]
# target_images = target_images[mask]
#
# renderer = NormalsRenderer(mv, proj, [options.image_resolution, options.image_resolution], device=options.device)
# # target_images = target_images[view_filter:]
# ### begin remeshing ###
# if options.voxelize_input:
#     voxelizer = Voxelizer(device=options.device,size_percentage=options.size_percent,
#                             max_voxel_amount=options.voxel_limit,
#                             samples_per_voxel=options.samples_per_voxel,
#                             mode="sdf")
#     bounds = torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32, device=options.device)
#     vertices, faces = voxelizer.voxelize(bounds, None, 5000, verbose=True, sdf=f)
#
# else:
#     if init_mesh_path is not None:  # override
#         options.init_mesh_path = init_mesh_path
#     if options.init_mesh_path is not None:
#         vertices, faces, valid = load_mesh(options.init_mesh_path, device=options.device)
#         import igl
#         vertices, faces, _ = igl.remove_duplicates(vertices.cpu().numpy(), faces.cpu().numpy(), 1e-6)
#         vertices = torch.tensor(vertices, dtype=torch.float32, device=options.device)
#         faces = torch.tensor(faces, device=options.device)
#         # vertices = normalize_vertices(vertices)
#         # vertices = target_vertices.clone()
#         # faces = target_faces.clone()
#         if not valid:
#             assert AssertionError("invalid initial mesh")
#     # elif Path(result_path, "marching_cubes_target_dec.obj").exists():
#         # vertices, faces, valid = load_mesh(Path(result_path, "marching_cubes_target_dec.obj"), device=options.device)
#     else:
#         vertices, faces = make_sphere(level=options.level, radius=0.5)
# save_mesh_properly(vertices, faces, str(Path(result_path, "init_mesh.obj")))
# init_vertex_normals = calc_vertex_normals(vertices, faces)
# init_images, _ = renderer.render(vertices, init_vertex_normals, faces)
# save_images(init_images, str(Path(result_path, "init_images")))
#
# opt = MeshOptimizer(vertices, faces, options)
#
# vertices = opt.vertices
# snapshots = []
# pbar = tqdm(range(options.steps + 1))
# min_loss = torch.ones(1).to(options.device)
# gt_chamfer = torch.ones(1).to(options.device)
# patience = 0
# best_step = 0
# for j in pbar:
#     patience += 1
#     opt.zero_grad()
#     ### losses ###
#     # loss = f(vertices).abs().mean()
#
#     normals = calc_vertex_normals(vertices, faces)
#     # images = images[..., :3]*images[..., -1:]
#     # target_images[..., :3]*target_images[..., -1:]
#     # images = pool(images.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
#     #soft
#     # weights = f(torch.inverse(mv)[:, :3, -1]).detach().squeeze()
#     # weights = weights ** 2
#     # cur_weights = torch.softmax(weights, dim=0)
#     # inverse_weights = torch.softmax(1/weights, dim=0)
#     # factor = j / (options.steps + 1)
#     # cur_weights = weights * (1 - factor) + inverse_weights * factor
#     # loss = ((images - target_images).abs().mean(dim=[1, 2, 3])*cur_weights).mean()
#     #hard
#     # if j > 1000:
#         # images, extra = renderer.render(vertices, normals, faces, mv=mv[:15], projection=proj[:15], get_visible_faces=True)
#         # loss = (images[:15] - target_images[:15]).abs().mean()
#     # else:
#         # images, extra = renderer.render(vertices, normals, faces, mv=mv[15:], projection=proj[15:], get_visible_faces=True)
#         # loss = (images - target_images[15:]).abs().mean()
#     #naive
#     images, extra = renderer.render(vertices, normals, faces, get_visible_faces=True)
#     loss = (images - target_images).abs().mean()
#     if j > 500:
#         projected = face_split_sdf(vertices, f)
#         proj_loss = 0.1*torch.mean((projected - vertices).norm(dim=-1))
#         # proj_loss = torch.mean(torch.abs(f(vertices)))
#         loss += proj_loss
#     # if opt.k is not None:
#     #     loss += edge_length_loss(vertices, faces, opt._k_hop_len)\
#     losses.append(loss.item())
#     verts_n.append(vertices.shape[0])
#     faces_n.append(faces.shape[0])
#     ### save best ###
#     if loss <= min_loss:
#         best_step = j
#         min_loss = loss
#         patience = 0
#     pbar.set_postfix({'loss': loss.item(),
#                     #   'gt_chamfer': gt_chamfer,
#                       'minloss': min_loss.item(),
#                       'cur_n_faces': len(faces),
#                       'best_step': best_step})
#     loss.backward()
#     # face_weights = torch.zeros(faces.shape[0], device=options.device)
#     # face_weights[extra["vis_faces"]] = 1. #extra["counts"].to(torch.float32)
#     # vertex_weights = calculate_vertex_incident_scalar(len(vertices), faces, face_weights, avg=True).squeeze()
#     # vertex_weights = vertex_weights > 0
#     # vertices.grad *= vertex_weights[:, None]
#     ### step ###
#     step_snapshot_data, _ = opt.step()  # vertex_weights
#     ### remesh ###
#     vertices, faces, snapshot_data = opt.remesh(f, None, None, time_step=j, with_grad=options.sdf_is_network, plane_sdf=plane_sdf)
#     # merge_dicts(snapshot_data, step_snapshot_data, 'step_')
#     if j % options.snapshot_interval == 0 and options.save_snapshots:
#         snapshots.append(snapshot(opt, snapshot_data))
#     if options.save_mesh_interval > 0 and j % options.save_mesh_interval == 0:
#         save_mesh(vertices, faces, str(Path(result_path, "{:04d}.obj".format(j))))
#     if patience > options.max_patience:
#         print("reached max patience !")
#         break
# gt_chamfer = chamfer_with_gt(vertices, faces, gt_pc, plane_normalized, options)
# print("gt_chamfer", gt_chamfer)
# vertices_to_save, faces_to_save = vertices.detach().clone(), faces.detach().clone()
# save_path = Path(result_path, name + ".obj")
# save_path_to_gt = Path(result_path, name + "_gt_coords.obj")
# v_gt = post_process_transform @ torch.cat((vertices_to_save, torch.ones_like(vertices_to_save[:, 0:1])), dim=-1).T
# v_gt = v_gt[:3].T
# save_mesh_properly(v_gt, faces_to_save, str(save_path_to_gt))
# save_mesh_properly(vertices_to_save, faces_to_save, str(save_path))
# if options.save_best_render:
#     vertices_to_save, faces_to_save = vertices.detach().clone(), faces.detach().clone()
#     normals = calc_vertex_normals(vertices_to_save, faces_to_save)
#     final_images, _ = renderer.render(vertices_to_save, normals, faces_to_save)
#     if options.use_silhouette_only:
#         final_images = final_images[..., 3:]
#     save_images(final_images, str(Path(result_path, "final_images")))
#
# if options.save_snapshots:
#     # for ss in snapshots:
#     #     step = ss.step
#     #     v = ss.step_data["step_v_raw"]
#     #     f = ss.step_data["step_f_raw"]
#     #     grad = ss.step_data["step_grad_raw"]
#     #     grad_lap = ss.step_data["step_grad_laplace"]
#     #     dv = ss.step_data["step_update_v"]
#     #     np.savez(Path(result_path, 'polyscope_data', 'step_{:05d}'.format(step)), v=v, f=f, grad=grad, grad_lap=grad_lap, dv=dv, step=step)
#     theirs_path = next(Path("/data1/yotam/volsdf/evals/bmvs_{}".format(scan_number)).glob("**/*.ply"))
#     theirs_v, theirs_f = load_mesh_o3d(theirs_path)
#     theirs_v = torch.tensor(theirs_v, dtype=torch.float32).to(options.device)
#     theirs_v = torch.inverse(post_process_transform) @ torch.cat((theirs_v, torch.ones_like(theirs_v[:, 0:1])), dim=-1).T
#     theirs_v = theirs_v[:3].T
#     theirs_f = torch.tensor(theirs_f).to(options.device)
#     save_output_for_polyscope(snapshots, result_path, theirs_v, theirs_f,
#                                 output_only_obj=options.output_obj_only)
#
# # np.savetxt(Path(result_path, "losses.txt"), np.array(losses))
# # np.savetxt(Path(result_path, "verts_n.txt"), np.array(verts_n))
# # np.savetxt(Path(result_path, "faces_n.txt"), np.array(faces_n))
