from pathlib import Path
import torch
import configargparse
import yaml

class Config:
    def __init__(self):
        self.parser = configargparse.ArgumentParser()
        # self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # data params
        self.parser.add_argument('--config', is_config_file=True,
                                 help='config file path')
        self.parser.add_argument('--root_dir', required=True,
                                 help='path to meshes or pointclouds (.obj or .ply files, respectively)')
        self.parser.add_argument('--experiment_name', type=str, default='run')
        self.parser.add_argument('--no_timestamp', action="store_true", help="don't add timestamp to experiment name")
        self.parser.add_argument('--level', type=int, default=2, help="number of subdivision of initial icosphere")
        self.parser.add_argument('--icosphere_radius', type=int, default=0.5, help="intial icosphere radius")
        self.parser.add_argument('--voxelize_input', action="store_true", default=False,
                                 help="will voxelize input as initial mesh if provided.")
        self.parser.add_argument('--target_names', type=str, nargs='+',
                                 help='list of names of target meshes (without .obj)')
        self.parser.add_argument('--xyz_colors', action="store_true", default=False,
                                 help='use xyz colors instead of normals')
        self.parser.add_argument('--preproc_only', action="store_true", help="if true only preprocess data and exit")
        # point cloud preprocess params
        self.parser.add_argument("--winding_number_threshold", type=float, default=0.49)
        self.parser.add_argument("--point_cloud_num_samples", type=int, default=150000)
        self.parser.add_argument("--point_cloud_delta", type=float, default=0.2)
        # optimizer params
        self.parser.add_argument('--seed', type=int, default=43)
        self.parser.add_argument('--steps', type=int, default=3500)
        self.parser.add_argument('--max_faces', type=int, default=4500)
        self.parser.add_argument('--face_score_mode', choices={'dihedral_angles', 'face_folds'},
                                 default='dihedral_angles')
        self.parser.add_argument('--sdf_model_class', type=str, default="", help='class name of the SDF model')
        self.parser.add_argument('--sdf_model_path', type=str, default="")
        self.parser.add_argument('--save_snapshots', action='store_true', help="will output data for debugging")
        self.parser.add_argument('--snapshot_interval', type=int, help='# of steps between snapshots', default=500)
        self.parser.add_argument('--save_mesh_interval', type=int,
                                 help='# of steps between mesh saves. 0 for not to save', default=0)
        self.parser.add_argument('--folded_face_cleanup_interval', type=int, default=1,
                                 help='# of steps between cleanup of folded faces')
        self.parser.add_argument('--edge_cleanup_interval', type=int, default=20,
                                 help='# of steps between removal of redundant faces through edge collapse')
        self.parser.add_argument('--edge_collapse_only', action='store_true')
        self.parser.add_argument('--improve_quality_interval', type=int, default=5,
                                 help='# of steps between applying adaptive botsch remeshing step')
        self.parser.add_argument('--face_split_interval', type=int, default=20,
                                 help='# of steps between face splittings')
        self.parser.add_argument('--topology_delay', type=int, default=20,
                                 help='# iterations to start remeshing')
        self.parser.add_argument('--rendering_backend', choices={'Pulsar', 'NvDiffRast'}, default='NvDiffRast',
                                 help="NvDiffRast is better but currently only Pulsar supports pointcloud targets")

        self.parser.add_argument('--rendering_mode', choices={'pointcloud', 'mesh', 'non_manifold_mesh', 'sdf'},
                                 default='mesh',
                                 help="required target format")
        self.parser.add_argument('--random_views', choices={'no', 'yes', 'each_iter'}, default='no',
                                 help="required target format")
        self.parser.add_argument('--num_views', type=int, default=6,
                                 help="# views will be <num_views>**2")
        self.parser.add_argument('--image_resolution', type=int, default=512)
        self.parser.add_argument('--face_split_mode', choices={'mesh', 'sdf'},
                                 default='mesh')
        self.parser.add_argument('--reconstruction_mode', choices={'mesh', 'sdf'},
                                 default='sdf')
        self.parser.add_argument('--geometry_sampling_mode', choices={'uniform', 'random'}, default='random')
        self.parser.add_argument('--step_mode', choices={'paflinger', 'wenzel', 'none'}, default='paflinger')
        self.parser.add_argument('--lref_mode', choices={'palfinger', 'ours', 'none'}, default='ours')
        self.parser.add_argument('--k', type=int, help='neighbourhood for quality correction', default=0)
        self.parser.add_argument('--lr', type=float, help='learning rate', default=0.3)
        self.parser.add_argument('--lr_final', type=float, help='learning rate final', default=1e-2)
        self.parser.add_argument('--betas', nargs='+', type=float, default=[0.8, 0.8, 0],
                                 help='betas[0:2] are the same as in Adam, betas[2] may be used to time-smooth the relative velocity nu')
        self.parser.add_argument('--gammas', nargs='+', type=float, default=[0, 0, 0],
                                 help='optional spatial smoothing for m1,m2,nu, values between 0 (no smoothing) and 1 (max. smoothing)')
        self.parser.add_argument('--nu_ref', type=float, default=0.3,
                                 help='reference velocity for edge length controller')
        self.parser.add_argument('--edge_len_lims', nargs='+', default=[.005, .15],
                                 help='smallest and largest allowed reference edge length')
        self.parser.add_argument('--gain', type=float, default=0.2,
                                 help='gain value for edge length controller')
        self.parser.add_argument('--laplacian_weight', type=float, default=0.02,
                                 help='for laplacian smoothing/regularization')
        self.parser.add_argument('--ramp', type=int, default=1,
                                 help='learning rate ramp, actual ramp width is ramp/(1-betas[0])')
        self.parser.add_argument('--grad_lim', type=float, default=10,
                                 help='gradients are clipped to m1.abs()*grad_lim')
        self.parser.add_argument('--delta', type=float, default=0.2 / 100, help="regular sampling delta")
        self.parser.add_argument('--max_samples', type=int, default=20, help="regular sampling max sampling points")
        self.parser.add_argument('--scratchpad_resolution', type=int, default=30000,
                                 help="amount of faces for exploration")

        # initialization params
        self.parser.add_argument('--init_resolution', type=int, default=10000,
                                 help='resolution for initial mesh creation')
        self.parser.add_argument('--init_num_faces', type=int, default=20000,
                                 help='maximal number of faces for initial mesh')

        geometry_opt_group = self.parser.add_argument_group('init_geometry_options')
        geometry_opt_group.add_argument('--init_mesh_mode', type = str, help="path for initial mesh")
        geometry_opt_group.add_argument('--init_mesh_info', type = yaml.safe_load, help="path for initial mesh", default= {'test1': 1,
                                                                                                                 'test2': 2})

        # sdf params
        self.parser.add_argument('--net', type=str, default='OverfitSDF',
                               help='The network architecture to be used.')
        self.parser.add_argument('--jit', action='store_true',
                               help='Use JIT.')
        self.parser.add_argument('--pos-enc', action='store_true',
                               help='Use positional encoding.')
        self.parser.add_argument('--feature-dim', type=int, default=32,
                               help='Feature map dimension')
        self.parser.add_argument('--feature-size', type=int, default=4,
                               help='Feature map size (w/h)')
        self.parser.add_argument('--joint-feature', action='store_true',
                               help='Use joint features')
        self.parser.add_argument('--num-layers', type=int, default=1,
                               help='Number of layers for the decoder')
        self.parser.add_argument('--num-lods', type=int, default=1,
                               help='Number of LODs')
        self.parser.add_argument('--base-lod', type=int, default=2,
                               help='Base level LOD')
        self.parser.add_argument('--ff-dim', type=int, default=-1,
                               help='Fourier feature dimension.')
        self.parser.add_argument('--ff-width', type=float, default='16.0',
                               help='Fourier feature width.')
        self.parser.add_argument('--hidden-dim', type=int, default=128,
                               help='Network width')
        self.parser.add_argument('--pretrained', type=str,
                               help='Path to pretrained model weights.')
        self.parser.add_argument('--periodic', action='store_true',
                               help='Use periodic activations.')
        self.parser.add_argument('--skip', type=int, default=None,
                               help='Layer to have skip connection.')
        self.parser.add_argument('--freeze', type=int, default=-1,
                               help='Freeze the network at the specified epoch.')
        self.parser.add_argument('--pos-invariant', action='store_true',
                               help='Use a position invariant network.')
        self.parser.add_argument('--joint-decoder', action='store_true',
                               help='Use a single joint decoder.')
        self.parser.add_argument('--feat-sum', action='store_true',
                               help='Sum the features.')

        # Sphere tracer params
        self.parser.add_argument('--grad-method', type=str, choices=['autodiff', 'finitediff'],
                                 default='finitediff', help='Mode of gradient computations.')

        # Arguments for renderer
        self.parser.add_argument('--sol', action='store_true',
                                    help='Use the SOL mode renderer.')
        self.parser.add_argument('--render-res', type=int, nargs=2, default=[512, 512],
                                    help='Width/height to render at.')
        self.parser.add_argument('--render-batch', type=int, default=0,
                                    help='Batch size for batched rendering.')
        self.parser.add_argument('--matcap-path', type=str,
                                    default='data/matcap/green.png',
                                    help='Path to the matcap texture to render with.')
        self.parser.add_argument('--camera-origin', type=float, nargs=3, default=[-2.8, 2.8, -2.8],
                                    help='Camera origin.')
        self.parser.add_argument('--camera-lookat', type=float, nargs=3, default=[0, 0, 0],
                                    help='Camera look-at/target point.')
        self.parser.add_argument('--camera-fov', type=float, default=30,
                                    help='Camera field of view (FOV).')
        self.parser.add_argument('--camera-proj', type=str, choices=['ortho', 'persp'], default='persp',
                                    help='Camera projection.')
        self.parser.add_argument('--camera-clamp', nargs=2, type=float, default=[-5, 20],
                                    help='Camera clipping bounds.')
        self.parser.add_argument('--lod', type=int, default=None,
                                    help='LOD level to use.')
        self.parser.add_argument('--interpolate', type=float, default=None,
                                    help='LOD interpolation value')
        self.parser.add_argument('--render-every', type=int, default=1,
                                    help='Render every N epochs')
        self.parser.add_argument('--num-steps', type=int, default=256,
                                    help='Number of steps')
        self.parser.add_argument('--step-size', type=float, default=1.0,
                                    help='Scale of step size')
        self.parser.add_argument('--min-dis', type=float, default=0.0003,
                                    help='Minimum distance away from surface')
        self.parser.add_argument('--ground-height', type=float,
                                    help='Ground plane y coords')
        self.parser.add_argument('--tracer', type=str, default='SphereTracer',
                                    help='The tracer to be used.')
        self.parser.add_argument('--ao', action='store_true',
                                    help='Use ambient occlusion.')
        self.parser.add_argument('--shadow', action='store_true',
                                    help='Use shadowing.')
        self.parser.add_argument('--shading-mode', type=str, default='matcap',
                                    help='Shading mode.')

        # run params
        self.parser.add_argument('--device', type=int, default='0')
        self.parser.add_argument('--use_silhouette_only', action='store_true',
                                 help="renderer will not use shading in optimization")
        self.parser.add_argument('--eps_val', type=int, default=1e-8,
                                 help='very small nonzero value used')
        self.parser.add_argument('--save_best_render', action='store_true', help='save final render')
        self.parser.add_argument('--max_patience', type=int, default=2000,
                                 help='number of allowed steps with no loss improvement')
        self.parser.add_argument('--output_for_video', action='store_true',
                                 help='output data for video creation using blender script')
        self.parser.add_argument('--output_obj_only', action='store_true',
                                 help='save only final mesh in snapshots for a step')
        self.parser.add_argument('--use_nn', action='store_true', default=False)
        self.parser.add_argument('--shapenet_class', type=str, default='airplane')  # sofa
        self.parser.add_argument('--use_qslim', action='store_true')
        self.parser.add_argument('--only_topology', action='store_true')
        self.parser.add_argument('--s2t_neighbours', type=int, default=10)
        self.parser.add_argument('--max_scratchpad_faces', type=int, default=50000,
                                 help="amount of faces for exploration")
        self.parser.add_argument('--regime_iterations', type=str, default="500 250 20",
                                 help='explore,exploit and finalize iterations')
        self.parser.add_argument('--only_final', action='store_true',
                                 help='deletes all intermediate saved data on completion')
        self.initialized = True

        # thresholds
        self.parser.add_argument('--normal_dot_prod_threshold', type=float, default=0.5)
        self.parser.add_argument('--out_of_face_penalty', type=float, default=10)
        self.parser.add_argument('--max_quality_att', type=float, default=5)
        # projection loss
        self.parser.add_argument('--projection_loss_vertex_norm_threshold', type=float, default=0.45)

        # voxelization params
        self.parser.add_argument('--voxel_limit', type=float, default=500000)
        self.parser.add_argument('--size_percent', type=float, default=0.013)
        self.parser.add_argument('--voxelization_surface_samples', type=float, default=280000)
        self.parser.add_argument('--samples_per_voxel', type=float, default=3)
        self.parser.add_argument('--voxelization_time_limit', type=float, default=240)
        self.parser.add_argument('--voxelization_face_budget', type=float, default=10000)

        # color params
        self.parser.add_argument('--color_weight', type=float, default=1.0)

        # improve quality params
        self.parser.add_argument('--gap', type=float, default=0.3)
        self.parser.add_argument('--tangential_relaxation_threshold', type=float, default=0.9)
        self.parser.add_argument('--k_weight', type=float, default=0.1)

        # edge flip params
        #self.parser.add_argument('--dihedral_angle_threshold', type=float, default=0.9999)
        self.parser.add_argument('--edge_flip_degree_weight', type=float, default=1.0)


        # face fold params
        self.parser.add_argument('--face_fold_threshold', type=float, default=0.7)
        self.parser.add_argument('--face_fold_vertex_norm_threshold', type=float, default=0.3)
        self.parser.add_argument('--finalize_iters', type=int, default=5)

        # edge collapse params
        self.parser.add_argument('--collapse_threshold', type=float, default=0.7)
        self.parser.add_argument('--collapse_vertex_norm_threshold', type=float, default=0.3)
        self.parser.add_argument('--potential_score_coeff', type=float, default=0.3,
                                 help='for edge score')

        # qslim params
        self.parser.add_argument('--qslim_avoid_folding_faces', action="store_true")
        self.parser.add_argument('--collapse_quality_threshold', type=float, default=-1,
                                 help='from 1 to inf, smaller = stricter quality condition in collapse, -1 means None')

        # face split params
        self.parser.add_argument('--face_curvature_threshold', type=float, default=0.00,
                                help='controls size of captured target features')
        self.parser.add_argument('--edge_curvature_threshold', type=float, default=1e-3,
                                 help='controls size of captured target features')
        self.parser.add_argument('--edge_flip_iters', type=int, default=6,
                                 help='controls size of captured target features')

        # edge collapse
        self.parser.add_argument('--edge_length_coeff', type=float, default=0.3)
        self.parser.add_argument('--face_score_coeff', type=float, default=0.1)

        # projection loss parameters
        self.parser.add_argument('--projection_alpha', type=float, default=0.3)
        self.parser.add_argument('--projection_beta', type=float, default=0.1)

        # weights
        self.parser.add_argument('--quality_weight', type=float, default=0.0,
                                 help='weight for triangle quality')

    def parse(self, args = None):
        if not self.initialized:
            self.initialize()
        if args is None:
            self.opt = self.parser.parse()
        else:
            self.opt = self.parser.parse_args(args=args)
        self.opt.root_dir = Path(self.opt.root_dir)
        # if self.opt.init_mesh_path is not None:
        #     assert self.opt.voxelize_input == False, "voxelize_input and init_mesh_path are mutually exclusive"
        # if self.opt.device == "-1":
        #     self.opt.device = "cpu"
        # else:
        #     self.opt.device = "cuda:{}".format(self.opt.device)
        #     torch.cuda.set_device(self.opt.device)
        if self.opt.collapse_quality_threshold < 0:
            self.opt.collapse_quality_threshold = None
        # if self.opt.k == 0:
        #     self.opt.k = None
        # if self.opt.init_mesh_path == "":
        #     self.opt.init_mesh_path = None
        return self.opt


    @staticmethod
    def save_to_file(folder, options):
        with open(Path(folder, 'args.txt'), 'w') as f:
            for i, arg in enumerate(vars(options)):
                key = arg
                value = getattr(options, arg)
                if (i != 0):
                    f.write('\n')
                f.write('--' + str(key) + '\n' + str(value))


if __name__ == "__main__":
    """
    create cfg file
    """
    options = Config().parse()
    pass