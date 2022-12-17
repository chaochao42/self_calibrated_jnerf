dataset = dict(
    type = 'NeuSDataset',
    dataset_dir = '/home/hj2/3d/jnerf/data/baseline_dtu/dtu_scan24_womask',
    render_cameras_name = 'cameras_sphere.npz',
    object_cameras_name = 'cameras_sphere.npz',
)

encoder = dict(
    nerf_pos_encoder = dict(
        type='FrequencyEncoder',
        multires=10,
        input_dims=4,
    ),
    nerf_dir_encoder = dict(
        type='FrequencyEncoder',
        multires=4,
        input_dims=3,
    ),
    sdf_encoder = dict(
        type='FrequencyEncoder',
        multires=6,
        input_dims=3,
    ),
    rendering_encoder = dict(
        type='FrequencyEncoder',
        multires=4,
        input_dims=3,
    ),
)

model = dict(
    type = 'NeuS',
    nerf_network = dict(
        D = 8,
        W = 256,
        output_ch = 4,
        skips=[4],
        use_viewdirs=True
    ),
    sdf_network = dict(
        d_out = 257,
        d_hidden = 256,
        n_layers = 8,
        skip_in = [4],
        bias = 0.5,
        scale = 1.0,
        geometric_init = True,
        weight_norm = True,
    ),
    variance_network = dict(
        init_val = 0.3,
    ),
    rendering_network = dict(
        d_feature = 256,
        mode = 'idr',
        d_out = 3,
        d_hidden = 256,
        n_layers = 4,
        weight_norm = True,
        squeeze_out = True,
    ),
)

render = dict(
    type = 'NeuSRenderer',
    n_samples = 64,
    n_importance = 64,
    n_outside = 32,
    up_sample_steps = 4,     # 1 for simple coarse-to-fine sampling
    perturb = 1.0,
)

optim = dict(
    type='Adam',
    lr=5e-4,
    eps=1e-15,
    betas=(0.9,0.99),
)

base_exp_dir = '../log/baseline/dtu_scan24_womask'
recording = [ './','./models']

learning_rate_alpha = 0.05
end_iter = 300000

batch_size = 512
validate_resolution_level = 4
warm_up_end = 5000
anneal_end = 50000
use_white_bkgd = False

save_freq = 10000
val_freq = 200
val_mesh_freq = 5000
report_freq = 100

igr_weight = 0.1
mask_weight = 0.0