from yacs.config import CfgNode as CN


cfg = CN()

cfg.device = 'cuda'

cfg.dist_backend = 'nccl'

cfg.log_dir = 'logs/'
cfg.output_dir = 'outputs/'
cfg.result_dir = 'results/'

cfg.seed = 42

cfg.workers = 4

cfg.pi = 'psnr'
cfg.cam_num = -1
cfg.fix_human = False
cfg.fix_pose = False
cfg.mask_bkgd = True
cfg.src_view_num = 3
cfg.num_frame = 200
cfg.xyz_res = 10
cfg.view_res = 4


# dataset
cfg.dataset = CN()

cfg.dataset.img_num_per_gpu = 1

cfg.dataset.H = 1024
cfg.dataset.W = 1024
cfg.dataset.ratio = 0.5
cfg.dataset.voxel_size = [0.005, 0.005, 0.005]

cfg.dataset.train = CN()
cfg.dataset.train.name = 'thuman'
cfg.dataset.train.data_root = 'data/thuman/'
cfg.dataset.train.file = 'CustomDataset'
cfg.dataset.train.dataset_cams = 24
cfg.dataset.train.sampler = ''
cfg.dataset.train.batch_sampler = 'default'
cfg.dataset.train.sampler_meta = CN({'min_hw': [256, 256], 'max_hw': [480, 640], 'strategy': 'range'})
cfg.dataset.train.drop_last = True
cfg.dataset.train.shuffle = True
cfg.dataset.train.seq_list = []
cfg.dataset.train.interval = 1
cfg.dataset.train.chunk = 400

cfg.dataset.test = CN()
cfg.dataset.test.name = 'zju_mocap'
cfg.dataset.test.data_root = 'data/zju_mocap/'
cfg.dataset.test.file = 'ZjumocapDataset'
cfg.dataset.test.dataset_cams = 24
cfg.dataset.test.sampler = ''
cfg.dataset.test.batch_sampler = 'default'
cfg.dataset.test.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})
cfg.dataset.test.drop_last = False
cfg.dataset.test.shuffle = False
cfg.dataset.test.seq_list = ['CoreView_315',]
cfg.dataset.test.interval = 7
cfg.dataset.test.chunk = 2000


# network render
cfg.render = CN()
cfg.render.file = 'BaseRender'
cfg.render.resume_path = ''

# encoder
cfg.encoder = CN()
cfg.encoder.name = 'resnet34'
cfg.encoder.file = 'UNet'
cfg.encoder.out_ch = 32

# nerfhead
cfg.head = CN()
cfg.head.file = 'BaseNeRFHead'
# head rgb branch
cfg.head.rgb = CN()
cfg.head.rgb.use_rgbhead = True
# head sigma branch
cfg.head.sigma = CN()
cfg.head.sigma.code_dim = 16
cfg.head.sigma.n_heads = 4
cfg.head.sigma.n_layers = 4
cfg.head.sigma.n_smpl = 6890
cfg.head.sigma.outdims = [32, 32, 32, 32]



# train
cfg.train = CN()

cfg.train.file = 'BaseTrainer'
cfg.train.criterion_file = 'BaseNeRFCriterion'

cfg.train.resume = False

cfg.train.body_sample_ratio = 0.5
cfg.train.n_rays = 1024
cfg.train.n_samples = 64

cfg.train.ep_iter = 500
cfg.train.lr = 1e-4
cfg.train.gamma = 0.1
cfg.train.decay_epochs = 1000
cfg.train.weight_decay = 0.0001
cfg.train.max_epoch = 1000

cfg.train.print_freq = 10
cfg.train.save_every_checkpoint = True
cfg.train.save_interval = 1
cfg.train.valiter_interval = 100
cfg.train.val_when_train = False


# test
cfg.test = CN()

cfg.test.save_imgs = True
cfg.test.test_seq = 'CoreView_315'
cfg.test.is_vis = False
cfg.test.mesh_th = 50

def update_config(config, args):
    config.defrost()
    # set cfg using yaml config file
    config.merge_from_file(args.yaml_file)
    # update cfg using args
    config.merge_from_list(args.opts)
    config.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
