# Test seen human bodies, unseen human poses
CUDA_VISIBLE_DEVICES=0 python3 tools/inference.py --cfg configs/trainzju_valzju.yaml render.resume_path checkpoints/cam3zju_zjuval_70.pth test.test_seq 'demo_zjutrain_zjuval' dataset.test.seq_list "'CoreView_387', 'CoreView_393', 'CoreView_392'," \
	test.is_vis True dataset.test.sampler 'FrameSampler' dataset.test.shuffle False render.file 'demo_render'

# Test seen human bodies, unseen human poses
CUDA_VISIBLE_DEVICES=0 python3 tools/inference.py --cfg configs/trainthu_valzju.yaml render.resume_path checkpoints/cam3thu_zjuval_dim16_100.pth test.test_seq 'demo_thutrain_zjuval' dataset.test.seq_list "'CoreView_387', 'CoreView_393', 'CoreView_392'," \
	test.is_vis True dataset.test.sampler 'FrameSampler' dataset.test.shuffle False render.file 'demo_render' head.sigma.code_dim 16
