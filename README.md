## GP-NeRF: Geometry-Guided Progressive NeRF for Generalizable and Efficient Neural Human Rendering (ECCV2022)

[Paper](https://arxiv.org/abs/2112.04312)

### ENV

- System: Ubuntu 16.04
- Nvidia: `TITAN Xp`
- Torch version: 1.4.0+cu100, torchvision: 0.4.0


### Install Env

We provide two ways to install the environment:
1. Conda-based:

```
conda create -n gpnerf python=3.7
conda activate gpnerf

# make sure that the pytorch cuda is consistent with the system cuda
# e.g., if your system cuda is 10.0, install torch 1.4 built from cuda 10.0
pip install torch==1.4.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# Install spconv
git clone https://github.com/traveller59/spconv --recursive
cd spconv
git checkout abf0acf30f5526ea93e687e3f424f62d9cd8313a
export CUDA_HOME="/usr/local/cuda-10.0"
python setup.py bdist_wheel
cd dist
pip install spconv-1.2.1-cp36-cp36m-linux_x86_64.whl
```

2. Docker-based:
```
docker pull caiyj/neuralbody:imi-server
# if there's any problem with spconv, try the following commands
cd ~/spconv
python setup.py bdist_wheel
cd dist
pip uninstall spconv-1.2.1-cp36-cp36m-linux_x86_64.whl
pip install spconv-1.2.1-cp36-cp36m-linux_x86_64.whl
pip install yacs
```

### Set up datasets

1. Download the ZJU-Mocap dataset [here](https://zjueducn-my.sharepoint.com/:f:/g/personal/pengsida_zju_edu_cn/Eo9zn4x_xcZKmYHZNjzel7gBdWf_d4m-pISHhPWB-GZBYw?e=Hf4mz7).
2. Create a soft link:
    ```
    ROOT=/path/to/neuralbody
    cd $ROOT/data
    ln -s /path/to/zju_mocap zju_mocap
    ```

### Train

Train on ZJU-Mocap dataset, eval on ZJU-Mocap dataset:

`CUDA_VISIBLE_DEVICES=0 python tools/train.py -cfg configs/trainzju_valzju.yaml`

Train on THuman dataset, eval on ZJU-Mocap dataset:

`CUDA_VISIBLE_DEVICES=0 python3 tools/train.py --cfg configs/trainthu_valzju.yaml`


### Evaluation and Visualization

Train on ZJU-Mocap dataset, eval on ZJU-Mocap dataset:

`CUDA_VISIBLE_DEVICES=0 python3 tools/inference.py --cfg configs/trainzju_valzju.yaml render.resume_path checkpoints/cam3zju_zjuval_70.pth test.test_seq 'demo_zjutrain_zjuval' dataset.test.seq_list "'CoreView_387', 'CoreView_393', 'CoreView_392'," \
	test.is_vis True dataset.test.sampler 'FrameSampler' dataset.test.shuffle False render.file 'demo_render'`

Train on THuman dataset, eval on ZJU-Mocap dataset:

`CUDA_VISIBLE_DEVICES=0 python3 tools/inference.py --cfg configs/trainthu_valzju.yaml render.resume_path checkpoints/cam3thu_zjuval_dim16_100.pth test.test_seq 'demo_thutrain_zjuval' dataset.test.seq_list "'CoreView_387', 'CoreView_393', 'CoreView_392'," \
	test.is_vis True dataset.test.sampler 'FrameSampler' dataset.test.shuffle False render.file 'demo_render' head.sigma.code_dim 16`

Visualization setting: test.is_vis = True, visualization dir path = [result_dir]/[test.test_seq].

We have provided two pretrained weights files and the corresponding training logs and visualization results [here](https://drive.google.com/drive/folders/136QXKFZlNUc4Q1WM5zb2BPunUZbynwnk?usp=sharing) for reference. 


### Notice

- If loading the provided checkpoint, please make sure the version of PyTorch and spconv are consistent with the provided code.
- You can change the dataset root by modifying dataset.train.data_root and dataset.test.data_root in the config file.
- Please set dataset.test.sampler to '' if you want to test every frame of the test set.
- To ensure the testing performance, the environment especially PyTorch version and spconv version should be consistent between the training and testing.

### Acknowledgement
- [Neuralbody](https://github.com/zju3dv/neuralbody)
- [SparseConv](https://github.com/traveller59/spconv)


### BibTex
Please cite this paper if you find the code/model helpful in your research:
```
@inproceedings{chen2022gpnerf,
	title={Geometry-guided progressive NeRF for generalizable and efficient neural human rendering},
	author={Chen, Mingfei and Zhang, Jianfeng and Xu, Xiangyu and Liu, Lijuan and Cai, Yujun and Feng, Jiashi and Yan, Shuicheng},
	booktitle={ECCV},
	year={2022}
}
```
