import copy
import cv2
import imageio
import math
import numpy as np
import os
import os.path as osp
import random
import json

import torch

from libs.datasets.data_utils import (get_nearest_pose_ids, project,
                                      sample_ray, transform_can_smpl)
from libs.datasets.transform import TrainTransform, EvalTransform


class ZjumocapDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_root,
                 split,
                 seq_data=['CoreView_315',],
                 src_view_num=3,
                 cam_num=-1, #[-1, 2, 3, 4, 6, 8, 12]
                 transform=None,
                 fix_human=False,
                 fix_pose=False,
                 ratio=0.5,
                 body_sample_ratio=0.5,
                 nrays=1024,
                 mask_bkgd=True,
                 voxel_size=[0.005, 0.005, 0.005],
                 interval=7,
                 dataset_cams=24,
                 frame_sampler=False,
                 inside_view=[0, 6, 12, 18]
                 ):
        super(ZjumocapDataset, self).__init__()
        # for zju
        self.dataset_cams = dataset_cams
        self.data_root = data_root
        self.split = split # ['train', 'test]
        self.seq_data = seq_data
        self.cam_num = cam_num
        self.src_view_num = src_view_num
        if self.cam_num > 12:
            self.cam_num = -1 # get nearest cam
        self.transform = transform

        self.fix_human = fix_human
        self.fix_pose = fix_pose

        self.voxel_size = np.array(voxel_size)
        self.ratio = ratio
        self.mask_bkgd = mask_bkgd
        self.body_sample_ratio = body_sample_ratio
        self.nrays = nrays
        self.interval = interval
        self.frame_sampler = frame_sampler

        # for mesh
        self.inside_view = inside_view
        
        self.data_config()
        self.load_data()


    def get_mask(self, seq_path, img_name, border=5):
        msk_path = osp.join(seq_path, 'mask',
                            img_name)[:-4] + '.png'
        if os.path.exists(msk_path):
            msk = imageio.imread(msk_path)
            msk = (msk != 0).astype(np.uint8)

        msk_path = osp.join(seq_path, 'mask_cihp',
                            img_name)[:-4] + '.png'
        msk_cihp = imageio.imread(msk_path)
        msk_cihp = (msk_cihp != 0).astype(np.uint8)

        msk = (msk | msk_cihp).astype(np.uint8) if 'msk' in locals() else msk_cihp.astype(np.uint8)
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100

        return msk
    
    def data_config(self):
        self.config_dict = {}
        # CoreView_313
        CoreView_313 = {
            'begin_i': 1,
            'intv': 1,
            'ni': 300,
        }
        self.config_dict['CoreView_313'] = CoreView_313
        # CoreView_315
        CoreView_315 = {
            'begin_i': 1,
            'intv': 1,
            'ni': 300,
        }
        self.config_dict['CoreView_315'] = CoreView_315
        # CoreView_377
        CoreView_377 = {
            'begin_i': 0,
            'intv': 1,
            'ni': 300,
        }
        self.config_dict['CoreView_377'] = CoreView_377
        # CoreView_386
        CoreView_386 = {
            'begin_i': 0,
            'intv': 1,
            'ni': 300,
        }
        self.config_dict['CoreView_386'] = CoreView_386
        # CoreView_387
        CoreView_387 = {
            'begin_i': 0,
            'intv': 1,
            'ni': 300,
        }
        self.config_dict['CoreView_387'] = CoreView_387
        # CoreView_390
        CoreView_390 = {
            'begin_i': 700,
            'intv': 1,
            'ni': 300,
        }
        self.config_dict['CoreView_390'] = CoreView_390
        # CoreView_392
        CoreView_392 = {
            'begin_i': 0,
            'intv': 1,
            'ni': 300,
        }
        self.config_dict['CoreView_392'] = CoreView_392
        # CoreView_393
        CoreView_393 = {
            'begin_i': 0,
            'intv': 1,
            'ni': 300,
        }
        self.config_dict['CoreView_393'] = CoreView_393
        # CoreView_394
        CoreView_394 = {
            'begin_i': 0,
            'intv': 1,
            'ni': 300,
        }
        self.config_dict['CoreView_394'] = CoreView_394
        # CoreView_396
        CoreView_396 = {
            'begin_i': 810,
            'intv': 1,
            'ni': 300,
        }
        self.config_dict['CoreView_396'] = CoreView_396

    def load_data(self):
        assert osp.exists(self.data_root)

        self.all_ims = []
        self.all_cam_inds = []
        self.all_cams = []
        self.all_seqs = []
        self.all_cam_R = []
        for seq_name in self.seq_data:
            seq_path = osp.join(self.data_root, seq_name)
            ann_path = osp.join(seq_path, 'annots.npy')
            annots = np.load(ann_path, allow_pickle=True).item()
            cams = annots['cams']

            seq_config = self.config_dict[seq_name]
            begin_i = seq_config['begin_i']
            ni = seq_config['ni']
            intv = seq_config['intv']
            # (frames, views)
            ims = np.array([
                np.array(ims_data['ims'])
                for ims_data in annots['ims'][begin_i: begin_i+ni*intv][::intv]
            ])
            num_cams = ims.shape[1]
            # (frames, views)
            cam_inds = np.array([
                np.arange(len(ims_data['ims']))
                for ims_data in annots['ims'][begin_i: begin_i+ni*intv][::intv]
            ])
            assert len(ims) == len(cam_inds)
            num_frames = len(ims)
            for i in range(num_frames):
                self.all_seqs.append(seq_name)
                self.all_cams.append(cams)
                self.all_ims.append(ims[i])
                self.all_cam_inds.append(cam_inds[i]) 
            num_cams = ims.shape[1]
            if self.cam_num == 3:
                seq_config['test_ids'] = np.arange(num_cams)[::self.interval]
                seq_config['train_ids'] = np.array([0, 8, 16], dtype=np.int32)
            else:
                seq_config['test_ids'] = np.arange(num_cams)[::self.interval]
                seq_config['train_ids'] = np.array([j for j in np.arange(num_cams) if
                                    (j not in seq_config['test_ids'])])
            self.num_cams = seq_config['train_ids'].shape[0] if self.split == 'train' else seq_config['test_ids'].shape[0]

    def prepare_input(self, seq_path, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(seq_path, 'vertices',
                                     '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[2] -= 0.05
        max_xyz[2] += 0.05
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(seq_path, 'params',
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # transformation augmentation
        xyz, center, rot, trans = transform_can_smpl(xyz)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[2] -= 0.05
        max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        nxyz = nxyz.astype(np.float32)
        feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        coord = np.round((dhw - min_dhw) / self.voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / self.voxel_size).astype(np.int32)
        x = 32
        # mask sure the output size is N times of 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans


    def prepare_inside_pts(self, pts, i, cams, ims, seq_path):
        sh = pts.shape
        pts3d = pts.reshape(-1, 3)

        inside = np.ones([len(pts3d)]).astype(np.uint8)
        for nv in self.inside_view:
            ind = inside == 1
            pts3d_ = pts3d[ind]

            RT = np.concatenate([np.array(cams['R'][nv]), np.array(cams['T'][nv])/1000.], axis=1)
            pts2d = project(pts3d_, np.array(cams['K'][nv]), RT)

            img_name = ims[nv]
            msk = self.get_mask(seq_path, img_name)
            H, W = msk.shape
            pts2d = np.round(pts2d).astype(np.int32)
            pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
            pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)
            msk_ = msk[pts2d[:, 1], pts2d[:, 0]]

            inside[ind] = msk_

        inside = inside.reshape(*sh[:-1])

        return inside


    def __getitem__(self, index):
        if isinstance(index, list):
            index = index[0]
        cam_id = index % self.num_cams
        index = index // self.num_cams
        seq_name = self.all_seqs[index]
        cams = self.all_cams[index]
        ims = self.all_ims[index]
        cam_inds = self.all_cam_inds[index]
        seq_config = self.config_dict[seq_name]
        seq_path = osp.join(self.data_root, seq_name)
        num_imgs = len(ims)

        all_rgbs = []
        all_rays = []
        all_imgs = []
        all_poses = []
        all_Ks = []
        all_Ds = []
        all_mask_at_box = []

        ori_train_ids = list(range(len(cams['R'])))
        cur_train_ids = seq_config['train_ids']

        if self.split == 'train':
            render_ids = [i for i in ori_train_ids if i not in cur_train_ids]
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            if self.frame_sampler:
                id_render = cam_id
            else:
                id_render = np.array(random.sample(range(len(render_ids)), 1))[0]
            test_ind = render_ids[id_render]
        else:
            render_ids = seq_config['test_ids']
            subsample_factor = 1
            if self.frame_sampler:
                test_ind = render_ids[cam_id]
            else:
                test_ind = render_ids[np.array(random.sample(range(len(render_ids)), 1))[0]]
            id_render = -1

        # one for test and select nearest ones for training
        target_R = np.array(cams['R'][test_ind])
        target_T = np.array(cams['T'][test_ind]) / 1000.       
        tar_cam_loc = -np.dot(target_R.T, target_T).ravel()

        ref_R = np.array(cams['R'])[cur_train_ids]
        ref_T = np.array(cams['T'])[cur_train_ids] / 1000.
        ref_cam_locs = -np.matmul(ref_R.transpose(0,2,1), ref_T).squeeze(axis=-1)

        if self.cam_num != -1 and self.cam_num <= self.src_view_num:
            sample_num = self.cam_num
        else:
            sample_num = min(self.src_view_num*subsample_factor, 8)

        nearest_pose_ids = get_nearest_pose_ids(
            tar_cam_loc,
            ref_cam_locs,
            sample_num,
            tar_id=-1,
            angular_dist_method='dist')
        
        if self.cam_num == -1 or self.cam_num > self.src_view_num:
            nearest_pose_ids = np.random.choice(nearest_pose_ids, min(self.src_view_num, len(nearest_pose_ids)), replace=False)
            # occasionally include input image
            if np.random.choice([0, 1], p=[0.995, 0.005]) and self.split == 'train':
                nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render
        src_inds = cur_train_ids[nearest_pose_ids]

        # obtain data from test image
        img_name = ims[test_ind]
        cam_ind = cam_inds[test_ind]
        img_path = osp.join(seq_path, img_name)
        img = imageio.imread(img_path)[..., :3]
        tar_img = np.array(cv2.resize(img, (512, 512)))
        img = cv2.resize(img, (1024, 1024))
        msk = self.get_mask(seq_path, img_name)
        K = np.array(cams['K'][cam_ind])
        D = np.array(cams['D'][cam_ind])
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)

        R = np.array(cams['R'][cam_ind])
        T = np.array(cams['T'][cam_ind]) / 1000.

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * self.ratio), int(img.shape[1] * self.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.mask_bkgd:
            img[msk == 0] = 0
        K[:2] *= self.ratio
        
        pose = np.concatenate([R, T], axis=-1)
        target_pose = pose
        target_K = K
        target_D = D

        if seq_name in ['CoreView_313', 'CoreView_315']:
            i = int(osp.basename(img_name).split('_')[4])
            frame_index = i - 1
        else:
            i = int(osp.basename(img_name)[:-4])
            frame_index = i

        feature, coord, out_sh, can_bounds, bounds, Rh, Th, \
            center, rot, trans = self.prepare_input(seq_path, i)
        
        # unique for mesh 
        voxel_size = self.voxel_size
        x = np.arange(can_bounds[0, 0], can_bounds[1, 0] + voxel_size[0],
                      voxel_size[0])
        y = np.arange(can_bounds[0, 1], can_bounds[1, 1] + voxel_size[1],
                      voxel_size[1])
        z = np.arange(can_bounds[0, 2], can_bounds[1, 2] + voxel_size[2],
                      voxel_size[2])
        pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        pts = pts.astype(np.float32)

        inside = self.prepare_inside_pts(pts, i, cams, ims, seq_path)

        # get rays
        rgb, ray_o, ray_d, near, far, coord_, mask_at_box, out_body_msk = \
            sample_ray(img, msk, K, R, T[..., 0], can_bounds,
                       self.nrays, self.split, self.body_sample_ratio)
        
        rgb = rgb / 255.0
        ray = np.concatenate([ray_o, ray_d, \
            near[..., None], far[..., None]], axis=-1)

        all_rgbs.append(rgb)
        all_rays.append(ray)
        all_mask_at_box.append(mask_at_box)
        all_body_msk = [out_body_msk]

        # obtain data from source images
        for idx, (img_name, cam_ind) in enumerate(zip(ims[src_inds], cam_inds[src_inds])):
            img_path = osp.join(seq_path, img_name)
            img = imageio.imread(img_path)[..., :3]
            img = cv2.resize(img, (1024, 1024))
            msk = self.get_mask(seq_path, img_name)

            K = np.array(cams['K'][cam_ind])
            D = np.array(cams['D'][cam_ind])
            img = cv2.undistort(img, K, D)
            msk = cv2.undistort(msk, K, D)

            R = np.array(cams['R'][cam_ind])
            T = np.array(cams['T'][cam_ind]) / 1000.

            # reduce the image resolution by ratio
            H, W = int(img.shape[0] * self.ratio), int(img.shape[1] * self.ratio)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

            if self.mask_bkgd:
                img[msk == 0] = 0
            K[:2] *= self.ratio

            pose = np.concatenate([R, T], axis=-1)
            img = self.transform(img).numpy()
            all_imgs.append(img)
            all_poses.append(pose)
            all_Ks.append(K)
            all_Ds.append(D)

        all_rgbs = np.array(all_rgbs).reshape(-1, all_rgbs[-1].shape[-1])
        all_rays = np.array(all_rays).reshape(-1, 8)
        rgb = all_rgbs
        ray = all_rays
        ray_o, ray_d = ray[:, :3], ray[:, 3:6]
        near, far = ray[:, 6], ray[:, 7]

        all_mask_at_box = np.array(all_mask_at_box).reshape(-1)
        mask_at_box = all_mask_at_box
        body_msk = np.array(all_body_msk).reshape(-1)

        ret = {
            "tar_img": tar_img,
            "feature": feature,
            "coord": coord,
            "out_sh": out_sh,
            "rgb": rgb,
            "ray_o": ray_o,
            "ray_d": ray_d,
            "near": near,
            "far": far,
            "mask_at_box": mask_at_box,
            "inside": inside,
            "pts": pts,
            "body_msk": body_msk,
            "target_pose": target_pose,
            "target_K": target_K,
            "target_K_inv": np.linalg.inv(target_K),
            "target_D": target_D,
        }
        
        all_imgs = np.array(all_imgs)
        all_poses = np.array(all_poses)
        all_Ks = np.array(all_Ks)
        all_Ds = np.array(all_Ds)

        src_imgs = all_imgs
        src_poses = all_poses
        src_Ks = all_Ks
        src_Ds = all_Ds
        
        src_dict = {
            "src_imgs": src_imgs,
            "src_poses": src_poses,
            "src_Ks": src_Ks,
            "src_Ds": src_Ds
        }
        ret.update(src_dict)

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = frame_index - self.config_dict[seq_name]['begin_i']
        meta = {
            "can_bounds": can_bounds,
            "bounds": bounds,
            "R": R,
            "Rh": R,
            "Th": Th,
            'latent_index': latent_index,
            'frame_index': frame_index,
            "center": center,
            "rot": rot,
            "trans": trans,
            'cam_ind': test_ind
        }
        ret.update(meta)

        return ret        

    def __len__(self):
        return len(self.all_ims) * self.num_cams

def build_dataset(cfg, is_train=True):
    if is_train:   
        dataset = ZjumocapDataset(
            data_root=cfg.dataset.train.data_root,
            split='train',
            seq_data=cfg.dataset.train.seq_list,
            src_view_num=cfg.src_view_num,
            cam_num=cfg.cam_num, #[-1, 2, 3, 4, 6, 8, 12]
            transform=TrainTransform(),
            fix_human=cfg.fix_human,
            fix_pose=cfg.fix_pose,
            ratio=cfg.dataset.ratio,
            body_sample_ratio=cfg.train.body_sample_ratio,
            nrays=cfg.train.n_rays,
            mask_bkgd=cfg.mask_bkgd,
            voxel_size=cfg.dataset.voxel_size,
            interval=cfg.dataset.train.interval,
            dataset_cams=cfg.dataset.train.dataset_cams,
            frame_sampler=(cfg.dataset.test.sampler == 'FrameSampler')
        )
    else:
        dataset = ZjumocapDataset(
            data_root=cfg.dataset.test.data_root,
            split='test',
            seq_data=cfg.dataset.test.seq_list,
            src_view_num=cfg.src_view_num,
            cam_num=cfg.cam_num, #[-1, 2, 3, 4, 6, 8, 12]
            transform=EvalTransform(),
            fix_human=cfg.fix_human,
            fix_pose=cfg.fix_pose,
            ratio=cfg.dataset.ratio,
            body_sample_ratio=cfg.train.body_sample_ratio,
            nrays=cfg.train.n_rays,
            mask_bkgd=cfg.mask_bkgd,
            voxel_size=cfg.dataset.voxel_size,
            interval=cfg.dataset.test.interval,
            dataset_cams=cfg.dataset.train.dataset_cams,
            frame_sampler=(cfg.dataset.test.sampler == 'FrameSampler')
        )
    return dataset


if __name__ == "__main__":
    from libs.datasets.transform import TrainTransform
    dataset = ZjumocapDataset(data_root='data/zju_mocap/', split='train', cam_num=3, transform=TrainTransform())
    dataset.__getitem__(0)
    print(dataset.__len__())