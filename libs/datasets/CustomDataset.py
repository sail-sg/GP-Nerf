# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import math
import os
import os.path as osp
import random

import cv2
import imageio
import numpy as np
import torch

from libs.datasets.data_utils import project, sample_ray, transform_can_smpl
from libs.datasets.transform import EvalTransform, TrainTransform


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        split,
        src_view_num=3,
        cam_num=-1,  # [-1, 2, 3, 4, 6, 8, 12]
        transform=None,
        fix_human=False,
        fix_pose=False,
        ratio=0.5,
        body_sample_ratio=0.5,
        nrays=1024,
        voxel_size=[0.005, 0.005, 0.005],
        mask_bkgd=True,
        inside_view=[0, 6, 12, 18],
    ):
        super(CustomDataset, self).__init__()
        self.data_root = data_root
        self.split = split  # ['train', 'test]
        self.cam_num = cam_num
        self.src_view_num = src_view_num
        if self.cam_num > 12:
            self.cam_num = -1  # get nearest cam
        self.transform = transform

        self.fix_human = fix_human
        self.fix_pose = fix_pose

        self.voxel_size = np.array(voxel_size)
        self.ratio = ratio
        self.mask_bkgd = mask_bkgd
        self.body_sample_ratio = body_sample_ratio
        self.nrays = nrays

        # for mesh
        self.inside_view = inside_view

        self.load_data()

    def get_mask(self, msk_path, border=5):
        if os.path.exists(msk_path):
            msk = imageio.imread(msk_path)
            msk = (msk != 0).astype(np.uint8)
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100

        return msk

    def load_data(self):
        assert osp.exists(self.data_root)
        if self.split in ["train"]:
            anno_path = osp.join(self.data_root, "train_anno.json")
        else:
            anno_path = osp.join(self.data_root, "test_anno.json")
        with open(anno_path) as f:
            annot_list = json.load(f)
        self.anno_list = []

        annot_list = annot_list[:7]

        if self.fix_human is True:
            human_ind = random.sample(range(len(annot_list)), 1)[0]
            annot_list = [annot_list[human_ind]]
        for annot in annot_list:
            seq_name = annot["human_dir"]
            pose_list = annot["multiposes"]
            count = 0
            if self.fix_pose is True:
                pose_ind = random.sample(range(len(pose_list)), 1)[0]
                pose_list = [pose_list[pose_ind]]
                count += 1
            for pose in pose_list:
                self.anno_list.append(pose)

    def prepare_inside_pts(self, pts, annot):
        sh = pts.shape
        pts3d = pts.reshape(-1, 3)

        inside = np.ones([len(pts3d)]).astype(np.uint8)
        for nv in self.inside_view:
            ind = inside == 1
            pts3d_ = pts3d[ind]
            cam_path = self.data_root + annot[nv]["camera_params_path"]
            cams = np.load(cam_path, allow_pickle=True).item()
            RT = np.concatenate(
                [np.array(cams["R"]), np.array(cams["T"]).reshape(-1, 1)], axis=1
            )
            pts2d = project(pts3d_, np.array(cams["K"]), RT)
            msk = self.get_mask(self.data_root + annot[nv]["masks_path"])[..., 0]
            H, W = msk.shape
            pts2d = np.round(pts2d).astype(np.int32)
            pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
            pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)
            msk_ = msk[pts2d[:, 1], pts2d[:, 0]]
            inside[ind] = msk_

        inside = inside.reshape(*sh[:-1])

        return inside

    def evaluate(self):
        pass

    def __getitem__(self, index):
        if isinstance(index, list):
            index = index[0]
        annot = self.anno_list[index]
        frame_i = index
        img_w = annot["img_w"]
        img_h = annot["img_h"]
        total_cam_num = int(360 / annot["vsight_gap"])
        annot = annot["multiviews"]

        if self.split == "train":
            if self.cam_num != -1:
                subsample_factor = np.random.choice(np.arange(1, 3), p=[0.75, 0.25])
                train_base_id = random.sample(range(total_cam_num), 1)[0]
                valid_train_ids = [
                    (train_base_id + i * int(total_cam_num / self.cam_num))
                    % total_cam_num
                    for i in range(self.cam_num)
                ]
                test_ids = [i for i in range(total_cam_num) if i not in valid_train_ids]
                if self.fix_human:
                    select_test_ids = [
                        i
                        for i in range(total_cam_num)
                        if i not in valid_train_ids and i not in [5, 10, 17, 23]
                    ]
                else:
                    select_test_ids = test_ids
                target_id = random.sample(select_test_ids, 1)[0]
            else:
                subsample_factor = np.random.choice(
                    np.arange(1, 4), p=[0.2, 0.45, 0.35]
                )
                if self.fix_human:
                    base_test_ids = [5, 10, 17, 23]
                else:
                    base_test_ids = []
                select_test_ids = [
                    i for i in range(total_cam_num) if i not in base_test_ids
                ]
                target_id = random.sample(select_test_ids, 1)[0]
                base_test_ids.append(target_id)
                test_ids = base_test_ids
        else:
            # default src input id start from cam id 0, test id [5, 10, 17, 23] for test
            subsample_factor = 1
            test_ids = [5, 10, 17, 23]
            target_id = random.sample(test_ids, 1)[0]
            if self.cam_num != -1:
                train_base_id = 0
                valid_train_ids = [
                    (train_base_id + i * int(total_cam_num / self.cam_num))
                    % total_cam_num
                    for i in range(self.cam_num)
                ]
                test_ids = [
                    i for i in range(total_cam_num) if i not in valid_train_ids
                ] + test_ids
                test_ids = list(set(test_ids))
        dists0 = np.array(
            [
                max(cam_id, target_id) - min(cam_id, target_id)
                for cam_id in range(total_cam_num)
            ]
        ).reshape(-1, 1)
        dists1 = (
            np.array(
                [
                    min(cam_id, target_id) - max(cam_id, target_id)
                    for cam_id in range(total_cam_num)
                ]
            ).reshape(-1, 1)
            + 24
        )
        dists = np.hstack([dists0, dists1]).min(axis=-1)
        num_select = min(
            self.src_view_num * subsample_factor, 8, total_cam_num - len(test_ids)
        )
        dists[test_ids] = 1e3
        sorted_ids = np.argsort(dists)
        nearest_view_ids = sorted_ids[:num_select]

        if self.cam_num == -1 or self.cam_num > self.src_view_num:
            nearest_view_ids = np.random.choice(
                nearest_view_ids,
                min(self.src_view_num, len(nearest_view_ids)),
                replace=False,
            )
            assert target_id not in nearest_view_ids
            # occasionally include input image
            if np.random.choice([0, 1], p=[0.995, 0.005]) and self.split == "train":
                nearest_view_ids[np.random.choice(len(nearest_view_ids))] = target_id
        src_ids = nearest_view_ids

        # obtain anno and img for test img
        target_cam_path = osp.join(
            self.data_root, annot[target_id]["camera_params_path"]
        )
        target_img_path = osp.join(self.data_root, annot[target_id]["rgb_imgs_path"])
        target_mask_path = osp.join(self.data_root, annot[target_id]["masks_path"])
        target_smpl_path = osp.join(
            self.data_root, annot[target_id]["smpl_vertices_path"]
        )

        rgb_img = imageio.imread(target_img_path)[..., :3]
        msk = self.get_mask(target_mask_path)
        target_cam = np.load(target_cam_path, allow_pickle=True).item()
        K, T, R = target_cam["K"], target_cam["T"], target_cam["R"]

        xyz = np.load(target_smpl_path).astype(np.float32)
        #### get bounds under world coord from smpl under camera coord
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = T
        # camera smpl coord -> world coord
        pose_inv = np.linalg.inv(pose)
        xyz_homo = np.hstack([xyz, xyz[..., 0][..., np.newaxis] * 0.0 + 1.0])
        xyz = (xyz_homo @ pose_inv.T)[..., :3]
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[2] -= 0.05
        max_xyz[2] += 0.05
        # world coord
        target_world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # smpl share the same coord with world coord
        target_bounds = target_world_bounds
        cxyz = xyz.astype(np.float32)
        nxyz = np.zeros_like(xyz).astype(np.float32)
        target_smpl_feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)
        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        target_smpl_coord = np.round((dhw - min_dhw) / self.voxel_size).astype(np.int32)
        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / self.voxel_size).astype(np.int32)
        # mask sure the output size is N times of 32
        target_smpl_out_sh = (out_sh | (32 - 1)) + 1

        # resize
        H, W = int(rgb_img.shape[0] * self.ratio), int(rgb_img.shape[1] * self.ratio)
        img = cv2.resize(rgb_img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        if self.mask_bkgd:
            img[msk == 0] = 0
        # target camera params
        K[:2] *= self.ratio
        RT = np.concatenate([R, T.reshape(-1, 1)], axis=1)
        target_pose = RT
        target_K = K
        target_meta = {
            "bounds": target_bounds,
            "R": R,
            "T": T,
            "Rh": np.eye(3),
            "Th": np.zeros(3),
        }

        # unique for mesh
        voxel_size = self.voxel_size
        x = np.arange(
            target_world_bounds[0, 0],
            target_world_bounds[1, 0] + voxel_size[0],
            voxel_size[0],
        )
        y = np.arange(
            target_world_bounds[0, 1],
            target_world_bounds[1, 1] + voxel_size[1],
            voxel_size[1],
        )
        z = np.arange(
            target_world_bounds[0, 2],
            target_world_bounds[1, 2] + voxel_size[2],
            voxel_size[2],
        )
        pts = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
        pts = pts.astype(np.float32)

        inside = self.prepare_inside_pts(pts, annot)

        # get rays
        rgb, ray_o, ray_d, near, far, coord_, mask_at_box, out_body_msk = sample_ray(
            img,
            msk,
            K,
            R,
            T,
            target_world_bounds,
            self.nrays,
            self.split,
            self.body_sample_ratio,
        )
        rgb = rgb / 255.0
        ray = np.concatenate([ray_o, ray_d, near[..., None], far[..., None]], axis=-1)
        all_rgbs = [rgb]
        all_rays = [ray]
        all_mask_at_box = [mask_at_box]
        all_body_msk = [out_body_msk]

        # obtain data from source images
        all_src_imgs = []
        all_src_poses = []
        all_src_Ks = []
        for idx, cam_ind in enumerate(src_ids):
            src_cam_path = self.data_root + annot[cam_ind]["camera_params_path"]
            src_img_path = self.data_root + annot[cam_ind]["rgb_imgs_path"]
            src_mask_path = self.data_root + annot[cam_ind]["masks_path"]
            img = imageio.imread(src_img_path)[..., :3]
            msk = self.get_mask(src_mask_path)
            src_cam = np.load(src_cam_path, allow_pickle=True).item()
            K = np.array(src_cam["K"])
            # no undistort operation
            R = np.array(src_cam["R"])
            T = np.array(src_cam["T"])

            # reduce the image resolution by ratio
            H, W = int(img.shape[0] * self.ratio), int(img.shape[1] * self.ratio)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            if self.mask_bkgd:
                img[msk == 0] = 0
            K[:2] *= self.ratio
            RT = np.concatenate([R, T.reshape(-1, 1)], axis=1)
            img = self.transform(img).numpy()
            all_src_imgs.append(img)
            all_src_poses.append(RT)
            all_src_Ks.append(K)

        # generate anno
        all_rgbs = np.array(all_rgbs).reshape(-1, 3)
        all_rays = np.array(all_rays).reshape(-1, 8)
        rgb = all_rgbs
        ray = all_rays
        ray_o, ray_d = ray[:, :3], ray[:, 3:6]
        near, far = ray[:, 6], ray[:, 7]
        mask_at_box = np.array(all_mask_at_box).reshape(-1)
        body_msk = np.array(all_body_msk).reshape(-1)

        ret = {
            "feature": target_smpl_feature,
            "coord": target_smpl_coord,
            "out_sh": target_smpl_out_sh,
            "rgb": rgb,
            "ray_o": ray_o,
            "ray_d": ray_d,
            "near": near,
            "far": far,
            "inside": inside,
            "pts": pts,
            "mask_at_box": mask_at_box,
            "body_msk": body_msk,
            "frame_index": frame_i,
            "latent_index": frame_i,
            "cam_ind": target_id,
            "target_pose": target_pose,
            "target_K": target_K,
        }
        src_dict = {
            "src_imgs": np.array(all_src_imgs),
            "src_poses": np.array(all_src_poses),
            "src_Ks": np.array(all_src_Ks),
        }
        ret.update(src_dict)
        ret.update(target_meta)

        return ret

    def __len__(self):
        return len(self.anno_list)


def build_dataset(cfg, is_train=True):
    if is_train:
        dataset = CustomDataset(
            data_root=cfg.dataset.train.data_root,
            split="train",
            src_view_num=cfg.src_view_num,
            cam_num=cfg.cam_num,  # [-1, 2, 3, 4, 6, 8, 12]
            transform=TrainTransform(),
            fix_human=cfg.fix_human,
            fix_pose=cfg.fix_pose,
            ratio=cfg.dataset.ratio,
            body_sample_ratio=cfg.train.body_sample_ratio,
            nrays=cfg.train.n_rays,
            voxel_size=cfg.dataset.voxel_size,
            mask_bkgd=cfg.mask_bkgd,
        )
    else:
        dataset = CustomDataset(
            data_root=cfg.dataset.test.data_root,
            split="test",
            src_view_num=cfg.src_view_num,
            cam_num=cfg.cam_num,  # [-1, 2, 3, 4, 6, 8, 12]
            transform=EvalTransform(),
            fix_human=cfg.fix_human,
            fix_pose=cfg.fix_pose,
            ratio=cfg.dataset.ratio,
            body_sample_ratio=cfg.train.body_sample_ratio,
            nrays=cfg.train.n_rays,
            voxel_size=cfg.dataset.voxel_size,
            mask_bkgd=cfg.mask_bkgd,
        )
    return dataset


if __name__ == "__main__":
    dataset = CustomDataset(
        data_root="data/thuman/", split="train", cam_num=3, transform=TrainTransform()
    )
    dataset.__getitem__(0)
    print(dataset.__len__())
