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

import json
import os
from copy import deepcopy

import cv2
import numpy as np
import torch
from data_utils import (
    demo_rays,
    load_cameras,
    load_images,
    project,
    slerp,
    transform_can_smpl,
)

# torch.multiprocessing.set_start_method('spawn')
from torch.utils.data import Dataset

from libs.datasets.transform import EvalTransform
from libs.masksegs.people_mask import PeopleMask
from libs.smpls.OptSMPL import OptSMPL


class DemoDataset(Dataset):
    def __init__(
        self,
        camera_path,
        cam_inds=None,
        image_path=None,
        ratio=0.5,
        body_sample_ratio=0.5,
        voxel_size=[0.005, 0.005, 0.005],
        mask_bkgd=True,
        args=None,
    ) -> None:
        super(DemoDataset, self).__init__()
        self.default_size = [1024, 1024]
        self.default_camera_inds = ["32649687", "34931148", "36956751"]
        self.ratio = ratio
        self.target_size = [
            self.default_size[0] * self.ratio,
            self.default_size[1] * self.ratio,
        ]
        self.body_sample_ratio = body_sample_ratio
        self.voxel_size = voxel_size
        self.mask_bkgd = mask_bkgd
        self.args = args
        self.transform = EvalTransform()

        # self.num_cams = 1       # used for frame sampler (no special meaning)
        # import pdb; pdb.set_trace()

        self.camera_path = camera_path
        if cam_inds is None:
            self.cam_inds = self.default_camera_inds
        else:
            self.cam_inds = cam_inds

        self.cameras = load_cameras(
            [
                f"{self.camera_path}/intri_200_2.yml",
                f"{self.camera_path}/extri_200_2.yml",
            ],
            self.cam_inds,
        )
        self.target_cameras = []
        for i in range(1, len(cam_inds) + 1):
            pre, end = (i - 1) % len(cam_inds), i % len(cam_inds)
            R1, T1 = (
                self.cameras["R"][self.cam_inds[pre]],
                self.cameras["T"][self.cam_inds[pre]],
            )
            R2, T2 = (
                self.cameras["R"][self.cam_inds[end]],
                self.cameras["T"][self.cam_inds[end]],
            )
            self.target_cameras.extend(
                slerp(R1, R2, T1, T2, self.args.target_nums[i - 1])
            )

        self.total_tgts = len(self.target_cameras)

        self.streamline = args.streamline
        self.image_path = image_path
        if self.streamline:
            print(f"not supported yet")
            raise NotImplementedError("streamline is not supported yet")
        else:
            if self.image_path is None:
                print("Need input the image path")
                raise Exception("Need input the image path")

        self.imglist = load_images(self.image_path, self.cam_inds, args.ext)
        self.num_imgs = len(self.imglist[self.cam_inds[0]])
        self.smpl_mode = args.smpl_mode

        smpl_args = self.args.smpl
        smpl_args.defrost()
        smpl_args.height = self.args.forward.dataset.H
        smpl_args.width = self.args.forward.dataset.W
        smpl_args.image_path = self.args.image_path
        smpl_args.ext = self.args.ext
        smpl_args.freeze()

        # import pdb; pdb.set_trace()

        if self.smpl_mode == "opt":
            self.smpl_model = OptSMPL(
                self.imglist, self.cam_inds, deepcopy(self.cameras), self.args.smpl
            )

        self.mask_mode = args.mask_mode
        if self.mask_mode == "people":
            self.seg_model = PeopleMask(args.device)

    def prepare_input(self, smpl_params, vertices):
        xyz = np.array(vertices)
        nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[2] -= 0.05
        max_xyz[2] += 0.05
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate
        Rh = np.array(smpl_params["Rh"])
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = np.array(smpl_params["Th"]).astype(np.float32)
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

    def get_mask(self, mask, border=5):
        msk = (mask != 0).astype(np.uint8)
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100
        return msk

    def prepare_inside_pts(self, pts, masks):
        sh = pts.shape
        pts3d = pts.reshape(-1, 3)
        inside = np.ones([len(pts3d)]).astype(np.uint8)

        for i, msk in enumerate(masks):
            ind = inside == 1
            pts3d_ = pts3d[ind]
            RT = np.concatenate(
                [
                    np.array(self.cameras["R"][self.cam_inds[i]]),
                    np.array(self.cameras["T"][self.cam_inds[i]] / 1000.0),
                ],
                axis=1,
            )
            pts2d = project(pts3d_, np.array(self.cameras["K"][self.cam_inds[i]]), RT)
            H, W = msk.shape
            pts2d = np.round(pts2d).astype(np.int32)
            pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
            pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)
            msk_ = msk[pts2d[:, 1], pts2d[:, 0]]

            inside[ind] = msk_
        inside = inside.reshape(*sh[:-1])
        return inside

    def __len__(
        self,
    ):
        return self.num_imgs * self.total_tgts

    def __getitem__(self, index):

        # import pdb; pdb.set_trace()

        pose_index = index // self.total_tgts
        tgt_cam_index = index % self.total_tgts

        smpl_param, vertices, c_imgs = self.smpl_model[pose_index]
        c_masks = [self.seg_model.process(x) for x in c_imgs]

        # import pdb; pdb.set_trace()

        (
            feature,
            coord,
            out_sh,
            can_bounds,
            bounds,
            Rh,
            Th,
            center,
            rot,
            trans,
        ) = self.prepare_input(smpl_param, vertices)

        ############## may delete later
        voxel_size = self.voxel_size

        x = np.arange(can_bounds[0, 0], can_bounds[1, 0] + voxel_size[0], voxel_size[0])
        y = np.arange(can_bounds[0, 1], can_bounds[1, 1] + voxel_size[1], voxel_size[1])
        z = np.arange(can_bounds[0, 2], can_bounds[1, 2] + voxel_size[2], voxel_size[2])
        pts = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
        pts = pts.astype(np.float32)

        inside = self.prepare_inside_pts(pts, c_masks)
        ###############

        target_pose = np.array(self.target_cameras[tgt_cam_index])
        R = target_pose[:, :3]
        T = target_pose[:, 3:] / 1000
        target_pose = np.concatenate([R, T], axis=-1)

        # fake target K: no ground truth target K, use one of the source views instead
        target_K = deepcopy(self.cameras["K"][self.cam_inds[0]])
        target_K[:2] *= self.ratio
        target_D = deepcopy(self.cameras["D"][self.cam_inds[0]])

        # get rays
        ray_o, ray_d, near, far, coord_, mask_at_box = demo_rays(
            self.target_size[0], self.target_size[1], target_K, R, T[..., 0], can_bounds
        )
        # ray = np.concatenate([ray_o, ray_d, near[..., None], far[..., None]], axis=-1)
        mask_at_box = np.array(mask_at_box).reshape(-1)

        ret = {
            "feature": feature,
            "coord": coord,
            "out_sh": out_sh,
            "ray_o": ray_o,
            "ray_d": ray_d,
            "near": near,
            "far": far,
            "mask_at_box": mask_at_box,
            "inside": inside,
            "pts": pts,
            "target_K": target_K,
            "target_K_inv": np.linalg.inv(target_K),
            "target_D": target_D,
            "target_pose": target_pose,
        }

        all_imgs = []
        all_poses = []
        all_Ks = []
        all_Ds = []

        for idx, (img, msk) in enumerate(zip(c_imgs, c_masks)):
            img = cv2.resize(img, tuple(self.default_size))
            msk = self.get_mask(msk)

            K = np.array(self.cameras["K"][self.cam_inds[idx]])
            D = np.array(self.cameras["D"][self.cam_inds[idx]])
            img = cv2.undistort(img, K, D)
            msk = cv2.undistort(msk, K, D)

            R = np.array(self.cameras["R"][self.cam_inds[idx]])
            T = np.array(self.cameras["T"][self.cam_inds[idx]]) / 1000.0
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

        src_imgs = np.array(all_imgs)
        src_poses = np.array(all_poses)
        src_Ks = np.array(all_Ks)
        src_Ds = np.array(all_Ds)

        src_dict = {
            "src_imgs": src_imgs,
            "src_poses": src_poses,
            "src_Ks": src_Ks,
            "src_Ds": src_Ds,
        }
        ret.update(src_dict)
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)

        meta = {
            "can_bounds": can_bounds,
            "bounds": bounds,
            "R": R,
            "Rh": R,
            "Th": Th,
            # 'latent_index': latent_index,
            "frame_index": pose_index,
            "center": center,
            "rot": rot,
            "trans": trans,
            "cam_ind": tgt_cam_index,
        }

        ret.update(meta)
        return ret


def build_dataset(args):
    dataset = DemoDataset(
        camera_path=args.camera_path,
        cam_inds=args.camera_ids,
        image_path=args.image_path,
        ratio=args.forward.dataset.ratio,
        body_sample_ratio=args.forward.dataset.body_sample_ratio,
        voxel_size=args.forward.dataset.voxel_size,
        mask_bkgd=args.forward.mask_bkgd,
        args=args,
    )
    return dataset
