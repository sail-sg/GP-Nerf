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

import math
import time
from importlib import import_module as impm

# from libs.utils.fillhole import fillHole÷
import _pickle as cPickle
import cv2
import mcubes
import numpy as np
import spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh


class Renderer(nn.Module):
    def __init__(
        self,
        encoder,
        nerfhead,
        is_train=True,
        neg_ray_train=False,
        neg_ray_val=False,
        n_rays=1024,
        n_samples=64,
        voxel_size=[0.005, 0.005, 0.005],
        chunk=64,
        mesh_th=-1,
        device="cuda",
    ):
        super().__init__()
        self.encoder = encoder
        self.nerfhead = nerfhead
        self.is_train = is_train
        self.neg_ray_train = neg_ray_train
        self.neg_ray_val = neg_ray_val
        self.n_rays = n_rays
        self.n_samples = n_samples
        self.voxel_size = np.array(voxel_size)
        self.chunk = chunk
        self.mesh_th = mesh_th
        self.code = None

    def get_sampling_points(self, ray_o, ray_d, near, far, perturb=1):
        # calculate the steps for each ray
        t_vals = torch.linspace(0.0, 1.0, steps=self.n_samples).to(near)
        z_vals = near[..., None] * (1.0 - t_vals) + far[..., None] * t_vals

        if perturb > 0.0 and self.is_train:
            # get intervals between samples
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand
        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def pts_to_can_pts(self, pts, batch):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = batch["Th"][:, None]
        pts = pts - Th
        R = batch["Rh"]
        sh = pts.shape
        pts = torch.matmul(pts.view(sh[0], -1, sh[3]), R)
        pts = pts.view(*sh)
        return pts

    def get_grid_coords(self, pts, sp_input, batch):
        # convert xyz to the voxel coordinate dhw
        min_xyz = batch["bounds"][:, 0, :]
        xyz = pts - min_xyz[:, None, None]
        xyz = xyz / 0.005
        # convert the voxel coordinate to [-1, 1]
        out_sh = sp_input["grid_out_sh"]
        grid_coords = xyz / out_sh * 2 - 1
        return grid_coords

    def render_rays(self, neg_ray, featmaps, src_imgs, src_cameras, batch):
        time_slots = {}
        torch.cuda.synchronize()
        start_time = time.time()

        device = featmaps.device
        sp_input = self.prepare_sp_input(batch)
        xyz = batch["feature"][..., :3]
        R = batch["Rh"].float()
        Th = batch["Th"].float()
        smpl_xyz = torch.bmm(xyz, R.transpose(1, 2)) + Th
        projector = Projector(device, neg_ray=neg_ray)
        coord = sp_input["coord"]
        out_sh = sp_input["out_sh"]
        batch_size = sp_input["batch_size"]
        bounds = sp_input["bounds"].float()
        voxel_size = sp_input["voxel_size"].float()
        target_pose = sp_input["target_pose"].float()
        target_K = sp_input["target_K"].float()

        torch.cuda.synchronize()
        bc_time = time.time() - start_time
        time_slots["bc_time"] = bc_time

        torch.cuda.synchronize()
        start_time = time.time()

        code = self.nerfhead.sigmahead.c(torch.arange(0, 6890).to(device))

        torch.cuda.synchronize()
        bc_time = time.time() - start_time
        time_slots["sigma_c"] = bc_time

        torch.cuda.synchronize()
        start_time = time.time()

        code_query = code.unsqueeze(1)
        smpl_feat_sampled = projector.compute_smpl(
            smpl_xyz, src_cameras, featmaps=featmaps
        ).flatten(0, 1)

        torch.cuda.synchronize()
        bc_time = time.time() - start_time
        time_slots["bc_attn"] = bc_time
        torch.cuda.synchronize()
        start_time = time.time()

        xyzc_fuse = self.nerfhead.sigmahead.xyzc_attn(
            code_query, smpl_feat_sampled, smpl_feat_sampled
        )[0].squeeze(1)

        torch.cuda.synchronize()
        bc_time = time.time() - start_time
        time_slots["sigma_attn"] = bc_time
        torch.cuda.synchronize()
        start_time = time.time()

        code = xyzc_fuse
        xyzc = spconv.SparseConvTensor(code, coord, out_sh, batch_size)
        self.nerfhead.sigmahead.xyzc_net.encode(
            xyzc, threshold=0.1
        )  # [batchsize, channels, point_nums]

        torch.cuda.synchronize()
        bc_time = time.time() - start_time
        time_slots["sp_encode"] = bc_time
        torch.cuda.synchronize()
        start_time = time.time()

        sp_input["xyzc"] = xyzc
        mask_xyz = self.nerfhead.sigmahead.xyzc_net.mask_xyz
        pts = mask_xyz * voxel_size + bounds[0, 0]
        pts = pts @ R[0].T + Th[0, 0]

        # can_bounds = batch['can_bounds']
        min_xyz = torch.min(pts, dim=0)[0]
        max_xyz = torch.max(pts, dim=0)[0]
        min_xyz[2] -= 0.05
        max_xyz[2] += 0.05
        can_bounds = torch.stack([min_xyz, max_xyz], dim=0)

        if self.nerfhead.use_rgbhead:
            # smpl_xyz
            pts_mask = pts.float() @ target_pose[0, :, :3].T + target_pose[0, :, 3:].T
            pts_mask = pts_mask @ target_K.T[..., 0]
            pts_xy = pts_mask[:, :2] / pts_mask[:, 2:]
            minx, miny = pts_xy[..., 0].long(), pts_xy[..., 1].long()
            maxx, maxy = minx + 1, miny + 1
            W = 512
            minx = minx.clamp(0, W - 1)
            miny = miny.clamp(0, W - 1)
            maxx = maxx.clamp(0, W - 1)
            maxy = maxy.clamp(0, W - 1)
            new_ray_idx1 = (miny * W) + minx
            new_ray_idx2 = (maxy * W) + minx
            new_ray_idx3 = (miny * W) + maxx
            new_ray_idx4 = (maxy * W) + maxx
            new_ray_idx = (
                torch.cat(
                    [new_ray_idx1, new_ray_idx2, new_ray_idx3, new_ray_idx4], dim=0
                )
            ).long()
            new_mask_at_box = torch.zeros((W * W)).to("cuda")
            new_mask_at_box[new_ray_idx] = 1.0
            j, i = torch.where(new_mask_at_box.view(W, -1) == 1)
            xy1 = torch.stack([i, j, torch.ones_like(i)], dim=-1)
            ori_rays_o = -target_pose[0, :, :3].T @ target_pose[0, :, 3:]

            pixel_camera = xy1.float() @ batch["target_K_inv"][0].T.float()
            pixel_world = (pixel_camera - target_pose[0, :, 3:].T) @ target_pose[
                0, :, :3
            ]
            # calculate the ray direction
            rays_o = ori_rays_o.view(-1)
            rays_d = pixel_world - rays_o[None]
            rays_o = rays_o.expand(rays_d.shape)
            nominator = can_bounds[None] - rays_o[:, None]
            d_intersect = (nominator / rays_d[:, None]).reshape(-1, 6)
            # calculate the six interections
            p_intersect = d_intersect[..., None] * rays_d[:, None] + rays_o[:, None]
            min_x, min_y, min_z, max_x, max_y, max_z = can_bounds.view(-1)
            eps = 1e-6
            p_mask_at_box = (
                (p_intersect[..., 0] >= (min_x - eps))
                * (p_intersect[..., 0] <= (max_x + eps))
                * (p_intersect[..., 1] >= (min_y - eps))
                * (p_intersect[..., 1] <= (max_y + eps))
                * (p_intersect[..., 2] >= (min_z - eps))
                * (p_intersect[..., 2] <= (max_z + eps))
            )
            mask_at_box = p_mask_at_box.sum(-1) == 2
            p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
                -1, 2, 3
            )
            # calculate the step of intersections
            rays_o = rays_o[mask_at_box]
            rays_d = rays_d[mask_at_box]
            norm_ray = torch.norm(rays_d, dim=1)
            d0 = torch.norm(p_intervals[:, 0, :] - rays_o, dim=1) / norm_ray
            d1 = torch.norm(p_intervals[:, 1, :] - rays_o, dim=1) / norm_ray
            if neg_ray:
                d1 = -d1
            near = torch.min(d0, d1)
            far = torch.max(d0, d1)
            new_pts, _ = self.get_sampling_points(
                rays_o.unsqueeze(0),
                rays_d.unsqueeze(0),
                near.unsqueeze(0),
                far.unsqueeze(0),
            )
            hold_len, sample_num = new_pts.shape[1:3]
            pts = new_pts.flatten(1, 2).unsqueeze(2).float()
            sh = pts.shape
        else:
            x = torch.range(
                can_bounds[0, 0],
                can_bounds[1, 0] + self.voxel_size[0],
                self.voxel_size[0],
            )
            y = torch.range(
                can_bounds[0, 1],
                can_bounds[1, 1] + self.voxel_size[1],
                self.voxel_size[1],
            )
            z = torch.range(
                can_bounds[0, 2],
                can_bounds[1, 2] + self.voxel_size[2],
                self.voxel_size[2],
            )
            pts = torch.stack(torch.meshgrid(x, y, z), -1).float().to(can_bounds.device)
            hold_sh = pts.shape
            pts = pts.view(1, -1, 1, 3)
            sh = pts.shape

        feat = self.nerfhead.sigmahead.xyzc_net.masks3d
        pts_smpl = self.pts_to_can_pts(pts, batch)
        grid_coords = self.get_grid_coords(pts_smpl, sp_input, batch)
        grid_coords = grid_coords.view(sh[0], -1, 3)
        sp_feats = F.grid_sample(
            feat.unsqueeze(0).unsqueeze(0),
            grid_coords[:, None, None].float(),
            padding_mode="zeros",
            align_corners=True,
        )

        valid = torch.where(sp_feats.view(1, -1)[0] > 0)[0]
        pts = pts[:, valid]
        grid_coords = grid_coords[:, valid]
        rgb_feat, mask = projector.compute(
            pts.squeeze(0), src_imgs, src_cameras, featmaps=featmaps
        )

        torch.cuda.synchronize()
        bc_time = time.time() - start_time
        time_slots["bf_sigma"] = bc_time
        torch.cuda.synchronize()
        start_time = time.time()

        sigma_feat, globalfeat = self.nerfhead.sigmahead.test_forward(
            sp_input, grid_coords, rgb_feat, mask
        )  # update
        sigma_x = globalfeat.squeeze(2)
        sigma = self.nerfhead.rgbhead.out_geometry_fc(sigma_x)
        num_valid_obs = torch.sum(mask, dim=2)
        sigma = sigma.masked_fill(
            num_valid_obs < 1, 0.0
        )  # set the sigma of invalid point to zero

        torch.cuda.synchronize()
        bc_time = time.time() - start_time
        time_slots["sigma_f"] = bc_time
        torch.cuda.synchronize()
        start_time = time.time()

        sigma = sigma[..., 0]
        sigma2alpha = lambda sigma: 1.0 - torch.exp(-sigma)
        alpha = sigma2alpha(sigma)

        if self.nerfhead.use_rgbhead:
            valid1 = torch.where(alpha[..., 0] > 1e-14)[0]
            rgb_feat = rgb_feat[valid1]

            torch.cuda.synchronize()
            bc_time = time.time() - start_time
            time_slots["bf_rgb"] = bc_time
            torch.cuda.synchronize()
            start_time = time.time()
            rgb_in, rgb_out, sigma_out = self.nerfhead.rgbhead(rgb_feat, sigma_feat[valid1], mask[valid1])
            raw = torch.cat([rgb_out, sigma_out], dim=-1)
            torch.cuda.synchronize()
            bc_time = time.time() - start_time
            time_slots["rgb_f"] = bc_time
            torch.cuda.synchronize()
            start_time = time.time()

            rgb = raw[:, :, :3]  # [N_rays, N_samples, 3]

            hold_rgb = torch.zeros((hold_len * sample_num, 3)).to(device)
            hold_alpha = torch.zeros((hold_len * sample_num)).to(device)
            hold_rgb[valid[valid1]] = rgb[:, 0, :]
            hold_alpha[valid] = alpha[..., 0]
            hold_rgb = hold_rgb.view(hold_len, sample_num, 3)
            hold_alpha = hold_alpha.view(hold_len, sample_num)
            T = torch.cumprod(1.0 - hold_alpha + 1e-10, axis=-1)[..., :-1]
            T = torch.cat((torch.ones_like(T[..., 0:1]).to(T.device), T), axis=-1)
            weights = hold_alpha * T
            rgb_map = (
                torch.sum(weights.unsqueeze(-1) * hold_rgb, axis=1).data.cpu().numpy()
            )
            new_mask_at_box[torch.where(new_mask_at_box == 1)[0][mask_at_box == 0]] = 0
            mask_at_box = (
                new_mask_at_box.reshape((int(W), int(W))).data.cpu().numpy() == 1
            )
            pred_img = np.zeros((int(W), int(W), 3))
            pred_img[mask_at_box] = rgb_map

            torch.cuda.synchronize()
            bc_time = time.time() - start_time
            time_slots["bc_render"] = bc_time

            ret = {
                "rgb_map": rgb_map,
                "pred_img": pred_img,
                "mask_at_box": mask_at_box.reshape(-1),
                "time_slots": time_slots,
            }
            return ret
        else:
            hold_alpha = torch.zeros((hold_sh[0] * hold_sh[1] * hold_sh[2])).to(device)
            hold_alpha[valid] = alpha[..., 0]
            hold_alpha = hold_alpha.view(*hold_sh[:-1])
            cube = hold_alpha.detach().cpu().numpy()
            cube = np.pad(cube, 10, mode="constant")
            vertices, triangles = mcubes.marching_cubes(cube, 1 / 50.0)
            mesh = trimesh.Trimesh(vertices, triangles)
            # mesh.export("mesh.ply")

            return {"mesh": mesh}

    def batchify_rays(self, rays, featmaps, src_imgs, src_cameras, chunk, batch):
        """Render rays in smaller minibatches to avoid OOM."""
        all_ret = {}
        if "body_msk" in batch.keys() and batch["body_msk"].shape[-1] > self.n_rays:
            neg_ray = self.neg_ray_val
        else:
            neg_ray = self.neg_ray_train

        ret = self.render_rays(neg_ray, featmaps, src_imgs, src_cameras, batch)

        all_ret = ret

        return all_ret


    def prepare_sp_input(self, batch):
        # feature, coordinate, shape, batch size
        sp_input = {}

        # feature: [N, f_channels]
        sh = batch["feature"].shape  # smpl coordiante
        sp_input["feature"] = batch["feature"].view(-1, sh[-1])

        # coordinate: [N, 4], batch_idx, z, y, x
        sh = batch["coord"].shape  # smpl coordinate
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]
        idx = torch.cat(idx).to(batch["coord"])
        coord = batch["coord"].view(-1, sh[-1])
        sp_input["coord"] = torch.cat([idx[:, None], coord], dim=1)

        out_sh, _ = torch.max(batch["out_sh"], dim=0)
        sp_input["out_sh"] = out_sh.tolist()
        out_sh = out_sh.flip(-1).float().to("cuda")
        sp_input["grid_out_sh"] = out_sh
        sp_input["bounds"] = batch["bounds"]
        sp_input["voxel_size"] = torch.tensor(self.voxel_size).to(out_sh.device)

        sp_input["batch_size"] = sh[0]

        sp_input["Rh"] = batch["Rh"]
        sp_input["R"] = batch["R"]
        sp_input["Th"] = batch["Th"]
        sp_input["src_imgs"] = batch["src_imgs"]

        sp_input["target_pose"] = batch["target_pose"]
        sp_input["target_K"] = batch["target_K"]
        sp_input["src_poses"] = batch["src_poses"]
        sp_input["src_Ks"] = batch["src_Ks"]

        return sp_input

    def render(self, batch):
        torch.cuda.synchronize()
        start_time = time.time()

        rays_o = batch["ray_o"]
        rays_d = batch["ray_d"]
        near = batch["near"]
        far = batch["far"]
        device = rays_o.device
        sh = rays_o.shape

        src_imgs = batch["src_imgs"]
        img_size = src_imgs.shape[-2:]
        featmaps = self.encoder(src_imgs.squeeze(0))

        torch.cuda.synchronize()
        encoder_time = time.time() - start_time
        start_time = time.time()

        src_poses = batch["src_poses"]
        src_Ks = batch["src_Ks"]
        target_pose = batch["target_pose"]
        target_K = batch["target_K"]

        # NOTE: Seems like ibrnet dont need to normalize src imgs
        src_imgs = batch["src_imgs"]
        src_imgs = src_imgs * 0.5 + 0.5

        target_pose_h = (
            torch.eye(4, device=device).unsqueeze(0).repeat(target_pose.shape[0], 1, 1)
        )
        target_pose_h[:, :3, :4] = target_pose
        target_K_h = (
            torch.eye(4, device=device).unsqueeze(0).repeat(target_pose.shape[0], 1, 1)
        )
        target_K_h[:, :3, :3] = target_K

        src_poses_h = (
            torch.eye(4, device=device)
            .unsqueeze(0)
            .unsqueeze(1)
            .repeat(1, src_poses.shape[1], 1, 1)
        )
        src_poses_h[:, :, :3, :4] = src_poses
        src_Ks_h = (
            torch.eye(4, device=device)
            .unsqueeze(0)
            .unsqueeze(1)
            .repeat(1, src_poses.shape[1], 1, 1)
        )
        src_Ks_h[:, :, :3, :3] = src_Ks

        src_cameras = torch.ones((1, src_poses.shape[1], 2 + 16 * 2), device=device)
        src_cameras[:, :, 0] = img_size[0]
        src_cameras[:, :, 1] = img_size[1]
        src_cameras[:, :, 2:18] = src_Ks_h.reshape(1, src_poses.shape[1], -1)
        src_cameras[:, :, -16:] = src_poses_h.reshape(1, src_poses.shape[1], -1)

        rays = torch.cat(
            [rays_o, rays_d, near.unsqueeze(-1), far.unsqueeze(-1)], dim=-1
        )
        ret = self.batchify_rays(
            rays, featmaps, src_imgs, src_cameras, self.chunk, batch
        )

        torch.cuda.synchronize()
        render_time = time.time() - start_time
        ret["etime"] = encoder_time
        ret["rtime"] = render_time
        return ret


class Projector:
    def __init__(self, device, neg_ray=False):
        self.device = device
        self.neg_ray = neg_ray

    def inbound(self, pixel_locations, h, w):
        """
        check if the pixel locations are in valid range
        :param pixel_locations: [..., 2]
        :param h: height
        :param w: weight
        :return: mask, bool, [...]
        """
        return (
            (pixel_locations[..., 0] <= w - 1.0)
            & (pixel_locations[..., 0] >= 0)
            & (pixel_locations[..., 1] <= h - 1.0)
            & (pixel_locations[..., 1] >= 0)
        )

    def normalize(self, pixel_locations, h, w):
        resize_factor = torch.tensor([w - 1.0, h - 1.0]).to(pixel_locations.device)[
            None, None, :
        ]
        normalized_pixel_locations = (
            2 * pixel_locations / resize_factor - 1.0
        )  # [n_views, n_points, 2]
        return normalized_pixel_locations

    def compute_projections(self, xyz, train_cameras):
        """
        project 3D points into cameras
        :param xyz: [..., 3]
        :param train_cameras: [n_views, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :return: pixel locations [..., 2], mask [...]
        """
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        num_views = len(train_cameras)
        train_intrinsics = train_cameras[:, 2:18].reshape(-1, 4, 4)  # [n_views, 4, 4]
        train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
        xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [n_points, 4]
        projections = train_intrinsics.bmm(train_poses).bmm(
            xyz_h.t()[None, ...].repeat(num_views, 1, 1)
        )  # [n_views, 4, n_points]
        projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]
        pixel_locations = (
            projections[..., :2] / projections[..., 2:3]
        )  # [n_views, n_points, 2]
        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
        # to check with the camera direction
        if self.neg_ray:
            mask = projections[..., 2] < 0  # a point is invalid if behind the camera
        else:
            mask = projections[..., 2] > 0
        return pixel_locations.reshape(
            (num_views,) + original_shape + (2,)
        ), mask.reshape((num_views,) + original_shape)

    def compute(self, xyz, train_imgs, train_cameras, featmaps):
        """
        :param xyz: [n_rays, n_samples, 3]
        :param train_imgs: [1, n_views, 3, h, w]
        :param train_cameras: [1, n_views, 34]
        :param featmaps: [n_views, d, h, w]
        :return: rgb_feat_sampled: [n_rays, n_samples, 3+n_feat],
                 mask: [n_rays, n_samples, 1]
        """
        # only support batch_size=1 for now
        assert (train_imgs.shape[0] == 1) and (train_cameras.shape[0] == 1)

        train_imgs = train_imgs.squeeze(0)  # [n_views, 3, h, w]
        train_cameras = train_cameras.squeeze(0)  # [n_views, 34]

        h, w = train_cameras[0][:2]

        # compute the projection of the query points to each reference image
        target_pose = train_cameras[0][-16:].reshape(-1, 4, 4)[:, :3, :4]
        target_K = train_cameras[0][2:18].reshape(-1, 4, 4)[:, :3, :3]

        pixel_locations, mask_in_front = self.compute_projections(xyz, train_cameras)
        normalized_pixel_locations = self.normalize(
            pixel_locations, h, w
        )  # [n_views, n_rays, n_samples, 2]
        # rgb sampling
        rgbs_sampled = F.grid_sample(
            train_imgs, normalized_pixel_locations, align_corners=True
        )
        rgb_sampled = rgbs_sampled.permute(
            2, 3, 0, 1
        )  # [n_rays, n_samples, n_views, 3]

        # deep feature sampling
        feat_sampled = F.grid_sample(
            featmaps, normalized_pixel_locations, align_corners=True
        )
        feat_sampled = feat_sampled.permute(
            2, 3, 0, 1
        )  # [n_rays, n_samples, n_views, d]
        rgb_feat_sampled = torch.cat(
            [rgb_sampled, feat_sampled], dim=-1
        )  # [n_rays, n_samples, n_views, d+3]

        # mask
        inbound = self.inbound(pixel_locations, h, w)
        mask = (
            (inbound * mask_in_front).float().permute(1, 2, 0)[..., None]
        )  # [n_rays, n_samples, n_views, 1]
        return rgb_feat_sampled, mask


    def compute_smpl(self, smpl_xyz, train_cameras, featmaps):
        train_cameras = train_cameras.squeeze(0)  # [n_views, 34]

        h, w = train_cameras[0][:2]

        # compute the projection of the query points to each reference image
        smpl_pixel_locations, smpl_mask_in_front = self.compute_projections(
            smpl_xyz, train_cameras
        )

        smpl_normalized_pixel_locations = self.normalize(
            smpl_pixel_locations, h, w
        )  # [n_views, n_rays, n_samples, 2]
        smpl_feat_sampled = F.grid_sample(
            featmaps, smpl_normalized_pixel_locations, align_corners=True
        )
        smpl_feat_sampled = smpl_feat_sampled.permute(
            2, 3, 0, 1
        )  # 3, 32, 1, 6890 -> 1, 6890, 3, 32

        return smpl_feat_sampled


def build_render(cfg):
    # build encoder
    encoder = getattr(impm(cfg.encoder.file), "build_encoder")(cfg)
    # build nerfhead
    nerfhead = getattr(impm(cfg.head.file), "build_head")(cfg)

    if "thuman" in cfg.dataset.train.name:
        neg_ray_train = True
    else:
        neg_ray_train = False
    if "thuman" in cfg.dataset.test.name:
        neg_ray_val = True
    else:
        neg_ray_val = False
    is_train = nerfhead.training or encoder.training
    if is_train:
        chunk = cfg.dataset.train.chunk
    else:
        chunk = cfg.dataset.test.chunk
    if cfg.head.rgb.use_rgbhead is False:
        mesh_th = 1.0 / cfg.test.mesh_th
    else:
        mesh_th = -1
    render_config = {
        "encoder": encoder,
        "nerfhead": nerfhead,
        "is_train": False,
        "neg_ray_train": neg_ray_train,
        "neg_ray_val": neg_ray_val,
        "n_rays": cfg.train.n_rays,
        "n_samples": cfg.train.n_samples,
        "voxel_size": cfg.dataset.voxel_size,
        "chunk": chunk,
        "mesh_th": mesh_th,
    }
    render = Renderer(**render_config)
    return render
