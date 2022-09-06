from importlib import import_module as impm
import mcubes
import numpy as np
import trimesh

import torch
import torch.nn as nn
import torch.nn.functional as F


class Renderer(nn.Module):
    def __init__(self, 
                 encoder,
                 nerfhead,
                 is_train=True,
                 neg_ray_train=False,
                 neg_ray_val=False,
                 n_rays=1024,
                 n_samples=64,
                 voxel_size=[0.005, 0.005, 0.005],
                 chunk=64,
                 mesh_th=-1):
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
    
    def get_sampling_points(self, ray_o, ray_d, near, far, perturb=1):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=self.n_samples).to(near)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if perturb > 0. and self.is_train:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand
        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals
    
    def pts_to_can_pts(self, pts, batch):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = batch['Th'][:, None]
        pts = pts - Th
        R = batch['Rh']
        sh = pts.shape
        pts = torch.matmul(pts.view(sh[0], -1, sh[3]), R)
        pts = pts.view(*sh)
        return pts

    def get_grid_coords(self, pts, sp_input, batch):
        # convert xyz to the voxel coordinate dhw
        dhw = pts[..., [2, 1, 0]]
        min_dhw = batch['bounds'][:, 0, [2, 1, 0]]
        dhw = dhw - min_dhw[:, None, None]
        dhw = dhw / torch.tensor(self.voxel_size).to(dhw)
        # convert the voxel coordinate to [-1, 1]
        out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
        dhw = dhw / out_sh * 2 - 1
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords

    @staticmethod
    def raw2outputs(raw, z_vals, mask, neg):
        '''
        :param raw: raw network output; tensor of shape [N_rays, N_samples, 4]
        :param z_vals: depth of point samples along rays; tensor of shape [N_rays, N_samples]
        :param ray_d: [N_rays, 3]
        :return: {'rgb': [N_rays, 3], 'depth': [N_rays,], 'weights': [N_rays,], 'depth_std': [N_rays,]}
        '''
        rgb = raw[:, :, :3]     # [N_rays, N_samples, 3]
        sigma = raw[:, :, 3]    # [N_rays, N_samples]

        if neg is True:
            rgb = torch.flip(rgb, [1])
            sigma = torch.flip(sigma, [1])

        sigma2alpha = lambda sigma: 1. - torch.exp(-sigma)
        alpha = sigma2alpha(sigma)  # [N_rays, N_samples]

        # Eq. (3): T
        T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :-1]   # [N_rays, N_samples-1]
        T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]

        # maths show weights, and summation of weights along a ray, are always inside [0, 1]
        weights = alpha * T     # [N_rays, N_samples]
        rgb_map = torch.sum(weights.unsqueeze(2) * rgb, dim=1)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, dim=-1) # [N_rays,]
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map).to(depth_map),
                                depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)
        mask = mask.float().sum(dim=1) > 8 # should at least have 8 valid observation on the ray, otherwise don't consider its loss
        
        return rgb_map, disp_map, acc_map, weights, depth_map, mask, alpha


    def render_rays(self, rays, neg_ray, featmaps, src_imgs, src_cameras, batch):
        if self.nerfhead.use_rgbhead is True:
            rays_o, rays_d = rays[..., :3], rays[..., 3:6]
            near, far = rays[..., 6], rays[..., 7]
            device = rays_o.device
            pts, z_vals = self.get_sampling_points(rays_o, rays_d, near, far)
        else:
            pts = rays
        sh = pts.shape
        device = pts.device
        viewdir = rays_d / torch.norm(rays_d, dim=2, keepdim=True)
        pts_smpl = self.pts_to_can_pts(pts.float(), batch)
        sp_input = self.prepare_sp_input(batch)
        
        # reshape to [batch_size, n, 3]
        grid_coords = self.get_grid_coords(pts_smpl, sp_input, batch)
        grid_coords = grid_coords.view(sh[0], -1, 3)

        xyz = batch['feature'][..., :3]
        R = batch['Rh'].float()
        Th = batch['Th'].float()
        smpl_xyz = torch.bmm(xyz, R.transpose(1, 2)) + Th
        
        N_rays, N_samples = pts.shape[1:3]
        projector = Projector(device, neg_ray=neg_ray)

        rgb_feat, smpl_feat_sampled, mask = projector.compute(
            pts.squeeze(0), smpl_xyz,
            src_imgs, src_cameras, featmaps=featmaps)
        pixel_mask = mask[..., 0].sum(dim=2) > 1  # [N_rays, N_samples], should at least have 2 observations
        raw, rgb_in = self.nerfhead(sp_input, grid_coords, smpl_feat_sampled, rgb_feat, mask)
        if self.nerfhead.use_rgbhead is False:
            ret = {'sigma': raw}
        else:
            rgb_map, disp_map, acc_map, weights, depth_map, mask, alpha = \
                self.raw2outputs(
                    raw, z_vals.squeeze(0), pixel_mask, neg=neg_ray)
            rgb_in_map = (weights.unsqueeze(-1).unsqueeze(-1)*rgb_in).sum(dim=1)
            ret = {
                'rgb_map': rgb_map,
                'disp_map': disp_map,
                'acc_map': acc_map,
                'depth_map': depth_map,
                "alpha": weights,
                "z_vals": z_vals.squeeze(0),
                "rgb_in_map": rgb_in_map
            }
        return ret


    def batchify_rays(self, rays, featmaps, src_imgs, src_cameras,
                      chunk, batch):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        if batch['body_msk'].shape[-1] > self.n_rays:
            neg_ray = self.neg_ray_val
        else:
            neg_ray = self.neg_ray_train
        for i in range(0, rays.shape[1], chunk):
            ret = self.render_rays(
                rays[:, i: i+chunk], neg_ray, featmaps,
                src_imgs, src_cameras, batch)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        
        if self.nerfhead.use_rgbhead is False:
            sigma = torch.cat(all_ret['sigma'], 0)
            all_ret = {'sigma': sigma}
        else:
            all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

        return all_ret

    
    def prepare_sp_input(self, batch):
        # feature, coordinate, shape, batch size
        sp_input = {}

        # feature: [N, f_channels]
        sh = batch['feature'].shape  # smpl coordiante
        sp_input['feature'] = batch['feature'].view(-1, sh[-1])

        # coordinate: [N, 4], batch_idx, z, y, x
        sh = batch['coord'].shape  # smpl coordinate
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]
        idx = torch.cat(idx).to(batch['coord'])
        coord = batch['coord'].view(-1, sh[-1])
        sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1)

        out_sh, _ = torch.max(batch['out_sh'], dim=0)
        sp_input['out_sh'] = out_sh.tolist()
        sp_input['batch_size'] = sh[0]

        sp_input['Rh'] = batch['Rh']
        sp_input['R'] = batch['R']
        sp_input['src_imgs'] = batch['src_imgs']
        return sp_input

    def render(self, batch):
        rays_o = batch['ray_o']
        rays_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        device = rays_o.device
        bz = rays_o.size(0)
        sh = rays_o.shape

        src_imgs = batch['src_imgs']
        img_size = src_imgs.shape[-2:]
        featmaps = self.encoder(src_imgs.squeeze(0))

        src_poses = batch['src_poses']
        src_Ks = batch['src_Ks']
        target_pose = batch['target_pose']
        target_K = batch['target_K']

        # NOTE: Seems like ibrnet dont need to normalize src imgs
        src_imgs = batch['src_imgs']
        src_imgs = src_imgs * 0.5 + 0.5

        target_pose_h = torch.eye(4, device=device).unsqueeze(0).repeat(target_pose.shape[0], 1, 1)
        target_pose_h[:, :3, :4] = target_pose
        target_K_h = torch.eye(4, device=device).unsqueeze(0).repeat(target_pose.shape[0], 1, 1)
        target_K_h[:, :3, :3] = target_K

        src_poses_h = torch.eye(4, device=device).unsqueeze(0).unsqueeze(1).repeat(1, src_poses.shape[1], 1, 1)
        src_poses_h[:, :, :3, :4] = src_poses
        src_Ks_h = torch.eye(4, device=device).unsqueeze(0).unsqueeze(1).repeat(1, src_poses.shape[1], 1, 1)
        src_Ks_h[:, :, :3, :3] = src_Ks

        src_cameras = torch.ones((1, src_poses.shape[1], 2+16*2), device=device)
        src_cameras[:, :, 0] = img_size[0]
        src_cameras[:, :, 1] = img_size[1]
        src_cameras[:, :, 2:18] = src_Ks_h.reshape(1, src_poses.shape[1], -1)
        src_cameras[:, :, -16:] = src_poses_h.reshape(1, src_poses.shape[1], -1)

        if self.nerfhead.use_rgbhead:
            rays = torch.cat([rays_o, rays_d, near.unsqueeze(-1), far.unsqueeze(-1)], dim=-1)
            ret = self.batchify_rays(
                rays, featmaps, src_imgs, src_cameras,
                self.chunk, batch)
            ret = {k: v.view(*sh[:-1], -1) for k, v in ret.items()}
        else:
            inside = batch['inside'][0].bool()
            pts = batch['pts']
            sh = pts.shape
            pts = pts[0][inside][None].view(sh[0], -1, 1, 3)
            ret = self.batchify_rays(
                pts, featmaps, src_imgs, src_cameras,
                self.chunk, batch)
            sigma = ret['sigma'][:, 0, 0]
            sigma2alpha = lambda sigma, dists: 1. - torch.exp(-sigma)
            alpha = sigma2alpha(sigma, None).detach().cpu().numpy()
            cube = np.zeros(sh[1:-1])
            inside = inside.detach().cpu().numpy()
            cube[inside == 1] = alpha
            cube = np.pad(cube, 10, mode='constant')
            vertices, triangles = mcubes.marching_cubes(cube, self.mesh_th)
            mesh = trimesh.Trimesh(vertices, triangles)
            ret = {'cube': cube, 'mesh': mesh}

        return ret



class Projector():
    def __init__(self, device, neg_ray=False):
        self.device = device
        self.neg_ray = neg_ray

    def inbound(self, pixel_locations, h, w):
        '''
        check if the pixel locations are in valid range
        :param pixel_locations: [..., 2]
        :param h: height
        :param w: weight
        :return: mask, bool, [...]
        '''
        return (pixel_locations[..., 0] <= w - 1.) & \
               (pixel_locations[..., 0] >= 0) & \
               (pixel_locations[..., 1] <= h - 1.) &\
               (pixel_locations[..., 1] >= 0)

    def normalize(self, pixel_locations, h, w):
        resize_factor = torch.tensor([w-1., h-1.]).to(pixel_locations.device)[None, None, :]
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
        return normalized_pixel_locations

    def compute_projections(self, xyz, train_cameras):
        '''
        project 3D points into cameras
        :param xyz: [..., 3]
        :param train_cameras: [n_views, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :return: pixel locations [..., 2], mask [...]
        '''
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        num_views = len(train_cameras)
        train_intrinsics = train_cameras[:, 2:18].reshape(-1, 4, 4)  # [n_views, 4, 4]
        train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
        xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [n_points, 4]
        projections = train_intrinsics.bmm(train_poses).bmm(xyz_h.t()[None, ...].repeat(num_views, 1, 1))  # [n_views, 4, n_points]
        projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]
        pixel_locations = projections[..., :2] / projections[..., 2:3]  # [n_views, n_points, 2]
        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
        # to check with the camera direction
        if self.neg_ray:
            mask = projections[..., 2] < 0   # a point is invalid if behind the camera
        else:
            mask = projections[..., 2] > 0
        return pixel_locations.reshape((num_views, ) + original_shape + (2, )), \
                mask.reshape((num_views, ) + original_shape)

    def compute(self, xyz, smpl_xyz, train_imgs, train_cameras, featmaps):
        '''
        :param xyz: [n_rays, n_samples, 3]
        :param train_imgs: [1, n_views, 3, h, w]
        :param train_cameras: [1, n_views, 34]
        :param featmaps: [n_views, d, h, w]
        :return: rgb_feat_sampled: [n_rays, n_samples, 3+n_feat],
                 mask: [n_rays, n_samples, 1]
        '''
        # only support batch_size=1 for now
        assert (train_imgs.shape[0] == 1) and (train_cameras.shape[0] == 1) 

        train_imgs = train_imgs.squeeze(0)  # [n_views, 3, h, w]
        train_cameras = train_cameras.squeeze(0)  # [n_views, 34]

        h, w = train_cameras[0][:2]
        
        # compute the projection of the query points to each reference image
        smpl_pixel_locations, smpl_mask_in_front = self.compute_projections(smpl_xyz, train_cameras)    
        smpl_normalized_pixel_locations = self.normalize(smpl_pixel_locations, h, w)   # [n_views, n_rays, n_samples, 2]
        smpl_feat_sampled = F.grid_sample(featmaps, smpl_normalized_pixel_locations, align_corners=True)
        smpl_feat_sampled = smpl_feat_sampled.permute(2, 3, 0, 1) # 3, 32, 1, 6890 -> 1, 6890, 3, 32

        pixel_locations, mask_in_front = self.compute_projections(xyz, train_cameras)
        normalized_pixel_locations = self.normalize(pixel_locations, h, w)   # [n_views, n_rays, n_samples, 2]
        # rgb sampling
        rgbs_sampled = F.grid_sample(train_imgs, normalized_pixel_locations, align_corners=True)
        rgb_sampled = rgbs_sampled.permute(2, 3, 0, 1)  # [n_rays, n_samples, n_views, 3]

        # deep feature sampling
        feat_sampled = F.grid_sample(featmaps, normalized_pixel_locations, align_corners=True)
        feat_sampled = feat_sampled.permute(2, 3, 0, 1)  # [n_rays, n_samples, n_views, d]
        rgb_feat_sampled = torch.cat([rgb_sampled, feat_sampled], dim=-1)   # [n_rays, n_samples, n_views, d+3]

        # mask
        inbound = self.inbound(pixel_locations, h, w)
        mask = (inbound * mask_in_front).float().permute(1, 2, 0)[..., None]   # [n_rays, n_samples, n_views, 1]
        return rgb_feat_sampled, smpl_feat_sampled, mask



def build_render(cfg):
    # build encoder
    encoder = getattr(impm(cfg.encoder.file), 'build_encoder')(cfg)
    # build nerfhead
    nerfhead = getattr(impm(cfg.head.file), 'build_head')(cfg)

    if 'thuman' in cfg.dataset.train.name:
        neg_ray_train = True
    else:
        neg_ray_train = False
    if 'thuman' in cfg.dataset.test.name:
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
        'encoder': encoder,
        'nerfhead': nerfhead,
        'is_train': is_train,
        'neg_ray_train': neg_ray_train,
        'neg_ray_val': neg_ray_val,
        'n_rays': cfg.train.n_rays,
        'n_samples': cfg.train.n_samples,
        'voxel_size': cfg.dataset.voxel_size,
        'chunk': chunk,
        'mesh_th': mesh_th
    }
    render = Renderer(**render_config)
    return render