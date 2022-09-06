import cv2
import numpy as np
import trimesh


def clear_msk_noise(msk, tag):
    flag_msk = msk.copy()
    flag_msk[msk!=tag] = 0
    flag_msk[msk==tag] = 1
    _, contours, hierarchy = cv2.findContours(flag_msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    for ci in range(len(contours)):
        area = cv2.contourArea(contours[ci])
        if area == 0:
            cv2.fillPoly(msk, [contours[ci]], 0)
    return msk


def get_nearest_camids(tar_cam_loc, ref_cam_locs, num_select, tar_id=-1, angular_dist_method='dist',
                         scene_center=(0, 0, 0), far_flag=False):
    '''
    Args:
        # tar_pose: target pose [3, 3]
        # ref_poses: reference poses [N, 3, 3]
        tar_cam_locs: target camera location [3]
        ref_cam_locs: reference camera locations [N, 3]
        num_select: the number of nearest views to select
    Returns: the selected indices
    '''
    num_cams = len(ref_cam_locs)
    num_select = min(num_select, num_cams-1)
    if angular_dist_method == 'dist':
        dists = np.linalg.norm(tar_cam_loc - ref_cam_locs, axis=1)
    else:
        raise Exception('unknown angular distance calculation method!')

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself

    if far_flag is True:
        dists = dists * -1
    sorted_ids = np.argsort(dists)
    selected_ids = sorted_ids[:num_select]
    return selected_ids


def get_rays(H, W, K, R, T):
    # calculate the camera origin
    R_inv = np.linalg.inv(R)
    T = -R_inv @ T
    rays_o = T.ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy') 
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = (pixel_camera @ R_inv.T) + T[np.newaxis, ...]
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)

    return rays_o, rays_d


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def get_bound_2d_mask(bounds, K, RT, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, RT)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    ray_d[np.abs(ray_d) < 1e-5] = 1e-5
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)
    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    neg_mask = np.array(((p_intervals[:, 0] - ray_o) * ray_d).sum(axis=1)<0.0, dtype=np.int) * -2 + 1
    d0 = d0 * neg_mask
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    neg_mask = np.array(((p_intervals[:, 0] - ray_o) * ray_d).sum(axis=1)<0.0, dtype=np.int) * -2 + 1
    d1 = d1 * neg_mask
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)
    return near, far, mask_at_box


def get_nearest_pose_ids(tar_cam_loc, ref_cam_locs, num_select, tar_id=-1, angular_dist_method='dist',
                         scene_center=(0, 0, 0)):
    '''
    Args:
        # tar_pose: target pose [3, 3]
        # ref_poses: reference poses [N, 3, 3]
        tar_cam_locs: target camera location [3]
        ref_cam_locs: reference camera locations [N, 3]
        num_select: the number of nearest views to select
    Returns: the selected indices
    '''
    num_cams = len(ref_cam_locs)
    if num_cams <= 3:
        num_select = min(num_select, num_cams)
    else:
        num_select = min(num_select, num_cams-1)
    if angular_dist_method == 'dist':
        dists = np.linalg.norm(tar_cam_loc - ref_cam_locs, axis=1)
    else:
        raise Exception('unknown angular distance calculation method!')

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself

    sorted_ids = np.argsort(dists)
    selected_ids = sorted_ids[:num_select]
    
    return selected_ids


def load_obj_data(filename):
    """load model data from .obj file"""
    v_list = []  # vertex coordinate
    vt_list = []  # vertex texture coordinate
    vc_list = []  # vertex color
    vn_list = []  # vertex normal
    f_list = []  # face vertex indices
    fn_list = []  # face normal indices
    ft_list = []  # face texture indices

    # read data
    fp = open(filename, 'r')
    lines = fp.readlines()
    fp.close()

    for line in lines:
        line_data = line.strip().split(' ')

        # parse vertex cocordinate
        if line_data[0] == 'v':
            v_list.append((float(line_data[1]), float(line_data[2]), float(line_data[3])))
            if len(line_data) == 7:
                vc_list.append((float(line_data[4]), float(line_data[5]), float(line_data[6])))
            else:
                vc_list.append((0.5, 0.5, 0.5))

        # parse vertex texture coordinate
        if line_data[0] == 'vt':
            vt_list.append((float(line_data[1]), float(line_data[2])))

        # parse vertex normal
        if line_data[0] == 'vn':
            vn_list.append((float(line_data[1]), float(line_data[2]), float(line_data[3])))

        # parse face
        if line_data[0] == 'f':
            # used for parsing face element data
            def segElementData(ele_str):
                fv = None
                ft = None
                fn = None
                eles = ele_str.strip().split('/')
                if len(eles) == 1:
                    fv = int(eles[0]) - 1
                elif len(eles) == 2:
                    fv = int(eles[0]) - 1
                    ft = int(eles[1]) - 1
                elif len(eles) == 3:
                    fv = int(eles[0]) - 1
                    fn = int(eles[2]) - 1
                    ft = None if eles[1] == '' else int(eles[1]) - 1
                return fv, ft, fn

            fv0, ft0, fn0 = segElementData(line_data[1])
            fv1, ft1, fn1 = segElementData(line_data[2])
            fv2, ft2, fn2 = segElementData(line_data[3])
            f_list.append((fv0, fv1, fv2))
            if ft0 is not None and ft1 is not None and ft2 is not None:
                ft_list.append((ft0, ft1, ft2))
            if fn0 is not None and fn1 is not None and fn2 is not None:
                fn_list.append((fn0, fn1, fn2))

    v_list = np.asarray(v_list)
    vn_list = np.asarray(vn_list)
    vt_list = np.asarray(vt_list)
    vc_list = np.asarray(vc_list)
    f_list = np.asarray(f_list)
    ft_list = np.asarray(ft_list)
    fn_list = np.asarray(fn_list)

    model = {'v': v_list, 'vt': vt_list, 'vc': vc_list, 'vn': vn_list,
             'f': f_list, 'ft': ft_list, 'fn': fn_list}
    return model


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    # camera coord
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    # camera coord -> rgb coord
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def sample_ray(img, msk, K, R, T, bounds, nrays, split, body_sample_ratio):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T.reshape(-1, 1)], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    img[bound_mask != 1] = 0
    if len(msk.shape) == 3:
        msk = msk[..., 0]
    msk = msk * bound_mask
    bound_mask[msk == 100] = 0
    if split != 'test':
        nsampled_rays = 0
        body_sample_ratio = body_sample_ratio
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []
        body_msk_list = []
        index_list = set()

        while nsampled_rays < nrays:
            # sample rays on body
            body_msk = clear_msk_noise(msk, 1)
            coord_body = np.argwhere(body_msk == 1)
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_body
            
            if len(coord_body) > 0:
                coord_body = coord_body[np.random.randint(0, len(coord_body), n_body)]
            # sample rays in the bound mask
            coord = np.argwhere(bound_mask == 1)
            coord = coord[np.random.randint(0, len(coord), n_rand)]
            if len(coord_body) > 0:
                coord = np.concatenate([coord_body, coord], axis=0)

            cur_set = set(list(coord[:, 1] * W + coord[:, 0]))
            new_set = cur_set - index_list
            no_repeat_indexs = np.array([index for index in new_set], dtype=np.int)
            coord = coord[:len(no_repeat_indexs)]
            coord[:, 0] = no_repeat_indexs % W
            coord[:, 1] = no_repeat_indexs / W
            index_list.update(new_set)
            
            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]

            out_body_msk = body_msk.copy()
            out_body_msk[out_body_msk>0] = 1
            msk_ = out_body_msk[coord[:, 0], coord[:, 1]]

            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            body_msk_list.append(msk_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            mask_at_box_list.append(mask_at_box[mask_at_box])
            nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        out_body_msk = np.concatenate(body_msk_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)
    else:
        body_msk = clear_msk_noise(msk, 1)
        rgb = img.reshape(-1, img.shape[-1]).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)

        out_body_msk = body_msk.copy()
        out_body_msk[out_body_msk>0] = 1
        out_body_msk = out_body_msk.reshape(-1)[mask_at_box]
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.zeros([len(rgb), 2]).astype(np.int64)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box, out_body_msk


def transform_can_smpl(xyz, rot_ratio=0.0):
    center = np.array([0, 0, 0]).astype(np.float32)
    rot = np.array([[np.cos(0), -np.sin(0)], [np.sin(0), np.cos(0)]])
    rot = rot.astype(np.float32)
    trans = np.array([0, 0, 0]).astype(np.float32)
    if np.random.uniform() > rot_ratio:
        return xyz, center, rot, trans

    xyz = xyz.copy()

    # rotate the smpl
    rot_range = np.pi / 32
    t = np.random.uniform(-rot_range, rot_range)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    rot = rot.astype(np.float32)
    center = np.mean(xyz, axis=0)
    xyz = xyz - center
    xyz[:, [0, 2]] = np.dot(xyz[:, [0, 2]], rot.T)
    xyz = xyz + center

    # translate the smpl
    x_range = 0.05
    z_range = 0.025
    x_trans = np.random.uniform(-x_range, x_range)
    z_trans = np.random.uniform(-z_range, z_range)
    trans = np.array([x_trans, 0, z_trans]).astype(np.float32)
    xyz = xyz + trans

    return xyz, center, rot, trans