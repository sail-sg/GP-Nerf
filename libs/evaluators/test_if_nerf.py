import numpy as np
from skimage.measure import compare_ssim
import os
import cv2


class Evaluator:
    def __init__(self, cfg, seq_name):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.cfg = cfg
        self.seq_name = seq_name

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, rgb_pred, rgb_gt, mask_at_box, batch):
        H, W = int(self.cfg.dataset.H * self.cfg.dataset.ratio), int(
            self.cfg.dataset.W * self.cfg.dataset.ratio)
        mask_at_box = mask_at_box.reshape(H, W)
        # convert the pixels into an image
        img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred
        img_gt = np.zeros((H, W, 3))
        img_gt[mask_at_box] = rgb_gt

        # save results
        if self.cfg.test.save_imgs:
            img = np.concatenate((img_gt , img_pred), axis=1)[..., [2, 1, 0]]
            img_save_path = os.path.join(self.cfg.result_dir,
                self.seq_name)
            os.system('mkdir -p {}'.format(img_save_path))

            img_name = str(batch['frame_index'].item())+'_cam'+str(batch['cam_ind'].item())+'.jpg'
            cv2.imwrite(os.path.join(img_save_path, img_name), img * 255)

        # crop the object region
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        img_pred = img_pred[y:y + h, x:x + w]
        img_gt = img_gt[y:y + h, x:x + w]
        # compute the ssim
        ssim = compare_ssim(img_pred, img_gt, multichannel=True)
        return ssim

    def evaluate(self, output, batch):
        rgb_pred = output['rgb'].detach().cpu().numpy()
        ori_rgb_gt = batch['rgb'].detach().cpu().numpy()
        ori_rgb_pred = np.zeros_like(ori_rgb_gt)
        if 'mask_at_box' not in output.keys():
            mask_at_box = batch['mask_at_box'].detach().cpu().numpy()
            ori_rgb_pred[mask_at_box] = rgb_pred[mask_at_box]
        else:
            mask_at_box = output['mask_at_box']
            ori_rgb_pred[0, mask_at_box] = rgb_pred
        ori_mask_at_box = batch['mask_at_box'].detach().cpu().numpy()
        ori_rgb_gt = ori_rgb_gt[ori_mask_at_box]
        ori_rgb_pred = ori_rgb_pred[ori_mask_at_box]
 
        mse = np.mean((ori_rgb_pred - ori_rgb_gt)**2)
        self.mse.append(mse)

        psnr = self.psnr_metric(ori_rgb_pred, ori_rgb_gt)
        self.psnr.append(psnr)
        
        ssim = self.ssim_metric(ori_rgb_pred, ori_rgb_gt, ori_mask_at_box[0], batch)
        self.ssim.append(ssim)

    def summarize(self):
        result_path = os.path.join(self.cfg.result_dir,
            self.seq_name, 'metrics.npy')
        os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
        mse = np.mean(self.mse)
        psnr = np.mean(self.psnr)
        ssim = np.mean(self.ssim)
        metrics = {'mse': mse, 'psnr': psnr, 'ssim': ssim}
        np.save(result_path, self.mse)
        print('mse: {}'.format(mse))
        print('psnr: {}'.format(psnr))
        print('ssim: {}'.format(ssim))
        self.mse = []
        self.psnr = []
        self.ssim = []
        return metrics