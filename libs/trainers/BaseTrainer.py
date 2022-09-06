import cv2
import datetime
import logging
import math
import numpy as np
import os
import sys
import time
from tqdm import tqdm

import torch
from torch import autograd
from tensorboardX import SummaryWriter

from libs.evaluators.if_nerf import Evaluator
from libs.evaluators.if_nerf_mesh import Evaluator as MeshEvaluator
import libs.utils.misc as utils
from libs.utils.utils import (save_checkpoint,
                              write_dict_to_json)


def data_loop(data_loader):
    """
    Loop an iterable infinitely
    """
    while True:
        for x in iter(data_loader):
            yield x


def pts_render(pts_xy, rgb, alpha, z_array, H, W, hold_len, sample_num, mask_at_box, neg=False):
    pred_img = np.zeros((H, W, 3))
    depth = {}
    hold_rgb = rgb[-hold_len*sample_num:].reshape(hold_len, sample_num, -1)
    rgb = rgb[:-hold_len*sample_num]
    hold_alpha = alpha[-hold_len*sample_num:].reshape(hold_len, sample_num)
    alpha = alpha[:-hold_len*sample_num]
    new_valid = np.where(alpha > 1e-14)[0]
    alpha = alpha[new_valid]
    rgb = rgb[new_valid]

    if neg is True:
        hold_alpha = hold_alpha[..., ::-1]
        hold_rgb = hold_rgb[..., ::-1, :]
    T = np.cumprod(1. - hold_alpha + 1e-10, axis=-1)[..., :-1]
    T = np.concatenate((np.ones_like(T[..., 0:1]), T), axis=-1)
    weights = hold_alpha * T
    rgb_map = np.sum(weights[..., np.newaxis] * hold_rgb, axis=1)
    mask_at_box = mask_at_box.reshape(H, W)
    pred_img[mask_at_box] = rgb_map

    return pred_img, rgb_map
        

class Trainer(object):
    def __init__(self,
                 cfg,
                 render,
                 criterion,
                 optimizer,
                 lr_scheduler,
                 logger,
                 log_dir,
                 performance_indicator='psnr',
                 last_iter=-1,
                 rank=0,
                 device='cuda'):
        self.cfg = cfg
        self.render = render
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.logger = logger
        if log_dir:
            self.log_dir = os.path.join(log_dir, self.cfg.output_dir)
            self.epoch = last_iter + 1
        self.PI = performance_indicator
        self.rank = rank
        self.best_performance = 0.0
        self.is_best = False
        self.max_epoch = self.cfg.train.max_epoch
        self.model_name = self.cfg.render.file
        self.device = device
        self.iter_count = 0
        if self.optimizer is not None and rank == 0:
            self.writer = SummaryWriter(self.log_dir, comment=f'_rank{rank}')
            logging.info(f"max epochs = {self.max_epoch} ")

    def _read_inputs(self, batch):
        for k in batch:
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                batch[k] = [b.to(self.device) for b in batch[k]]
            if isinstance(batch[k], dict):
                batch[k] = {key: value.to(self.device) for key, value in batch[k].items()}
            else:
                batch[k] = batch[k].to(self.device)
        return batch

    def _forward(self, data):
        ret = self.render.module.render(data)
        loss = self.criterion(ret, data, is_train=True)
        return loss

    def train(self, train_loader, eval_loader):
        self.evaluator = Evaluator(self.cfg, 'eval')
        start_time = time.time()
        self.render.train()
        self.criterion.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(self.epoch)
        print_freq = self.cfg.train.print_freq
        eval_data_iter = data_loop(eval_loader)
        if self.epoch > self.max_epoch:
            logging.info("Optimization is done !")
            sys.exit(0)
        for data in metric_logger.log_every(train_loader, print_freq, header, self.logger):
            data = self._read_inputs(data)
            loss_dict = self._forward(data)   
            losses = sum(loss_dict[k] for k in loss_dict.keys())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_value = sum(loss_dict_reduced.values()).item()
            if not math.isfinite(loss_value):
                self.logger.info("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward(retain_graph=True)
            self.optimizer.step()
            metric_logger.update(loss=loss_value, **loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

            self.iter_count += 1
            # quick val
            if self.rank == 0 and self.iter_count % self.cfg.train.valiter_interval == 0:
                # evaluation
                if self.cfg.train.val_when_train:
                    performance = self.quick_val(eval_data_iter)
                    self.writer.add_scalar(self.PI, performance, self.iter_count)
                    logging.info('Now: {} is {:.4f}'.format(self.PI, performance))

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': self.epoch, 'iter': self.iter_count}
        if self.rank == 0:
            for (key, val) in log_stats.items():
                self.writer.add_scalar(key, val, log_stats['iter'])
        self.lr_scheduler.step()

        # save checkpoint
        if self.rank == 0 and self.epoch > 0 and self.epoch % self.cfg.train.save_interval == 0:
            if self.cfg.train.val_when_train:
                performance = self.quick_val(eval_data_iter)
                self.writer.add_scalar(self.PI, performance, self.iter_count)  
                if performance > self.best_performance:
                    self.is_best = True
                    self.best_performance = performance
                else:
                    self.is_best = False
                logging.info(f'Now: best {self.PI} is {self.best_performance}')
            else:
                performance = -1

            # save checkpoint
            try:
                state_dict = self.render.module.state_dict() # remove prefix of multi GPUs
            except AttributeError:
                state_dict = self.render.state_dict()

            if self.rank == 0:
                if self.cfg.train.save_every_checkpoint:
                    filename = f"{self.epoch}.pth"
                else:
                    filename = "latest.pth"
                save_dir = os.path.join(self.log_dir, self.cfg.output_dir)
                save_checkpoint(
                    {
                        'epoch': self.epoch,
                        'model': self.model_name,
                        f'performance/{self.PI}': performance,
                        'state_dict': state_dict,
                        'optimizer': self.optimizer.state_dict(),
                    },
                    self.is_best,
                    save_dir,
                    filename=f'{filename}'
                )
                # remove previous pretrained model if the number of models is too big
                pths = [
                    int(pth.split('.')[0]) for pth in os.listdir(save_dir)
                     if pth != 'latest.pth' and pth != 'model_best.pth'
                ]
                if len(pths) > 30:
                    os.system('rm {}'.format(
                        os.path.join(save_dir, '{}.pth'.format(min(pths)))))
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('Training time {}'.format(total_time_str))
        self.logger.info('Training time {}'.format(total_time_str))
        self.epoch += 1

    def quick_val(self, eval_data_iter):
        self.render.eval()
        self.criterion.eval()
        val_stats = {}
        image_stats = {}
        H, W = int(self.cfg.dataset.H * self.cfg.dataset.ratio), \
             int(self.cfg.dataset.W * self.cfg.dataset.ratio)
        with torch.no_grad():
            val_data = next(eval_data_iter)
            val_data = self._read_inputs(val_data)
            ret = self.render.module.render(val_data)
            image_stats.update(self.process_img(ret, val_data, W, H))
            loss_dict = self.criterion(ret, val_data, is_train=False)
            self.evaluator.evaluate(ret, val_data)
            loss_stats = utils.reduce_dict(loss_dict)
            for k, v in loss_stats.items():
                val_stats.setdefault(k, 0)
                val_stats[k] += v
            result = {
                'mse': self.evaluator.mse[-1], 
                'psnr': self.evaluator.psnr[-1], 
                'ssim': self.evaluator.ssim[-1],
            }
            val_stats.update(result)

        # save metrics and loss
        log_stats = {**{f'eval_{k}': v for k, v in val_stats.items()},
                     'epoch': self.epoch, 'iter': self.iter_count}
        for (key, val) in log_stats.items():
            self.writer.add_scalar(key, val, log_stats['iter'])

        # save_img
        if image_stats is not None:
            pattern = 'val_iter/{}'
            for k, v in image_stats.items():
                if v.shape[0] == 3:
                    v = np.transpose(v, (1, 2, 0))
                self.writer.add_image(pattern.format(k), v, log_stats['iter'])

        rgb_loss, mse, psnr, ssim = val_stats['rgb_loss'], val_stats['mse'], val_stats['psnr'], val_stats['ssim']
        msg = 'rgb_loss: {:.4f}, mse: {:.4f}, psnr: {:.4f}, ssim: {:.4f}'.format(rgb_loss, mse, psnr, ssim)
        self.logger.info(msg)

        self.render.train()
        self.criterion.train()
        return val_stats[self.PI]


    def evaluate(self, eval_loader, result_path, is_vis=False):
        self.render.eval()
        self.evaluator = Evaluator(self.cfg, self.cfg.test.test_seq)
        count = 0
        H, W = int(self.cfg.dataset.H * self.cfg.dataset.ratio), \
             int(self.cfg.dataset.W * self.cfg.dataset.ratio)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        total_time = 0.0
        for data in tqdm(eval_loader):
            with torch.no_grad():
                val_data = self._read_inputs(data)
                ret = self.render.module.render(val_data)
                # visual
                if is_vis:
                    vis = self.process_img(ret, val_data, W, H)['render_img'].transpose(1, 2, 0)
                    new_vis = vis.copy()
                    new_vis[..., 0] = vis[..., 2]
                    new_vis[..., 2] = vis[..., 0]
                    cv2.imwrite(f'{result_path}/{count}.jpg', new_vis)
                self.evaluator.evaluate(ret, val_data)
            total_time += ret["rtime"]
            count += 1
        if self.cfg.head.rgb.use_rgbhead:
            self.evaluator.summarize()
        print(f'avg total render time: {total_time / count}s per sample',)



    @staticmethod
    def process_img(pred, batch, W, H):
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        mask_at_box = mask_at_box.reshape(H, W)
        if 'pred_img' not in pred.keys():
            pred_img = np.zeros((H, W, 3))
            pred_img[mask_at_box] = pred['rgb_map'][0][..., :3].detach().cpu().numpy()
        else:
            pred_img = pred['pred_img']
        gt_img = np.zeros((H, W, 3))
        gt_img[mask_at_box] = batch['rgb'][0][..., :3].detach().cpu().numpy()

        src_imgs = batch['src_imgs'][0].permute(0, 2, 3, 1).detach().cpu().numpy()
        # un-normalize source images
        src_imgs = src_imgs * 0.5 + 0.5

        vis_list = [
            *src_imgs,
            gt_img,
            pred_img,
        ]
        vis = np.hstack(vis_list)
        vis = cv2.resize(vis, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        vis = (vis*255).astype(np.uint8).transpose(2, 0, 1)

        return {'render_img': vis}