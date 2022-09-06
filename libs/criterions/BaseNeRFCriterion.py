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

import torch
import torch.nn as nn


def get_focal_loss(pred, target, alpha=0.25, gamma=2, reduction="mean"):
    label = target.long()
    pt = (1 - pred) * label + pred * (1 - label)
    focal_weight = (alpha * label + (1 - alpha) * (1 - label)) * pt.pow(gamma)
    loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction="none"
        )
        * focal_weight
    )
    if reduction == "mean":
        return loss.mean()
    else:
        return loss.mean()


class Criterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.rgb_loss = lambda x, y: torch.mean((x - y) ** 2)
        self.alpha_loss = get_focal_loss

    def resolve(self, ret, batch):
        mask = batch["mask_at_box"]
        gt_rgb = batch["rgb"]
        if mask.size(1) != ret["rgb_map"].size(1):
            mask = mask[mask].reshape(mask.size(0), ret["rgb_map"].size(1))
        pred_rgb = ret["rgb_map"][mask]
        gt_rgb = gt_rgb[mask]
        pred_alpha = ret["alpha"][mask].sum(-1)
        return pred_rgb, gt_rgb, pred_alpha

    def forward(self, ret, batch, is_train=True):
        scalar_stats = {}
        pred_rgb, gt_rgb, pred_alpha = self.resolve(ret, batch)
        rgb_loss = self.rgb_loss(pred_rgb, gt_rgb)
        scalar_stats["rgb_loss"] = rgb_loss
        return scalar_stats
