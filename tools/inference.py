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

from __future__ import division, print_function, with_statement

import argparse
import os
from importlib import import_module as impm

import _init_paths
import torch

from configs import cfg, update_config
from libs.datasets.samplers import build_batchsampler


def parse_args():
    parser = argparse.ArgumentParser(description="HOI Detection Task")
    parser.add_argument(
        "--cfg",
        dest="yaml_file",
        help="experiment configure file name, e.g. configs/hico.yaml",
        required=True,
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    return args


def main_per_worker():
    args = parse_args()
    update_config(cfg, args)
    ngpus_per_node = torch.cuda.device_count()
    device = torch.device(cfg.device)

    result_dir = cfg.result_dir
    test_seq = cfg.test.test_seq
    result_path = f"{result_dir}/{test_seq}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # model
    model = getattr(impm(cfg.render.file), "build_render")(cfg)
    model = torch.nn.DataParallel(model).to(device)
    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # load model checkpoints
    resume_path = cfg.render.resume_path
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location="cpu")
        # resume
        if "state_dict" in checkpoint:
            model.module.load_state_dict(checkpoint["state_dict"], strict=True)
            print(f"==> model pretrained from {resume_path}")

    # get datset
    eval_dataset = getattr(impm(cfg.dataset.test.file), "build_dataset")(
        cfg, is_train=False
    )
    eval_sampler = build_batchsampler(cfg, eval_dataset, False, 1, is_train=False)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        sampler=eval_sampler,
        num_workers=cfg.workers,
    )
    # start evaluate in Trainer
    Trainer = getattr(impm(cfg.train.file), "Trainer")(
        cfg,
        model,
        criterion=None,
        optimizer=None,
        lr_scheduler=None,
        logger=None,
        log_dir=None,
        performance_indicator=cfg.pi,
        last_iter=None,
        rank=0,
        device=device,
    )
    print(f"==> start eval...")

    Trainer.evaluate(eval_loader, result_path, is_vis=cfg.test.is_vis)


if __name__ == "__main__":
    main_per_worker()
