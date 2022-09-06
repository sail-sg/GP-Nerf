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
import pprint
import random
from importlib import import_module as impm

import _init_paths
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from configs import cfg, update_config
from libs.datasets.samplers import build_batchsampler
from libs.utils import misc
from libs.utils.lr_scheduler import ExponentialLR
from libs.utils.utils import create_logger, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Digital Human NeRF")
    parser.add_argument(
        "--cfg",
        dest="yaml_file",
        help="experiment configure file name, e.g. configs/base_config.yaml",
        required=True,
        type=str,
    )
    # default distributed training
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="if use distribute train",
    )

    parser.add_argument(
        "--dist-url",
        dest="dist_url",
        default="tcp://10.5.38.36:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--world-size",
        dest="world_size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank",
        default=0,
        type=int,
        help="node rank for distributed training, machine level",
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    return args


def get_ip(ip_addr):
    ip_list = ip_addr.split("-")[2:6]
    for i in range(4):
        if ip_list[i][0] == "[":
            ip_list[i] = ip_list[i][1:].split(",")[0]
    # TODO random ip
    return f"tcp://{ip_list[0]}.{ip_list[1]}.{ip_list[2]}.{ip_list[3]}:23456"


def main_per_worker():
    args = parse_args()

    update_config(cfg, args)
    ngpus_per_node = torch.cuda.device_count()

    if "SLURM_PROCID" in os.environ.keys():
        proc_rank = int(os.environ["SLURM_PROCID"])
        local_rank = proc_rank % ngpus_per_node
        args.world_size = int(os.environ["SLURM_NTASKS"])
    else:
        proc_rank = 0
        local_rank = 0
        args.world_size = 1

    args.distributed = args.world_size > 1 or args.distributed

    # create logger
    if proc_rank == 0:
        logger, output_dir = create_logger(cfg, proc_rank)

    # distribution
    if args.distributed:
        dist_url = get_ip(os.environ["SLURM_STEP_NODELIST"])
        if proc_rank == 0:
            logger.info(
                f"Init process group: dist_url: {dist_url},  "
                f"world_size: {args.world_size}, "
                f"proc_rank: {proc_rank}, "
                f"local_rank:{local_rank}"
            )
        dist.init_process_group(
            backend=cfg.dist_backend,
            init_method=dist_url,
            world_size=args.world_size,
            rank=proc_rank,
        )
        torch.distributed.barrier()
        # torch seed
        seed = cfg.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.cuda.set_device(local_rank)
        device = torch.device(cfg.device)
        # TODO build render
        model = getattr(impm(cfg.render.file), "build_render")(cfg)
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        batch_size = cfg.dataset.img_num_per_gpu
    else:
        assert proc_rank == 0, (
            "proc_rank != 0, it will influence " "the evaluation procedure"
        )
        # torch seed
        seed = cfg.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if cfg.device == "cuda":
            torch.cuda.set_device(local_rank)
        device = torch.device(cfg.device)
        model = getattr(impm(cfg.render.file), "build_render")(cfg)
        model = torch.nn.DataParallel(model).to(device)
        if ngpus_per_node == 0:
            batch_size = cfg.dataset.img_num_per_gpu
        else:
            batch_size = cfg.dataset.img_num_per_gpu * ngpus_per_node

    train_dataset = getattr(impm(cfg.dataset.train.file), "build_dataset")(
        cfg, is_train=True
    )
    eval_dataset = getattr(impm(cfg.dataset.test.file), "build_dataset")(
        cfg, is_train=False
    )

    train_sampler = build_batchsampler(
        cfg, train_dataset, args.distributed, batch_size, is_train=True
    )
    eval_sampler = build_batchsampler(
        cfg, eval_dataset, args.distributed, batch_size, is_train=False
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        drop_last=cfg.dataset.train.drop_last,
        num_workers=cfg.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=(eval_sampler is None),
        drop_last=cfg.dataset.test.drop_last,
        num_workers=cfg.workers,
        sampler=eval_sampler,
    )

    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)
    param_dicts = [
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters() if p.requires_grad
            ]
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    lr_scheduler = ExponentialLR(
        optimizer, decay_epochs=cfg.train.decay_epochs, gamma=cfg.train.gamma
    )
    model, optimizer, lr_scheduler, last_iter = load_checkpoint(
        cfg, model, optimizer, lr_scheduler, device
    )

    criterion = getattr(impm(cfg.train.criterion_file), "Criterion")(cfg)

    # build trainer
    Trainer = getattr(impm(cfg.train.file), "Trainer")(
        cfg,
        model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        logger=logger,
        log_dir=cfg.log_dir,
        performance_indicator=cfg.pi,
        last_iter=last_iter,
        rank=proc_rank,
        device=device,
    )

    print("start training...")
    while True:
        Trainer.train(train_loader, eval_loader)


if __name__ == "__main__":
    main_per_worker()
