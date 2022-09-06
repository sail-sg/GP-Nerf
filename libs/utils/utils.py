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

import importlib
import logging
import os
import os.path as osp
import time
from bisect import bisect_right
from functools import partial

import numpy as np
import torch
import torch.optim as optim
from six.moves import map, zip
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import ConcatDataset


def resource_path(relative_path):
    """To get the absolute path"""
    base_path = osp.abspath(".")

    return osp.join(base_path, relative_path)


def ensure_dir(root_dir, rank=0):
    if not osp.exists(root_dir) and rank == 0:
        print(f"=> creating {root_dir}")
        os.mkdir(root_dir)
    else:
        while not osp.exists(root_dir):
            print(f"=> wait for {root_dir} created")
            time.sleep(10)

    return root_dir


def create_logger(cfg, rank=0):
    # working_dir root
    abs_working_dir = resource_path("work_dirs")
    working_dir = ensure_dir(abs_working_dir, rank)
    # output_dir root
    output_root_dir = ensure_dir(os.path.join(working_dir, cfg.output_dir), rank)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    final_output_dir = ensure_dir(os.path.join(output_root_dir, time_str), rank)
    # set up logger
    logger = setup_logger(final_output_dir, time_str, rank)

    return logger, final_output_dir


def setup_logger(final_output_dir, time_str, rank, phase="train"):
    log_file = f"{phase}_{time_str}_rank{rank}.log"
    final_log_file = os.path.join(final_output_dir, log_file)
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    return logger


def load_checkpoint(cfg, model, optimizer, lr_scheduler, device, module_name="model"):
    last_iter = -1
    resume_path = cfg.render.resume_path
    resume = cfg.train.resume
    if resume_path and resume:
        if osp.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location="cpu")
            # resume
            if "state_dict" in checkpoint:
                model.module.load_state_dict(checkpoint["state_dict"], strict=False)
                logging.info(f"==> model pretrained from {resume_path} \n")
            elif "model" in checkpoint:
                if module_name == "detr":
                    model.module.detr_head.load_state_dict(
                        checkpoint["model"], strict=False
                    )
                    logging.info(f"==> detr pretrained from {resume_path} \n")
                else:
                    model.module.load_state_dict(checkpoint["model"], strict=False)
                    logging.info(f"==> model pretrained from {resume_path} \n")
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                logging.info(f"==> optimizer resumed, continue training")
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
            if (
                "optimizer" in checkpoint
                and "lr_scheduler" in checkpoint
                and "epoch" in checkpoint
            ):
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                last_iter = checkpoint["epoch"]
                logging.info(f"==> last_epoch = {last_iter}")
            if "epoch" in checkpoint:
                last_iter = checkpoint["epoch"]
                logging.info(f"==> last_epoch = {last_iter}")
            # pre-train
        else:
            logging.error(f'==> checkpoint do not exists: "{resume_path}"')
            raise FileNotFoundError
    else:
        logging.info("==> train model without resume")

    return model, optimizer, lr_scheduler, last_iter


class WarmupMultiStepLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def save_checkpoint(states, is_best, output_dir, filename="checkpoint.pth"):
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(states, os.path.join(output_dir, filename))
    logging.info(f"save model to {output_dir}")
    if is_best:
        torch.save(states["state_dict"], os.path.join(output_dir, "model_best.pth"))


def load_eval_model(resume_path, model):
    if resume_path != "":
        if osp.exists(resume_path):
            print(f"==> model load from {resume_path}")
            checkpoint = torch.load(resume_path)
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f'==> checkpoint do not exists: "{resume_path}"')
            raise FileNotFoundError
    return model


def write_dict_to_json(mydict, f_path):
    import json

    import numpy

    class DateEnconding(json.JSONEncoder):
        def default(self, obj):
            if isinstance(
                obj,
                (
                    numpy.int_,
                    numpy.intc,
                    numpy.intp,
                    numpy.int8,
                    numpy.int16,
                    numpy.int32,
                    numpy.int64,
                    numpy.uint8,
                    numpy.uint16,
                    numpy.uint32,
                    numpy.uint64,
                ),
            ):
                return int(obj)
            elif isinstance(
                obj, (numpy.float_, numpy.float16, numpy.float32, numpy.float64)
            ):
                return float(obj)
            elif isinstance(obj, (numpy.ndarray,)):  # add this line
                return obj.tolist()  # add this line
            return json.JSONEncoder.default(self, obj)

    with open(f_path, "w") as f:
        json.dump(mydict, f, cls=DateEnconding)
        print("write down det dict to %s!" % (f_path))
