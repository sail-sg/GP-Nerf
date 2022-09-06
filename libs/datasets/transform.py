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

import cv2
import math
import numpy as np
import random

import PIL
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F

from libs.utils.box_ops import box_xyxy_to_cxcywh


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target

def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


class RandomAffine(object):
    def random_affine(self, image, target=None, pre_params=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                 borderValue=(103.53, 116.28, 123.675)):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        height = img.shape[0]
        width = img.shape[1]

        if pre_params is None:
            border = 0  # width of added border (optional)
            # Rotation and Scale
            R = np.eye(3)
            a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
            # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
            s = random.random() * (scale[1] - scale[0]) + scale[0]
            R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

            # Translation
            T = np.eye(3)
            T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
            T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

            # Shear
            S = np.eye(3)
            S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
            S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

            pre_params = [R, a, T, S]
        else:
            R, a, T, S = pre_params

        M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
        imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                                  borderValue=borderValue)  # BGR order borderValue

        # Return warped points also
        if target is not None:
            boxes = target["boxes"].numpy()
            if len(boxes) > 0:
                n = boxes.shape[0]
                points = boxes.copy()
                area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

                # warp points
                xy = np.ones((n * 4, 3))
                xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = (xy @ M.T)[:, :2].reshape(n, 8)

                # create new boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # apply angle-based reduction
                radians = a * math.pi / 180
                reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
                x = (xy[:, 2] + xy[:, 0]) / 2
                y = (xy[:, 3] + xy[:, 1]) / 2
                w = (xy[:, 2] - xy[:, 0]) * reduction
                h = (xy[:, 3] - xy[:, 1]) * reduction
                xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

                w = xy[:, 2] - xy[:, 0]
                h = xy[:, 3] - xy[:, 1]
                area = w * h
                ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
                i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

                for t in target.keys():
                    if t in ['boxes', 'labels', 'ids']:
                        target[t] = target[t][i]
                target["boxes"] = torch.from_numpy(xy[i].reshape(-1, 4)).float()
                target["valid_idx"] = i
            imw = Image.fromarray(cv2.cvtColor(imw, cv2.COLOR_BGR2RGB))
            return imw, target, pre_params
        else:
            imw = Image.fromarray(cv2.cvtColor(imw, cv2.COLOR_BGR2RGB))
            return imw, target, pre_params

    def __call__(self, img, target=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                 borderValue=(103.53, 116.28, 123.675)):
        pre_params = None
        if isinstance(img, list):
            assert len(img) == len(target)
            out_img = []
            out_target = []
            for im, t in zip(img, target):
                out_im, out_t, pre_params = self.random_affine(im, t, pre_params)
                out_img.append(out_im)
                out_target.append(out_t)
            return out_img, out_target
        else:
            out_img = []
            out_target = []
            out_im, out_t, _ = self.random_affine(img.copy(), target.copy(), pre_params)
            out_img = [img, out_im]
            out_target = [target, out_t]
            return out_img, out_target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            if isinstance(img, list):
                assert len(img) == len(target)
                out_img = []
                out_target = []
                for im, t in zip(img, target):
                    out_im, out_t = hflip(im, t)
                    out_img.append(out_im)
                    out_target.append(out_t)
                return out_img, out_target
            else:
                return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        if isinstance(img, list):
            assert len(img) == len(target)
            out_img = []
            out_target = []
            for im, t in zip(img, target):
                out_im, out_t = resize(im, t, size, self.max_size)
                out_img.append(out_im)
                out_target.append(out_t)
            return out_img, out_target
        else:
            return resize(img, target, size, self.max_size)


class ToTensor(object):
    def __call__(self, img, target):
        if isinstance(img, list):
            out_img = [F.to_tensor(im) for im in img]
        else:
            out_img = F.to_tensor(img)
        return out_img, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        if isinstance(image, list):
            out_image = [F.normalize(im, mean=self.mean, std=self.std) for im in image]
            if target is None:
                return out_image, None
            assert len(out_image) == len(target)
            out_target = []
            for im, t in zip(out_image, target):
                out_t = t.copy()
                h, w = im.shape[-2:]
                if "boxes" in out_t:
                    boxes = out_t["boxes"]
                    wh_boxes = box_xyxy_to_cxcywh(boxes)
                    cx = wh_boxes[..., 0].clamp(min=0, max=w)
                    cy = wh_boxes[..., 1].clamp(min=0, max=h)
                    lw = (cx - boxes[..., 0]).unsqueeze(-1)
                    lh = (cy - boxes[..., 1]).unsqueeze(-1)
                    rw = (boxes[..., 2] - cx).unsqueeze(-1)
                    rb = (boxes[..., 3] - cy).unsqueeze(-1)
                    cx = cx.unsqueeze(-1)
                    cy = cy.unsqueeze(-1)
                    boxes = torch.cat([cx, cy, lw, lh, rw, rb], dim=-1)
                    boxes = boxes / torch.tensor([w, h, w, h, w, h], dtype=torch.float32)
                    out_t["boxes"] = boxes
                    out_target.append(out_t)
            return out_image, out_target
        else:
            image = F.normalize(image, mean=self.mean, std=self.std)
            if target is None:
                return image, None
            target = target.copy()
            h, w = image.shape[-2:]
            if "boxes" in target:
                boxes = target["boxes"]
                wh_boxes = box_xyxy_to_cxcywh(boxes)
                cx = wh_boxes[..., 0].clamp(min=0, max=w)
                cy = wh_boxes[..., 1].clamp(min=0, max=h)
                lw = (cx - boxes[..., 0]).unsqueeze(-1)
                lh = (cy - boxes[..., 1]).unsqueeze(-1)
                rw = (boxes[..., 2] - cx).unsqueeze(-1)
                rb = (boxes[..., 3] - cy).unsqueeze(-1)
                cx = cx.unsqueeze(-1)
                cy = cy.unsqueeze(-1)
                boxes = torch.cat([cx, cy, lw, lh, rw, rb], dim=-1)
                boxes = boxes / torch.tensor([w, h, w, h, w, h], dtype=torch.float32)
                target["boxes"] = boxes
            return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class TrainTransform(object):
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        normalize = Compose([
            ToTensor(),
            Normalize(mean, std)
        ])
        self.augment = Compose([
            normalize,
        ])

    def __call__(self, img, target=None):
        return self.augment(img, target)[0]


class EvalTransform(object):
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        normalize = Compose([
            ToTensor(),
            Normalize(mean, std)
        ])
        self.augment = Compose([
            normalize,
        ])

    def __call__(self, img, target=None):
        return self.augment(img, target)[0]