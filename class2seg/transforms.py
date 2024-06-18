import random
from typing import Callable

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch import nn
import numpy as np
import ttach as tta
from pytorch_grad_cam.utils import get_2d_projection

# MEAN = [.485,.456,.406]
# STD = [.229, .224, .225]
MEAN = [.3227064591016829, .3227064591016829, .3227064591016829]
STD = [.1874794535405723, .1874794535405723, .1874794535405723]


# https://github.com/pytorch/vision/blob/main/references/segmentation/
def pad_if_smaller(img, size, fill=0):
    min_size = min(img.shape)
    if min_size < size:
        ow, oh = img.shape[0], img.shape[1]
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target=None):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        if target is not None:
            target = F.resize(target.unsqueeze(0), size, interpolation=F.InterpolationMode.NEAREST).squeeze(0)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if target is not None:
                target = F.hflip(target)
        return image, target


class RandomVerticalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            if target is not None:
                target = F.vflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        if target is not None:
            target = pad_if_smaller(target, self.size, fill=0)
            target = F.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor:
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        if target is not None:
            target = torch.as_tensor(target, dtype=torch.int64)
        return image, target


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x, y):
        if random.random() > self.p:
            return x, y
        return self.fn(x, y)


class RandomRot90(nn.Module):
    def __init__(self, dims=[-2, -1]):
        super().__init__()
        self.dims = dims

    def forward(self, image, target):
        rot = torch.randint(high=4, size=(1,))
        if target is not None:
            target = torch.rot90(target, int(rot), self.dims)
        return torch.rot90(image, int(rot), self.dims), target


class PixelNoise(nn.Module):
    """
    for each pixel, same across all bands
    """

    def __init__(self, std_noise=0.1):
        super().__init__()
        self.std_noise = std_noise

    def forward(self, x, y):
        C, H, W = x.shape
        noise_level = x.std() * self.std_noise
        pixel_noise = torch.rand(H, W, device=x.device)
        return x + pixel_noise.view(1, H, W) * noise_level, y


class ChannelNoise(nn.Module):
    """
    for each channel
    """

    def __init__(self, std_noise=0.1):
        super().__init__()
        self.std_noise = std_noise

    def forward(self, x, y):
        C, H, W = x.shape
        noise_level = x.std() * self.std_noise

        channel_noise = torch.rand(C, device=x.device)
        return x + channel_noise.view(-1, 1, 1).to(x.device) * noise_level, y


class Noise(nn.Module):
    """
    for each channel
    """

    def __init__(self, std_noise=0.1):
        super().__init__()
        self.std_noise = std_noise

    def forward(self, x, y):
        noise_level = x.std() * self.std_noise
        noise = torch.rand(x.shape[0], x.shape[1], x.shape[2], device=x.device)
        return x + noise * noise_level, y


class AddInverse(nn.Module):

    def __init__(self, dim=0):
        """
            
        """
        super().__init__()
        self.dim = dim

    def forward(self, in_tensor, target):
        out = torch.cat([in_tensor, 1-in_tensor], self.dim)
        return out, target


class Normalize(nn.Module):

    def __init__(self, mean, std):
        """
            Adds (1-in_tensor) as additional channels to its input via torch.cat().
            Can be used for images to give all spatial locations the same sum over the channels to reduce color bias.
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, in_tensor, target=None):
        out = F.normalize(in_tensor, self.mean, self.std)
        return out, target


def get_train_transform(crop_size=64, size=256, add_inverse=False, mean=MEAN, std=STD):
    transforms = [
        ToTensor(),
        # RandomApply(
        #    T.ColorJitter(0.8, 0.8, 0.8, 0.2),
        #    p = 0.3
        # ),
        # T.RandomGrayscale(p=0.2),
        # RandomRot90(),
        # T.RandomRotation(90),
        # RandomResizedCrop(crop_size, interpolation=T.InterpolationMode.BILINEAR),
        # RandomCrop(crop_size),
        RandomResize(size),
        RandomHorizontalFlip(flip_prob=0.5),
        RandomVerticalFlip(flip_prob=0.5),
        # RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.5),
        # RandomApply(PixelNoise(std_noise=0.75), p=0.5),
        # RandomApply(ChannelNoise(std_noise=0.25), p=0.5),
        # RandomApply(Noise(std_noise=0.25), p=0.5),
        # TopoPaperAugments(),
        # Normalize(mean, std)
    ]
    if add_inverse:
        transforms.append(AddInverse())
    return Compose(transforms)


def get_val_transform(crop_size=64, size=256, add_inverse=False, mean=MEAN, std=STD):
    transforms = [
        ToTensor(),
        RandomResize(size),
        # Normalize(mean, std)
    ]
    if add_inverse:
        transforms.append(AddInverse())
    return Compose(transforms)


def augmentation_smoothing(
    map_func: Callable,
    input_tensor: torch.Tensor,
    #    targets: List[torch.nn.Module],
    #    eigen_smooth: bool = False
    ) -> np.ndarray:
        transforms = tta.Compose(
            [
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            # cam = self.forward(augmented_tensor,
            #                    targets,
            #                    eigen_smooth)
            cam = map_func(image=augmented_tensor)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam


def augmentation_intersection(
    map_func: Callable,
    input_tensor: torch.Tensor,
    #    targets: List[torch.nn.Module],
    #    eigen_smooth: bool = False
    ) -> np.ndarray:
        transforms = tta.Compose(
            [
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            # cam = self.forward(augmented_tensor,
            #                    targets,
            #                    eigen_smooth)
            cam = map_func(image=augmented_tensor)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        # Min-max scaling to be between 0 and 1
        cams = np.float32(cams)
        cams = (cams - cams.min()) / (cams.max() - cams.min() + 1e-12)

        cam = np.prod(np.float32(cams), axis=0)
        return cam


def eigen_smooth(map):
    return get_2d_projection(map)


class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
