# -*- coding: utf-8 -*-

import os
import cv2
import math
import torch
import random
import kornia
import numpy as np
import torch.nn as nn
import torchvision as tv
from glob import glob
from cellpose import models
from scipy import ndimage as ndi
from torch.utils.data import Dataset
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import threshold_local


class ImageDataset(Dataset):
    def __init__(self, root, transforms_com, transforms_sep, random_erase=0, random_kernel=True):
        self.transforms_com = transforms_com
        self.transforms_sep = transforms_sep
        self.files = sorted(glob(os.path.join(root, "*.*")))
        self.norm = tv.transforms.Normalize(0.5, 0.5)
        self.random_kernel = random_kernel
        self.random_erase = None
        if random_erase:
            self.random_erase = tv.transforms.Compose([tv.transforms.RandomErasing(p=0.8, scale=[0.01, 0.05],
                ratio=[0.8, 1.2], value=-1, inplace=True) for _ in range(random_erase)])

    def cutmix(self, t1, t2, scale, ratio):
        if random.randint(0, 1) == 1:
            t1, t2 = t2, t1
        s = random.uniform(*scale)
        r = random.uniform(*ratio)
        _, H, W = t1.shape
        w = min(int(math.sqrt(W*H*s/r)), W)
        h = min(int(w*r), H)
        x = random.randrange(0, W-w)
        y = random.randrange(0, H-h)
        t1[:, y:y+h, x:x+w] = t2[:, y:y+h, x:x+w]
        return t1

    def create_kernel(self, hsize, sigma):
        ax = np.linspace(-hsize, hsize, hsize*2+1)
        gauss = np.exp(-0.5 * np.square(ax) / (sigma**2))
        kernel = np.outer(gauss, gauss)
        return kernel

    def add_kernels(self, tensor, num=10, sigma_range=[0.5,3], max_range=[0.2, 1]):
        _, h, w = tensor.shape
        tmp = tensor.squeeze().numpy()
        for i in range(num):
            sigma = random.uniform(*sigma_range)
            maxv = random.uniform(*max_range)
            hsize = int(sigma*3)
            kernel = self.create_kernel(hsize, sigma) * maxv
            cx = random.randint(hsize, h-hsize-1)
            cy = random.randint(hsize, w-hsize-1)
            tmp[cx-hsize:cx+hsize+1, cy-hsize:cy+hsize+1] = np.maximum(kernel.astype(np.uint8), tmp[cx-hsize:cx+hsize+1, cy-hsize:cy+hsize+1])
        tensor = torch.from_numpy(tmp)
        return tensor.unsqueeze(0)

    def __getitem__(self, index):
        image = cv2.imread(self.files[index])
        tensor = self.transforms_com(image[:, :, 1])
        if self.random_kernel:
            item_A = self.cutmix(self.add_kernels(tensor), self.transforms_sep(tensor), (0.2, 0.5), (0.8, 1.2))
            item_B = self.cutmix(self.add_kernels(tensor), self.transforms_sep(tensor), (0.2, 0.5), (0.8, 1.2))
        else:
            item_A = self.cutmix(tensor.clone(), self.transforms_sep(tensor), (0.2, 0.5), (0.8, 1.2))
            item_B = self.cutmix(tensor.clone(), self.transforms_sep(tensor), (0.2, 0.5), (0.8, 1.2))
        if self.random_erase:
            item_A = self.random_erase(item_A)
            item_B = self.random_erase(item_B)
        return {"orig": self.norm(tensor), "A": self.norm(item_A), "B": self.norm(item_B)}

    def __len__(self):
        return len(self.files)


class Normalize(nn.Module):
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0/self.power)
        return x.div(norm + 1e-7)


class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()
        self.register_buffer("real_label", torch.ones(1))
        self.register_buffer("fake_label", torch.zeros(1))

    def __call__(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label.expand_as(prediction)
        else:
            target_tensor = self.fake_label.expand_as(prediction)
        loss = self.loss(prediction, target_tensor)
        return loss


class TopoLoss(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.criterion = nn.BCELoss().to(device)
        self.device = device

    def __call__(self, prediction, output_ref=False):
        src = prediction.detach().cpu().numpy()
        ref = np.zeros(src.shape, dtype=np.float32)
        for i in range(src.shape[0]):
            img = src[i, :, :, :].squeeze()
            thresh = threshold_local(img, 15, offset=-0.02)
            bw = (img < thresh) * 255
            contours, _ = cv2.findContours(bw.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            border = np.zeros(img.shape, dtype=np.uint8) * 255
            cv2.drawContours(border, contours, -1, 255, -1, cv2.LINE_8)
            border = cv2.morphologyEx(border, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
            # border = ndi.binary_opening(border, np.ones((3, 3)), border_value=255).astype(np.uint8) * 255
            border = 255 - border
            contours, _ = cv2.findContours(border.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros(img.shape, dtype=np.uint8)
            cs = [c for c in contours if cv2.contourArea(c) > 10]
            cv2.drawContours(mask, cs, -1, 255, -1, cv2.LINE_8)
            border[mask!=255] = 0
            ref[i, :, :, :] = border / 255
        if output_ref:
            return self.criterion((prediction+1)/2, torch.from_numpy(ref).to(self.device)), torch.from_numpy(ref)
        else:
            return self.criterion((prediction+1)/2, torch.from_numpy(ref).to(self.device))


class ConvBlock(nn.Module):
    """Convolution Block"""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 padding_mode="zeros", bias=False, norm="batch", activation="relu"):
        super().__init__()

        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.Identity

        if activation == "relu":
            activation_layer = nn.ReLU(inplace=True)
        elif activation == "leaky":
            activation_layer = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "hardswish":
            activation_layer = nn.Hardswish(inplace=True)
        else:
            activation_layer = nn.Identity()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                      padding_mode=padding_mode, bias=bias),
            norm_layer(out_ch),
            activation_layer
        )

    def forward(self, x):
        return self.net(x)


class DeconvBlock(nn.Module):
    """Deconvolution Block"""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 output_padding=0, bias=False, norm="batch", activation="relu"):
        super().__init__()

        if norm == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm == "instance":
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.Identity

        if activation == "relu":
            activation_layer = nn.ReLU(inplace=True)
        elif activation == "leaky":
            activation_layer = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "hardswish":
            activation_layer = nn.Hardswish(inplace=True)
        else:
            activation_layer = nn.Identity()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding, bias=bias),
            norm_layer(out_ch),
            activation_layer
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    """Residual Block"""

    def __init__(self, ch, norm="batch"):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(ch, ch, 3, stride=1, padding=1, padding_mode="reflect", norm='batch'),
            ConvBlock(ch, ch, 3, stride=1, padding=1, padding_mode="reflect", norm='batch'),
        )

    def forward(self, x):
        return x + self.net(x)


class ResBlockV2(nn.Module):
    """Residual Block v2"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, norm="batch", activation="relu"):
        super().__init__()

        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.Identity

        if activation == "relu":
            activation_layer = nn.ReLU(inplace=True)
        elif activation == "leaky":
            activation_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            activation_layer = nn.Identity()

        self.net = nn.Sequential(
            norm_layer(in_ch),
            activation_layer,
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, padding_mode="reflect"),
            norm_layer(out_ch),
            activation_layer,
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding, padding_mode="reflect")
        )

        self.shortcut = nn.Identity()
        if stride > 1 or out_ch != in_ch:
            self.shortcut = ConvBlock(in_ch, out_ch, 1, stride=stride, activation=None)

    def forward(self, x):
        return self.shortcut(x) + self.net(x)


class DoubleConvBlock(nn.Module):
    """Two Convolution Layers"""

    def __init__(self, in_ch, out_ch, norm="batch", activation="relu"):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(in_ch, out_ch, 3, stride=1, padding=1, norm=norm,
                      padding_mode="reflect", activation=activation),
            ConvBlock(out_ch, out_ch, 3, stride=1, padding=1, norm=norm,
                      padding_mode="reflect", activation=activation)
        )

    def forward(self, x):
        return self.net(x)


class UpConvBlock(nn.Module):
    """Up-Sampling Layer plus Convolution Layer"""

    def __init__(self, in_ch, out_ch, norm="batch", activation="relu"):
        super().__init__()

        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(in_ch, out_ch, 3, stride=1, padding=1, norm=norm,
                      padding_mode="reflect", activation=activation)
        )

    def forward(self, x):
        return self.net(x)


def init_weights(module):
    class_name = module.__class__.__name__
    if class_name.find("Conv2") != -1 or class_name.find("Linear") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif class_name.find("Norm2d") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


def mix_loss(loss1, loss2, epoch, start, end, start_ratio, end_ratio):
    if epoch < start:
        ratio = start_ratio
    elif epoch > end:
        ratio = end_ratio
    else:
        ratio = ((epoch - start) * end_ratio +
                (end - epoch) * start_ratio) / (end - start)
    return loss1 * ratio + loss2 * (1 - ratio)


class randomGaussianNoise(nn.Module):

    def __init__(self, mean=0, std=1, p=0.5):
        super().__init__()
        self.gaussian_noise = kornia.augmentation.RandomGaussianNoise(mean=mean, std=std, p=p)

    def forward(self, x):
        return self.gaussian_noise(x).squeeze(0)


class randomGaussianBlur(nn.Module):

    def __init__(self, kernel_size=(3, 3), sigma=(0.1, 2), p=0.5):
        super().__init__()
        self.gaussian_blur = kornia.augmentation.RandomGaussianBlur(kernel_size=kernel_size, sigma=sigma, p=p)

    def forward(self, x):
        return self.gaussian_blur(x).squeeze(0)
