import torch
from torchvision.transforms import RandomResizedCrop, ToTensor
import numpy as np
from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import ImageOps, ImageFilter
from PIL import Image
import random

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import _interpolation_modes_from_int
import torchvision.transforms.functional as FT
from torchvision.transforms import ColorJitter, RandomResizedCrop, RandomRotation



### TRANSFORMATIONS FOR VANILLA TRAINING PIPELINE ###
class ImageNetValTransform:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, x):
        return self.transform(x)


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class GaussianBlur(object):
    def __init__(self, p, min_sigma=0.1, max_sigma=2.0):
        self.p = p
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * (self.max_sigma - self.min_sigma) + self.min_sigma
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Barlow_Transform:
    def __init__(self):
        spatial_transform = torch.RandomResizedCrop(224, interpolation=Image.BICUBIC)

        self.transform = transforms.Compose(
            [
                spatial_transform,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)

        return y1, y2

### TRANSFORMATIONS FOR PAIRED TRAINING PIPELINE ###
class RandomApplyGaussianBlur(object):
    def __init__(self, p: float, return_param: bool=True, min_sigma: float=0.1, max_sigma: float=2.0):
        super().__init__()
        self.p = p
        self.return_param = return_param
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img):
        if np.random.rand() > self.p:
            if self.return_param:
                return img, 0.0, 0.0
            return img

        #sigma = np.random.rand() * 1.9 + 0.1
        sigma = np.random.rand() * (self.max_sigma - self.min_sigma) + self.min_sigma
        #sigma = np.random.rand() * 2.0
        if self.return_param:
            return img.filter(ImageFilter.GaussianBlur(sigma)), 1.0, sigma
  
        return img.filter(ImageFilter.GaussianBlur(sigma))

class RandomApplyGrayscale(object):
    def __init__(self, p: float, return_param: bool=True):
        super().__init__()
        self.p = p
        self.return_param = return_param

    def __call__(self, img):
        if np.random.rand() < self.p:
            g_img = FT.to_grayscale(img, num_output_channels=3)
            if self.return_param:
                return g_img, 1.0
            return g_img

        if self.return_param:
            return img, 0.0
        return img


class RandomApplySolarization(object):
    def __init__(self, p: float, return_param: bool=True):
        self.p = p
        self.return_param = return_param

    def __call__(self, img):
        if np.random.rand() > self.p:
            if self.return_param:
                return img, 0.0
            return img

        if self.return_param:
            return ImageOps.solarize(img), 1.0
        return ImageOps.solarize(img)
 
class RandomApplyHorizontalFlip(object):
    def __init__(self, p: float, return_param: bool=True):
        self.p = p
        self.return_param = return_param
    
    def __call__(self, img):
        if np.random.rand() > self.p:
            if self.return_param:
                return img, 0.0
            return img

        if self.return_param:
            return FT.hflip(img), 1.0
        return FT.hflip(img)


class RandomResizedCropWithParams(RandomResizedCrop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.antialias = None

    def forward(self, img):
        H, W = FT.get_image_size(img)
        i, j, h, w = super().get_params(img, self.scale, self.ratio)

        return FT.resized_crop(img, i, j, h, w, self.size, self.interpolation), (i, j, h, w)

    
class ColorJiterWithParams(ColorJitter):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p

    def forward(self, img):
        if np.random.rand() < self.p:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )

            
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = FT.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = FT.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = FT.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = FT.adjust_hue(img, hue_factor)

            return img, (1.0, brightness_factor, contrast_factor, saturation_factor, hue_factor, fn_idx[0],fn_idx[1],fn_idx[2],fn_idx[3])
        return img, (0.0, 1.0, 1.0, 1.0, 0.0, 0., 1., 2., 3.)

class MatchedTransform(object):
    def __init__(
        self,
    ):
        super().__init__()
        self.min_crop_scale = 0.08
        self.min_sigma, self.max_sigma = 0.1, 2.0
        self.contrast_factor = 0.4
        self.brightness_factor = 0.4
        self.saturation_factor = 0.2
        self.hue_factor = 0.1

        self.RRC = RandomResizedCropWithParams(scale=(self.min_crop_scale, 1.0), size=224, interpolation=Image.BICUBIC)
        self.GB1 = RandomApplyGaussianBlur(1.0, min_sigma=self.min_sigma, max_sigma=self.max_sigma)
        self.GB2 = RandomApplyGaussianBlur(0.2, min_sigma=self.min_sigma, max_sigma=self.max_sigma)
        self.SOL1 = RandomApplySolarization(p=0.0)
        self.SOL2 = RandomApplySolarization(p=0.2)
        self.CJ = ColorJiterWithParams(
            p=0.8,
            brightness=self.brightness_factor,
            contrast=self.contrast_factor,
            saturation=self.saturation_factor,
            hue=self.hue_factor
        )
        self.HF = RandomApplyHorizontalFlip(p=0.5)
        self.GS = RandomApplyGrayscale(p=0.2)
        self.ToTensor = transforms.ToTensor()
        self.Normalize = transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )

    def do_transform(self, image, image0):
        W1, H1 = FT.get_image_size(image)

        # resized crop, with jittered jitters
        crop_1, (i_1, j_1, h_1, w_1) = self.RRC(image) 
        if i_1 + h_1 > H1:
            print('wut height 1')
            print(H1, W1)
            print(i_1, j_1, h_1, w_1)
        if j_1 + w_1 > W1:
            print('wut width 1')
            print(H1, W1)
            print(i_1, j_1, h_1, w_1)
        crop_1, hf_1 = self.HF(crop_1)
        crop_1, (_, bri_1, con_1, sat_1, hue_1, a_1, b_1, c_1, d_1) = self.CJ(crop_1)
        crop_1, gs_1 = self.GS(crop_1)
        crop_1, _, sigma_1 = self.GB1(crop_1)
        crop_1, sol_1 = self.SOL1(crop_1)
        crop_1 = self.ToTensor(crop_1)
        crop_1 = self.Normalize(crop_1)

        W2, H2 = FT.get_image_size(image0)

        # take crop in the same relative location (images have different resolutions)
        i_2, j_2 = int(i_1 * H2 / H1), int(j_1 * W2 / W1)
        h_2, w_2 = int(h_1 * H2 / H1), int(w_1 * W2 / W1)
        if i_2 > H2 or i_2 + h_2 > H2 or i_2 < 0:
            print('wut height 2')
            print(H1, W1)
            print(H2, W2)
            print(i_1, j_1, h_1, w_1)
            print(i_2, j_2, h_2, w_2)
        if j_2 > W2 or j_2 + w_2 > W2 or j_2 < 0:
            print('wut width 2')
            print(H1, W1)
            print(H2, W2)
            print(i_1, j_1, h_1, w_1)
            print(i_2, j_2, h_2, w_2)
        crop_2 = FT.resized_crop(image0, i_2, j_2, h_2, w_2, (224, 224), Image.BICUBIC)

        if hf_1 == 1.0:
            crop_2 = FT.hflip(crop_2)

        for fn_idx in [a_1, b_1, c_1, d_1]:
            if fn_idx == 0 and bri_1 is not None:
                crop_2 = FT.adjust_brightness(crop_2, bri_1)
            elif fn_idx == 1 and con_1 is not None:
                crop_2 = FT.adjust_contrast(crop_2, con_1)
            elif fn_idx == 2 and sat_1 is not None:
                crop_2 = FT.adjust_saturation(crop_2, sat_1)
            elif fn_idx == 3 and hue_1 is not None:
                crop_2 = FT.adjust_hue(crop_2, hue_1)

        if gs_1 == 1.0:
            crop_2 = FT.to_grayscale(crop_2, num_output_channels=3)

        if sigma_1 > 0.0:
            crop_2 = crop_2.filter(ImageFilter.GaussianBlur(sigma_1))

        if sol_1 == 1.0:
            crop_2 = ImageOps.solarize(crop_2) 
        crop_2 = self.ToTensor(crop_2)
        crop_2 = self.Normalize(crop_2)

        return crop_1, crop_2
            
    def do_transform_prime(self, image, image0):
        W1, H1 = FT.get_image_size(image)

        crop_1, (i_1, j_1, h_1, w_1) = self.RRC(image) 
        if i_1 + h_1 > H1:
            print('wut height 1 prime')
            print(H1, W1)
            print(i_1, j_1, h_1, w_1)
        if j_1 + w_1 > W1:
            print('wut width 1 prime')
            print(H1, W1)
            print(i_1, j_1, h_1, w_1)

        crop_1, hf_1 = self.HF(crop_1)
        crop_1, (_, bri_1, con_1, sat_1, hue_1, a_1, b_1, c_1, d_1) = self.CJ(crop_1)
        crop_1, gs_1 = self.GS(crop_1)
        crop_1, _, sigma_1 = self.GB2(crop_1)
        crop_1, sol_1 = self.SOL2(crop_1)
        crop_1 = self.ToTensor(crop_1)
        crop_1 = self.Normalize(crop_1)

        W2, H2 = FT.get_image_size(image0)

        # take crop in the same relative location (images have different resolutions)
        i_2, j_2 = int(i_1 * H2 / H1), int(j_1 * W2 / W1)
        h_2, w_2 = int(h_1 * H2 / H1), int(w_1 * W2 / W1)

        if i_2 > H2 or i_2 + h_2 > H2 or i_2 < 0:
            print('wut height 2 2 prime')
            print(H1, W1)
            print(H2, W2)
            print(i_1, j_1, h_1, w_1)
            print(i_2, j_2, h_2, w_2)
        if j_2 > W2 or j_2 + w_2 > W2 or j_2 < 0:
            print('wut width 2 primne')
            print(H1, W1)
            print(H2, W2)
            print(i_1, j_1, h_1, w_1)
            print(i_2, j_2, h_2, w_2)

        crop_2 = FT.resized_crop(image0, i_2, j_2, h_2, w_2, size=(224, 224), interpolation=Image.BICUBIC)

        if hf_1 == 1.0:
            crop_2 = FT.hflip(crop_2)

        for fn_idx in [a_1, b_1, c_1, d_1]:
            if fn_idx == 0 and bri_1 is not None:
                crop_2 = FT.adjust_brightness(crop_2, bri_1)
            elif fn_idx == 1 and con_1 is not None:
                crop_2 = FT.adjust_contrast(crop_2, con_1)
            elif fn_idx == 2 and sat_1 is not None:
                crop_2 = FT.adjust_saturation(crop_2, sat_1)
            elif fn_idx == 3 and hue_1 is not None:
                crop_2 = FT.adjust_hue(crop_2, hue_1)

        if gs_1 == 1.0:
            crop_2 = FT.to_grayscale(crop_2, num_output_channels=3)

        if sigma_1 > 0.0:
            crop_2 = crop_2.filter(ImageFilter.GaussianBlur(sigma_1))

        if sol_1 == 1.0:
            crop_2 = ImageOps.solarize(crop_2) 
        crop_2 = self.ToTensor(crop_2)
        crop_2 = self.Normalize(crop_2)

        return crop_1, crop_2

    def __call__(self, image, image0):
        t0, t1 = self.do_transform(image, image0)
        p0, p1 = self.do_transform_prime(image, image0)

        return t0, t1, p0, p1