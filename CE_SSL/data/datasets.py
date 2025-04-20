from zipfile import ZipFile
import random
import torch

import torchvision
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter

from CE_SSL.data.transforms import ImageNetValTransform, Barlow_Transform


class ZipImageNet(torchvision.datasets.ImageNet):
    """
    Loads imagenet files from a zip archive.
    """

    def __init__(self, zip_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zip_path = zip_path
        self.zip_archive = None

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        parts = path.split("/")
        idx = parts.index(self.split) + 1
        _path = "/".join(parts[idx:])
        if self.zip_archive is None:
            self.zip_archive = ZipFile(self.zip_path)
        fh = self.zip_archive.open(_path)
        image = Image.open(fh)
        sample = image.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class Zip_ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, zip_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zip_path = zip_path
        self.zip_archvive = None

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        if self.zip_archvive is None:
            self.zip_archvive = ZipFile(self.zip_path)

        path_split = path.split("/")
        fh = self.zip_archvive.open(
            path_split[-3] + "/" + path_split[-2] + "/" + path_split[-1]
        )

        image = Image.open(fh)
        sample = image.convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            sample = self.target_transform(sample)

        return sample, target


def get_datasets(dataset="imagenet", **kwargs):
    if dataset == "imagenet":
        imagenet_path = "/mnt/ceph/users/tyerxa/datasets/imagenet1k/ILSVRC_2012"
        train_zip_path = '/mnt/home/gkrawezik/ceph/AI_DATASETS/ImageNet/2012/imagenet/train.zip'
        val_zip_path = '/mnt/home/gkrawezik/ceph/AI_DATASETS/ImageNet/2012/imagenet/val.zip'
        train_data = ZipImageNet(
            zip_path=train_zip_path,
            root=imagenet_path,
            split="train",
            transform=Barlow_Transform()
        )
        memory_data = ZipImageNet(
            zip_path=train_zip_path,
            root=imagenet_path,
            split="train",
            transform=ImageNetValTransform(),
        )
        test_data = ZipImageNet(
            zip_path=val_zip_path,
            root=imagenet_path,
            split="val",
            transform=ImageNetValTransform(),
        )
    if dataset == "imagenet_100":
        imagenet_100_path = "/mnt/ceph/users/tyerxa/datasets/imagenet_100/"
        train_data = Zip_ImageFolder(
            zip_path=imagenet_100_path + "train.zip",
            root=imagenet_100_path + "train/",
            transform=Barlow_Transform(),
        )
        memory_data = Zip_ImageFolder(
            zip_path=imagenet_100_path + "train.zip",
            root=imagenet_100_path + "train/",
            transform=ImageNetValTransform(),
        )
        test_data = Zip_ImageFolder(
            zip_path=imagenet_100_path + "val.zip",
            root=imagenet_100_path + "val/",
            transform=ImageNetValTransform(),
        )

    return train_data, memory_data, test_data
