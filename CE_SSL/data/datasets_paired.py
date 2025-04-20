from zipfile import ZipFile
import random
import torch
import numpy as np

import torchvision
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter

from CE_SSL.data.transforms import MatchedTransform, ImageNetValTransform
from CE_SSL.data.datasets import ZipImageNet, Zip_ImageFolder
        
class Zip_ImageNet_Split(torchvision.datasets.ImageNet):
    def __init__(self, zip_path, num_split=2, distributed=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zip_path = zip_path
        self.zip_archvive = None
        self.num_split = num_split
        self.distributed = distributed
        self.first_shuffle = True
        self.split_length = len(self.samples) // self.num_split
        if not self.distributed:
            self.reset_splits()

    def reset_splits(self):
        self.shuffled_indices = list(range(len(self.samples)))
        random.shuffle(self.shuffled_indices)
        self.splits = []
        for i in range(self.num_split):
            self.splits.append(
                self.shuffled_indices[
                    i * self.split_length : (i + 1) * self.split_length
                ]
            )
        # cast to tensor and broadcast to whole group
        if self.distributed:
            shuffled_indices_tensor = torch.tensor(self.shuffled_indices).cuda()
            torch.distributed.broadcast(shuffled_indices_tensor, 0)
            # cast back to list
            self.shuffled_indices = shuffled_indices_tensor.cpu().tolist()

            if self.first_shuffle:
                self.first_shuffle = False
                print("Shuffled indices: ", self.shuffled_indices[:200])

        return None

    def __getitem__(self, index: int):
        imgs, targets = [], []
        for split_index in range(self.num_split):
            img_index = self.splits[split_index][index]
            path, target = self.samples[img_index]
            if self.zip_archvive is None:
                self.zip_archvive = ZipFile(self.zip_path)

            path_split = path.split("/")
            idx = path_split.index(self.split) + 1
            _path = "/".join(path_split[idx:])
            fh = self.zip_archvive.open(_path)

            image = Image.open(fh)
            sample = image.convert("RGB")
            imgs.append(sample)
            targets.append(target)

        t0, t1, p0, p1 = self.transform(image=imgs[0], image0=imgs[1])
        return t0, t1, p0, p1, targets[0], targets[1]

    def __len__(self):
        return self.split_length

class Zip_ImageFolder_Split(torchvision.datasets.ImageFolder):
    def __init__(self, zip_path, num_split=2, distributed=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zip_path = zip_path
        self.zip_archvive = None
        self.num_split = num_split
        self.distributed = distributed
        self.first_shuffle = True
        self.split_length = len(self.samples) // self.num_split
        if not self.distributed:
            self.reset_splits()

    def reset_splits(self):
        self.shuffled_indices = list(range(len(self.samples)))
        random.shuffle(self.shuffled_indices)
        self.splits = []
        for i in range(self.num_split):
            self.splits.append(
                self.shuffled_indices[
                    i * self.split_length : (i + 1) * self.split_length
                ]
            )
        # cast to tensor and broadcast to whole group
        if self.distributed:
            shuffled_indices_tensor = torch.tensor(self.shuffled_indices).cuda()
            torch.distributed.broadcast(shuffled_indices_tensor, 0)
            # cast back to list
            self.shuffled_indices = shuffled_indices_tensor.cpu().tolist()

            if self.first_shuffle:
                self.first_shuffle = False
                print("Shuffled indices: ", self.shuffled_indices[:200])

        return None

    def __getitem__(self, index: int):
        imgs, targets = [], []
        for split_index in range(self.num_split):
            img_index = self.splits[split_index][index]
            path, target = self.samples[img_index]
            if self.zip_archvive is None:
                self.zip_archvive = ZipFile(self.zip_path)

            path_split = path.split("/")
            fh = self.zip_archvive.open(
                path_split[-3] + "/" + path_split[-2] + "/" + path_split[-1]
            )

            image = Image.open(fh)
            sample = image.convert("RGB")
            imgs.append(sample)
            targets.append(target)

        t0, t1, p0, p1 = self.transform(image=imgs[0], image0=imgs[1])
        return t0, t1, p0, p1, targets[0], targets[1]

    def __len__(self):
        return self.split_length


def get_datasets(dataset="imagenet", distributed: bool = True, **kwargs):
    imagenet_path = "/mnt/ceph/users/tyerxa/datasets/imagenet1k/ILSVRC_2012"
    train_zip_path = '/mnt/home/gkrawezik/ceph/AI_DATASETS/ImageNet/2012/imagenet/train.zip'
    val_zip_path = '/mnt/home/gkrawezik/ceph/AI_DATASETS/ImageNet/2012/imagenet/val.zip'
    if dataset == "imagenet":
        train_data = Zip_ImageNet_Split(
            zip_path=train_zip_path,
            root=imagenet_path,
            distributed=distributed,
            split="train",
            transform=MatchedTransform()
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

    else:
        imagenet_100_path = "/mnt/ceph/users/tyerxa/datasets/imagenet_100/"
        train_data = Zip_ImageFolder_Split(
            zip_path=imagenet_100_path + "train.zip",
            root=imagenet_100_path + "train/",
            distributed=distributed,
            transform=MatchedTransform()
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
