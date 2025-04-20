import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import resnet50, resnet101, resnet18, resnet34
from torch import Tensor
from typing import Tuple
import einops

from composer.models import ComposerModel
from typing import Any, Tuple

import sys

sys.path.append("..")

class Model(nn.Module):
    def __init__(
        self,
        projector_dims: list = [8192, 8192, 512],
        bias_last: bool = False,
        bias_proj: bool = False,
        num_classes: int = 1000,
        resnet_size: int = 50,
    ):
        super(Model, self).__init__()

        # insures output of encoder for all datasets is 2048-dimensional
        if resnet_size == 50:
            self.f = resnet50(zero_init_residual=True)
            output_dim = 2048
        elif resnet_size == 101:
            self.f = resnet101(zero_init_residual=True)
            output_dim = 2048
        elif resnet_size == 18: 
            self.f = resnet18(zero_init_residual=True)
            output_dim = 512
        else:
            self.f = resnet34(zero_init_residual=True)
            output_dim =  512
        self.f.fc = nn.Identity()

        # projection head (Following exactly barlow twins offical repo)
        projector_dims = [output_dim] + projector_dims
        layers = []
        for i in range(len(projector_dims) - 2):
            layers.append(
                nn.Linear(projector_dims[i], projector_dims[i + 1], bias=bias_proj)
            )
            layers.append(nn.BatchNorm1d(projector_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(projector_dims[-2], projector_dims[-1], bias=bias_last))
        self.g = nn.Sequential(*layers)

        self.num_classes = num_classes
        if self.num_classes is not None:
            self.lin_cls = nn.Linear(output_dim, num_classes)



    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_ = self.f(x)
        feature = torch.flatten(x_, start_dim=1)
        out = self.g(feature)
        if self.num_classes is not None:
            logits = self.lin_cls(feature.detach()) 
            return feature, out, logits
        else:
            return feature, out



class ComposerWrapper(ComposerModel):
    def __init__(self, module: torch.nn.Module, objective: torch.nn.Module, online_classifier: bool = True):
        super().__init__()

        self.module = module
        self.objective = objective
        self.criterion = nn.CrossEntropyLoss()
        self.do_class = self.module.num_classes is not None
        self.c = 0

    def loss(self, outputs: Any, batch: Any, *args, **kwargs) -> Tensor:
        _, labels = batch
        if self.do_class:
            outputs, logits = outputs[:-2], outputs[-2:]
        loss, loss_dict = self.objective(outputs)
        self.loss_dict = loss_dict

        if self.do_class:
            class_loss = (self.criterion(logits[0], labels) + self.criterion(logits[1], labels)) / 2
            self.loss_dict['class_loss'] = class_loss.item()
            loss = loss + class_loss

        if self.c % 100 == 0:
            print(self.loss_dict)

        self.c += 1
        return loss

    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        if isinstance(batch, Tensor):
            inputs = batch
            if self.do_class:
                features, outputs, logits = self.module(inputs)
            else:
                features, outputs = self.module(inputs)
            return features, outputs
        else:
            inputs, labels = batch
            x1, x2 = inputs
            if self.do_class:
                _, outputs1, logits1 = self.module(x1)
                _, outputs2, logits2 = self.module(x2)
                return outputs1, outputs2, logits1, logits2
            else:
                _, outputs1 = self.module(x1)
                _, outputs2 = self.module(x2)
                return outputs1, outputs2


    def get_backbone(self):
        return self.module
