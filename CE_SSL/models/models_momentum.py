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

class MomentumModel(nn.Module):
    def __init__(
        self,
        projector_dims: list = [8192, 8192, 512],
        bias_last: bool = False,
        bias_proj: bool = False,
        num_classes: int = 1000,
        resnet_size: int = 50
    ):
        super(MomentumModel, self).__init__()
        # insures output of encoder for all datasets is 2048-dimensional
        if resnet_size == 50:
            self.f = resnet50(zero_init_residual=True)
            self.mom_f = resnet50(zero_init_residual=True)
            output_dim = 2048
        elif resnet_size == 101:
            self.f = resnet101(zero_init_residual=True)
            self.mom_f = resnet101(zero_init_residual=True)
            output_dim = 2048
        elif resnet_size == 18: 
            self.f = resnet18(zero_init_residual=True)
            self.mom_f = resnet18(zero_init_residual=True)
            output_dim = 512
        else:
            self.f = resnet34(zero_init_residual=True)
            self.mom_f = resnet34(zero_init_residual=True)
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

        # initialize momentum background and projector
        self.mom_f.fc = nn.Identity()

        # projection head (Following exactly barlow twins offical repo)
        layers = []
        for i in range(len(projector_dims) - 2):
            layers.append(
                nn.Linear(projector_dims[i], projector_dims[i + 1], bias=bias_proj)
            )
            layers.append(nn.BatchNorm1d(projector_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(projector_dims[-2], projector_dims[-1], bias=bias_last))
        self.mom_g = nn.Sequential(*layers)

        params_f_online, params_f_mom = self.f.parameters(), self.mom_f.parameters()
        params_g_online, params_g_mom = self.g.parameters(), self.mom_g.parameters()

        for po, pm in zip(params_f_online, params_f_mom):
            pm.data.copy_(po.data)
            pm.requires_grad = False

        for po, pm in zip(params_g_online, params_g_mom):
            pm.data.copy_(po.data)
            pm.requires_grad = False

        self.num_classes = num_classes
        if num_classes is not None:
            self.lin_cls = nn.Linear(output_dim, num_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_ = self.f(x)
        feature = torch.flatten(x_, start_dim=1)
        out = self.g(feature)
        logits = self.lin_cls(feature.detach())

        x_momentum = self.mom_f(x)
        feature_momentum = torch.flatten(x_momentum, start_dim=1)
        out_momentum = self.mom_g(feature_momentum)

        return feature, out, feature_momentum, out_momentum, logits


class MomentumComposerWrapper(ComposerModel):
    def __init__(self, module: torch.nn.Module, objective):
        super().__init__()

        self.module = module
        self.objective = objective
        self.c = 0  # counts the number of forward calls
        self.criterion = nn.CrossEntropyLoss()

    def loss(self, outputs: Any, batch: Any, *args, **kwargs) -> Tensor:
        _, labels = batch
        outputs, logits = outputs[:-2], outputs[-2:]
        loss, loss_dict = self.objective(outputs)
        self.loss_dict = loss_dict
        class_loss = (self.criterion(logits[0], labels) + self.criterion(logits[1], labels)) / 2
        self.loss_dict['class_loss'] = class_loss.item()
        if self.c % 100 == 0:
            print(self.loss_dict)

        self.c += 1
        return loss + class_loss

    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        if isinstance(batch, Tensor):
            inputs = batch
            features, outputs, features_momentum, outputs_momentum = self.module(inputs)
            return features, outputs
        else:
            inputs, _ = batch
            x1, x2 = inputs
            _, out1, _, out1_momentum, logits1 = self.module(x1)
            _, out2, _, out2_momentum, logits2 = self.module(x2)

            return out1, out2, out1_momentum, out2_momentum, logits1, logits2

    def get_backbone(self):
        return self.module
