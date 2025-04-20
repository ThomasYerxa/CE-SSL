import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import resnet50, resnet18, resnet101, resnet34
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
        projector_dims_1: list = [8192, 8192, 512],
        projector_dims_2: list = [8192, 8192, 512],
        bias_last: bool = False,
        bias_proj: bool = False,
        num_classes: int = 1000,
        num_equi_dims: int = 0,
        do_split: bool = False,
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
        self.num_equi_dims = num_equi_dims
        self.do_split = do_split

        if self.do_split:
            self.build_split_projectors(projector_dims_1, projector_dims_2, num_equi_dims, bias_last, bias_proj, output_dim=output_dim)
        else:
            self.build_unsplit_projectors(projector_dims_1, projector_dims_2, bias_last, bias_proj, output_dim=output_dim)

        self.num_classes = num_classes
        if num_classes is not None:
            self.lin_cls = nn.Linear(output_dim, num_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.do_split:
            return self.split_forward(x)
        else:
            return self.unsplit_forward(x)

    def unsplit_forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_ = self.f(x)
        feature = torch.flatten(x_, start_dim=1)
        out_inv = self.g_1(feature)
        out_equi = self.g_2(feature)
        if self.num_classes is not None:
            logits = self.lin_cls(feature.detach())
        else:
            logits = None

        return feature, out_inv, out_equi, logits

    def split_forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_ = self.f(x)
        feature = torch.flatten(x_, start_dim=1)
        feature_equi = feature[:, :self.num_equi_dims]
        feature_inv = feature[:, self.num_equi_dims:]
        out_inv = self.g_1(feature_inv)
        out_equi = self.g_2(feature_equi)
        if self.num_classes is not None:
            logits = self.lin_cls(feature.detach())
        else:
            logits = None

        return feature, out_inv, out_equi, logits
    

    def build_unsplit_projectors(self, projector_dims_1, projector_dims_2, bias_last, bias_proj, output_dim=2048):
        projector_dims = [output_dim] + projector_dims_1
        layers = [nn.Identity()]
        if len(projector_dims) > 1:
            for i in range(len(projector_dims) - 2):
                layers.append(
                    nn.Linear(projector_dims[i], projector_dims[i + 1], bias=bias_proj)
                )
                layers.append(nn.BatchNorm1d(projector_dims[i + 1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(projector_dims[-2], projector_dims[-1], bias=bias_last))
        self.g_1 = nn.Sequential(*layers)

        projector_dims = [output_dim] + projector_dims_2
        layers = [nn.Identity()]
        if len(projector_dims) > 1:
            for i in range(len(projector_dims) - 2):
                layers.append(
                    nn.Linear(projector_dims[i], projector_dims[i + 1], bias=bias_proj)
                )
                layers.append(nn.BatchNorm1d(projector_dims[i + 1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(projector_dims[-2], projector_dims[-1], bias=bias_last))
        self.g_2 = nn.Sequential(*layers)

    def build_split_projectors(self, projector_dims_1, projector_dims_2, num_equi_dims, bias_last, bias_proj, output_dim=2048):
        # projection head (Following exactly barlow twins offical repo)
        projector_dims = [output_dim - num_equi_dims] + projector_dims_1
        layers = [nn.Identity()]
        if len(projector_dims) > 1:
            for i in range(len(projector_dims) - 2):
                layers.append(
                    nn.Linear(projector_dims[i], projector_dims[i + 1], bias=bias_proj)
                )
                layers.append(nn.BatchNorm1d(projector_dims[i + 1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(projector_dims[-2], projector_dims[-1], bias=bias_last))
        self.g_1 = nn.Sequential(*layers)

        if self.num_equi_dims > 0:
            projector_dims = [num_equi_dims] + projector_dims_2
            layers = [nn.Identity()]
            if len(projector_dims) > 1:
                for i in range(len(projector_dims) - 2):
                    layers.append(
                        nn.Linear(projector_dims[i], projector_dims[i + 1], bias=bias_proj)
                    )
                    layers.append(nn.BatchNorm1d(projector_dims[i + 1]))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(projector_dims[-2], projector_dims[-1], bias=bias_last))
            self.g_2 = nn.Sequential(*layers)
        else:
            self.g_2 = nn.Identity()


class ComposerWrapper(ComposerModel):
    def __init__(self, module: torch.nn.Module, objective, pure_eval=False):
        super().__init__()

        self.module = module
        self.objective = objective
        self.criterion = nn.CrossEntropyLoss()
        self.c = 0
        self.do_class_loss = self.module.num_classes is not None
        self.pure_eval = pure_eval

    def loss(self, outputs: Any, batch: Any, *args, **kwargs) -> Tensor:
        t0, t1, p0, p1, labels0, labels1 = batch
        outputs, logits = outputs[:-4], outputs[-4:]
        loss, loss_dict = self.objective(outputs)
        self.loss_dict = loss_dict
        if self.pure_eval:
            return loss_dict
        
        if self.do_class_loss:
            class_loss = (self.criterion(logits[0], labels0) + self.criterion(logits[1], labels0) + self.criterion(logits[2], labels1) + self.criterion(logits[3], labels1)) / 4
            self.loss_dict['class_loss'] = class_loss.item()
            loss += class_loss



        if self.c % 100 == 0:
            print(self.loss_dict)

        self.c += 1
        return loss


    def forward(self, batch: Tuple[Tensor, Tensor], is_eval=False) -> Tensor:
        if isinstance(batch, Tensor):
            inputs = batch
            features, outputs_inv, outputs_equi, logits = self.module(inputs)
            return features, outputs_inv
        else:
            t0, t1, p0, p1, _, _ = batch

            _, inv_00, equi_00, logits_00 = self.module(t0)
            _, inv_01, equi_01, logits_01 = self.module(p0)
            _, inv_10, equi_10, logits_10 = self.module(t1)
            _, inv_11, equi_11, logits_11 = self.module(p1)

            return inv_00, equi_00, inv_01, equi_01, inv_10, equi_10, inv_11, equi_11, logits_00, logits_01, logits_10, logits_11


    def get_backbone(self):
        return self.module
