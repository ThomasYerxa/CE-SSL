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

class MomentumModel(nn.Module):
    def __init__(
        self,
        projector_dims_1: list = [8192, 8192, 512],
        projector_dims_2: list = [8192, 8192, 512],
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

        # initialize momentum background and projector
        self.mom_f.fc = nn.Identity()

        # projector_heads
        self.g_1, self.g_2 = self.build_unsplit_projectors(projector_dims_1, projector_dims_2, bias_last, bias_proj, output_dim=output_dim)
        self.mom_g_1, self.mom_g_2= self.build_unsplit_projectors(projector_dims_1, projector_dims_2, bias_last, bias_proj, output_dim=output_dim)


        params_f_online, params_f_mom = self.f.parameters(), self.mom_f.parameters()
        params_g_1_online, params_g_1_mom = self.g_1.parameters(), self.mom_g_1.parameters()
        params_g_2_online, params_g_2_mom = self.g_2.parameters(), self.mom_g_2.parameters()

        for po, pm in zip(params_f_online, params_f_mom):
            pm.data.copy_(po.data)
            pm.requires_grad = False

        for po, pm in zip(params_g_1_online, params_g_1_mom):
            pm.data.copy_(po.data)
            pm.requires_grad = False

        for po, pm in zip(params_g_2_online, params_g_2_mom):
            pm.data.copy_(po.data)
            pm.requires_grad = False

        self.num_classes = num_classes
        if num_classes is not None:
            self.lin_cls = nn.Linear(output_dim, num_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_ = self.f(x)
        feature = torch.flatten(x_, start_dim=1)
        out_inv = self.g_1(feature)
        out_equi = self.g_2(feature)

        feature_momentum = torch.flatten(self.mom_f(x), start_dim=1)
        out_inv_momentum = self.mom_g_1(feature_momentum)
        out_equi_momentum = self.mom_g_2(feature_momentum)

        if self.num_classes is not None:
            logits = self.lin_cls(feature.detach())
        else:
            logits = None

        return feature, out_inv, out_equi, feature_momentum, out_inv_momentum, out_equi_momentum, logits

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
        g_1 = nn.Sequential(*layers)

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
        g_2 = nn.Sequential(*layers)

        return g_1, g_2

class MomentumComposerWrapper(ComposerModel):
    def __init__(self, module: torch.nn.Module, objective, pure_eval=False):
        super().__init__()

        self.module = module
        self.objective = objective
        self.c = 0  # counts the number of forward calls
        self.criterion = nn.CrossEntropyLoss()
        self.do_class_loss = self.module.num_classes is not None
        self.pure_eval = pure_eval

    def loss(self, outputs: Any, batch: Any, *args, **kwargs) -> Tensor:
        t0, t1, p0, p1, labels0, labels1 = batch
        outputs, logits = outputs[:-4], outputs[-4:]
        loss, loss_dict = self.objective(outputs)
        self.loss_dict = loss_dict
        if self.pure_eval:
            return self.loss_dict
        
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
            features, outputs_inv, outputs_equi, features_momentum, outputs_inv_momentum, outputs_equi_momentum, logits = self.module(inputs)
            return features, outputs_inv
        else:
            t0, t1, p0, p1, _, _ = batch

            _, inv_00, equi_00, _, inv_mom_00, equi_mom_00, logits_00 = self.module(t0)
            _, inv_01, equi_01, _, inv_mom_01, equi_mom_01, logits_01 = self.module(p0)
            _, inv_10, equi_10, _, inv_mom_10, equi_mom_10, logits_10 = self.module(t1)
            _, inv_11, equi_11, _, inv_mom_11, equi_mom_11, logits_11 = self.module(p1)

            return inv_00, inv_mom_00, equi_00, equi_mom_00, inv_01, inv_mom_01, equi_01, equi_mom_01, inv_10, inv_mom_10, equi_10, equi_mom_10, inv_11, inv_mom_11, equi_11, equi_mom_11, logits_00, logits_01, logits_10, logits_11

    def get_backbone(self):
        return self.module
