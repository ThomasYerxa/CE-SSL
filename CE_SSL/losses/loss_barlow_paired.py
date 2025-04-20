import torch
from torch import nn, Tensor
import torch.nn.functional as F
import einops
import random
from typing import Tuple
import sys

from CE_SSL.losses.loss_barlow import Barlow_Loss as Vanilla_Barlow_Loss


class Barlow_Loss(nn.Module):
    def __init__(self, lmbda: float, distributed: bool = True, equi_lmbda=1.0, out_dim=8192, scale_loss: float = 0.048):
        super(Barlow_Loss, self).__init__()
        self.lmbda = lmbda 
        self.distributed = distributed
        self.equi_lambda = equi_lmbda
        self.first = True
        self.scale_loss = scale_loss

        self.loss_fn_inv = Vanilla_Barlow_Loss(lmbda=self.lmbda, distributed=self.distributed, scale_loss=scale_loss, out_dim=out_dim)
        self.loss_fn_equi = Vanilla_Barlow_Loss(lmbda=self.lmbda, distributed=self.distributed, scale_loss=scale_loss, out_dim=out_dim)

    def forward(self, outputs: Tuple[Tensor]) -> Tuple[Tensor, dict]:
        inv_00, equi_00, inv_01, equi_01, inv_10, equi_10, inv_11, equi_11 = outputs

        # compute normal Barlow Loss
        z_inv_view_1 = torch.cat([inv_00, inv_10], dim=0)
        z_inv_view_2 = torch.cat([inv_01, inv_11], dim=0)
        loss_inv, loss_dict_inv = self.loss_fn_inv((z_inv_view_1, z_inv_view_2))

        # compute equi MMCR Loss 
        diff_img_0 = equi_00 - equi_01
        diff_img_1 = equi_10 - equi_11
        loss_equi, loss_dict_equi = self.loss_fn_equi((diff_img_0, diff_img_1))

        loss = (1.0 - self.equi_lambda) * loss_inv + self.equi_lambda * loss_equi

        loss_dict = {
            "loss": loss.item(),
            "loss_inv": loss_inv.item(),
            "loss_equi": loss_equi.item(),
        }

        return loss, loss_dict