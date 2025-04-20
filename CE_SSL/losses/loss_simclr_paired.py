import torch
from torch import nn, Tensor
import torch.nn.functional as F
import einops
import random
from typing import Tuple
import sys

from CE_SSL.losses.loss_simclr import SimCLR_Loss as Vanilla_SimCLR_Loss


class SimCLR_Loss(nn.Module):
    def __init__(self, lmbda: float, distributed: bool = True, equi_lmbda=1.0):
        super(SimCLR_Loss, self).__init__()
        self.lmbda = lmbda
        self.equi_lambda = equi_lmbda
        self.distributed = distributed

        self.loss_fn_inv = Vanilla_SimCLR_Loss(lmbda=self.lmbda, distributed=self.distributed)
        self.loss_fn_equi = Vanilla_SimCLR_Loss(lmbda=self.lmbda, distributed=self.distributed)

    def forward(self, outputs: Tuple[Tensor]) -> Tuple[Tensor, dict]:
        inv_00, equi_00, inv_01, equi_01, inv_10, equi_10, inv_11, equi_11 = outputs

        # compute normal SimCLR Loss
        z_inv_view_1 = torch.cat([inv_00, inv_10], dim=0)
        z_inv_view_2 = torch.cat([inv_01, inv_11], dim=0)
        loss_inv, loss_dict_inv = self.loss_fn_inv((z_inv_view_1, z_inv_view_2))

        # compute equi SimCLR Loss 
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
