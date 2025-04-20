import torch
from torch import nn, Tensor
import torch.nn.functional as F
import einops
import random
from typing import Tuple
import sys

from CE_SSL.losses.loss_mmcr_momentum import MMCR_Momentum_Loss as Vanilla_MMCR_Loss


class MMCR_Momentum_Loss(nn.Module):
    def __init__(self, lmbda: float, distributed: bool = True, equi_lmbda=1.0):
        super(MMCR_Momentum_Loss, self).__init__()
        self.lmbda = lmbda
        self.equi_lambda = equi_lmbda
        self.distributed = distributed

        self.loss_fn_inv = Vanilla_MMCR_Loss(lmbda=self.lmbda, distributed=self.distributed)
        self.loss_fn_equi = Vanilla_MMCR_Loss(lmbda=self.lmbda, distributed=self.distributed)

    def forward(self, outputs: Tuple[Tensor]) -> Tuple[Tensor, dict]:
        inv_00, equi_00, inv_01, equi_01, inv_10, equi_10, inv_11, equi_11 = outputs[::2]
        inv_mom_00, equi_mom_00, inv_mom_01, equi_mom_01, inv_mom_10, equi_mom_10, inv_mom_11, equi_mom_11 = outputs[1::2]

        # compute normal MMCR Loss
        z_inv_view_1 = torch.cat([inv_00, inv_10], dim=0)
        z_inv_view_2 = torch.cat([inv_01, inv_11], dim=0)
        z_inv_view_1_mom = torch.cat([inv_mom_00, inv_mom_10], dim=0)
        z_inv_view_2_mom = torch.cat([inv_mom_01, inv_mom_11], dim=0)
        loss_inv, loss_dict_inv = self.loss_fn_inv((z_inv_view_1, z_inv_view_2, z_inv_view_1_mom, z_inv_view_2_mom ))

        # compute equi MMCR Loss 
        diff_img_0 = equi_00 - equi_01
        diff_img_1 = equi_10 - equi_11
        diff_img_0_mom = equi_mom_00 - equi_mom_01
        diff_img_1_mom = equi_mom_10 - equi_mom_11
        loss_equi, loss_dict_equi = self.loss_fn_equi((diff_img_0, diff_img_1, diff_img_0_mom, diff_img_1_mom))

        loss = (1.0 - self.equi_lambda) * loss_inv + self.equi_lambda * loss_equi

        loss_dict = {
            "loss": loss.item(),
            "loss_inv": loss_inv.item(),
            "loss_equi": loss_equi.item(),
        }

        return loss, loss_dict