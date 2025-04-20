import torch
from torch import nn, Tensor
import torch.nn.functional as F
import einops
import random
from typing import Tuple
import sys

class Barlow_Loss(nn.Module):
    def __init__(self, lmbda: float, out_dim=8192, scale_loss: float = 0.048, distributed: bool = False):
        super(Barlow_Loss, self).__init__()
        self.lmbda = lmbda 
        self.bn_inv = nn.BatchNorm1d(out_dim, affine=False)
        self.scale_loss = scale_loss
        self.distributed = distributed

    def forward(self, outputs: Tuple[Tensor]) -> Tuple[Tensor, dict]:
        z1, z2 = outputs

        # sum the cross-correlation matrix between all gpus
        bs = z1.shape[0]
        if self.distributed:
            bs *= torch.distributed.get_world_size()

        c_inv = self.bn_inv(z1).T @ self.bn_inv(z2)
        c_inv.div_(bs)
        if self.distributed:
            torch.distributed.all_reduce(c_inv)

        on_diag_inv = torch.diagonal(c_inv).add_(-1).pow_(2).sum()
        off_diag_inv = off_diagonal(c_inv).pow_(2).sum()
        loss = on_diag_inv + self.lmbda * off_diag_inv
        loss = loss * self.scale_loss

        loss_dict = {
            "loss": loss.item(),
            "on_diag": on_diag_inv.item(),
            "off_diag": off_diag_inv.item(),
        }

        return loss, loss_dict


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()