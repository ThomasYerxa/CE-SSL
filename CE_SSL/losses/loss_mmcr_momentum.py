import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import List, Tuple
import einops


class MMCR_Momentum_Loss(nn.Module):
    def __init__(
        self,
        lmbda: float,
        distributed: bool = True,
    ):
        super(MMCR_Momentum_Loss, self).__init__()
        self.lmbda = lmbda
        self.distributed = distributed

    def forward(self, outputs: Tuple[Tensor]) -> Tuple[Tensor, dict]:
        z1, z2, z1_momentum, z2_momentum = outputs
        z1, z2,  = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
        z1_momentum, z2_momentum = F.normalize(z1_momentum, dim=-1), F.normalize(z2_momentum, dim=-1)


        # gather across devices into list
        if self.distributed:
            z1, z2 = self.gather(z1), self.gather(z2)
            z1_momentum, z2_momentum = self.gather(z1_momentum), self.gather(z2_momentum)

        z = torch.stack([z1, z2, z1_momentum, z2_momentum], dim=-1)
        if self.lmbda == 0:
            local_nuc = 0
        else:
            local_nuc = torch.linalg.svdvals(z).sum()


        centroids = torch.mean(z, dim=-1)
        global_nuc = torch.linalg.svdvals(centroids).sum()

        batch_size = z.shape[0]
        loss = -1.0 * global_nuc + self.lmbda * local_nuc / batch_size

        loss_dict = {
            "loss": loss.item(),
            "local_nuc": local_nuc,
            "global_nuc": global_nuc.item(),
        }

        return loss, loss_dict

    def gather(self, tensor: Tensor) -> Tensor:
        tensor_list = [torch.zeros_like(tensor) for i in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensor_list, tensor, async_op=False)
        tensor_list[torch.distributed.get_rank()] = tensor
        return torch.cat(tensor_list)