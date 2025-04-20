import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import einops
import random
from typing import Tuple

import sys


# modified from https://github.com/vturrisi/solo-learn/blob/main/solo/losses/simclr.py
class SimCLR_Loss(nn.Module):
    def __init__(self, lmbda: float, distributed: bool = True):
        super(SimCLR_Loss, self).__init__()
        self.temperature = lmbda
        self.distributed = distributed
        self.rank = None
        self.world_size = None
        self.pos_mask = None
        self.neg_mask = None

    def forward(self, outputs: Tuple[Tensor]) -> Tuple[Tensor, dict]:
        if (self.rank is None or self.world_size is None) and self.distributed:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        if self.pos_mask is None:
            self.num_pos = 2
            self.effective_batch_size = self.world_size * outputs[0].shape[0] * self.num_pos
            self.precompute_pos_neg_mask()

        z1, z2 = outputs
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=-1)

        pos, neg, loss = self.calculate_loss(z)
        

        loss_dict = {
            'loss': loss.item(),
            'pos': pos.abs().mean().item(),
            'neg': neg.abs().mean().item()
        }

        return loss, loss_dict

    def calculate_loss(self, embedding: torch.Tensor):
        """
        Calculate the loss. Operates on embeddings tensor.
        """
        assert embedding.ndim == 2

        batch_size = embedding.shape[0]
        T = self.temperature
        num_pos = self.num_pos
        assert batch_size % num_pos == 0, "Batch size should be divisible by num_pos"

        # Step 1: gather all the embeddings. Shape example: 4096 x 128
        embeddings_buffer = self.gather(embedding)

        # Step 2: matrix multiply: 64 x 128 with 4096 x 128 = 64 x 4096 and
        # divide by temperature.
        similarity = torch.exp(torch.mm(embedding, embeddings_buffer.t()) / T)
        pos = torch.sum(similarity * self.pos_mask, 1)
        neg = torch.sum(similarity * self.neg_mask, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))
        return pos, neg, loss


    def precompute_pos_neg_mask(self):
        """
        We precompute the positive and negative masks to speed up the loss calculation
        """
        # computed once at the begining of training
        total_images = self.effective_batch_size
        world_size = self.world_size
        batch_size = total_images // world_size
        orig_images = batch_size // self.num_pos
        rank = self.rank

        pos_mask = torch.zeros(batch_size, total_images)
        neg_mask = torch.zeros(batch_size, total_images)

        all_indices = np.arange(total_images)
        pos_members = orig_images * np.arange(self.num_pos)
        orig_members = torch.arange(orig_images)
        for anchor in np.arange(self.num_pos):
            for img_idx in range(orig_images):
                delete_inds = batch_size * rank + img_idx + pos_members
                neg_inds = torch.tensor(np.delete(all_indices, delete_inds)).long()
                neg_mask[anchor * orig_images + img_idx, neg_inds] = 1
            for pos in np.delete(np.arange(self.num_pos), anchor):
                pos_inds = batch_size * rank + pos * orig_images + orig_members
                pos_mask[
                    torch.arange(
                        anchor * orig_images, (anchor + 1) * orig_images
                    ).long(),
                    pos_inds.long(),
                ] = 1
        self.pos_mask = pos_mask.cuda(non_blocking=True) #if self.use_gpu else pos_mask
        self.neg_mask = neg_mask.cuda(non_blocking=True) #if self.use_gpu else neg_mask

    def gather(self, tensor: Tensor) -> Tensor:
        tensor_list = [torch.zeros_like(tensor) for i in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensor_list, tensor, async_op=False)
        tensor_list[torch.distributed.get_rank()] = tensor
        return torch.cat(tensor_list)