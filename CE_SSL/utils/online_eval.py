import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from composer import Callback, State, Logger


###  For use during unsupervised pretrining to track progress ###
def test_one_epoch(
    net: nn.Module,
    test_data_loader: DataLoader,
):
    net.eval()
    total_top1, total_top5, total_num = 0.0, 0.0, 0
    test_bar = tqdm(test_data_loader)
    with torch.no_grad():
        for batch in test_bar:
            data, target = batch
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            logits = torch.flatten(net.lin_cls(net.f(data)), start_dim=1)
            pred_labels = logits.argsort(dim=-1, descending=True)
            total_num += data.size(0)
            total_top1 += torch.sum(
                (pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()
            ).item()
            total_top5 += torch.sum(
                (pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()
            ).item()

            test_bar.set_description(
                "Test Epoch: Acc@1:{:.2f}% Acc@5:{:.2f}%".format(
                    total_top1 / total_num * 100, total_top5 / total_num * 100
                )
            )

            if total_num == 0:
                total_num += 1
    net.train()

    if total_num == 0:
        total_num += 1
    return total_top1 / total_num * 100, total_top5 / total_num * 100


### COMPOSER EVALUATION VIA CALLBACK ###
class OnlineEval(Callback):
    def __init__(self, test_loader: DataLoader):
        super(OnlineEval, self).__init__()
        self.test_loader = test_loader
        self.count_online_eval = 0
        self.distributed = False
        self.top_acc = 0.0
        self.epochs_to_classify = []

    def epoch_end(self, state: State, logger: Logger):
        if self.count_online_eval % 10 == 0:
            if self.distributed:
                net = nn.parallel.DistributedDataParallel(state.model.module.module)
            else:
                net = state.model.module.module

            if net.num_classes is None:
                return

            top_1, top_5 = test_one_epoch(
                test_data_loader=self.test_loader,
                net=net,
            )

            print(f"Online Linear top_1={top_1}")
            print(f"Online Linear top_5={top_5}")

            if top_1 > self.top_acc:
                self.top_acc = top_1

        self.count_online_eval += 1
