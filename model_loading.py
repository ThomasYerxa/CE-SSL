from collections import OrderedDict

import torch
import torchvision
from torchvision.models import resnet

from urllib.request import urlretrieve

def load_core_model(
    objective_name: str,
    lmda: float,
):
    """
    Load a model from a given objective name and lambda value.

    Args:
        objective_name (str): The name of the objective.
        lmda (float): The lambda value.
        architecture (str): The architecture of the model. Default is "resnet50".

    Returns:
        torch.nn.Module: The loaded model.
    """
    objectives = ['SimCLR', 'MMCR', 'Barlow']
    if objective_name not in objectives:
        raise ValueError(f"Objective name must be one of {objectives}.")
    if lmda not in [0.0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5]:
        raise ValueError("Lambda must be one of [0.0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5].")

    ckpt_location = f'~tyerxa/equi_proj/training_checkpoints/final/resnet_50/{objective_name}/lmda_{lmda}/ep100-ba62500-rank0'
    url = f'https://users.flatironinstitute.org/{ckpt_location}'
    fh = urlretrieve(url)
    state_dict = torch.load(fh[0], map_location=torch.device("cpu"))["state"]["model"]
    model = _load_composer_classifier_r50(state_dict)

    return model.eval()

def load_top_model():
    """
    Load the Barlow Twins model trained with lmda=0.2 for 1000 epochs.
    (Top performing model from paper on Brain-Score IT evaluation)

    Returns:
        torch.nn.Module: The loaded model.
    """
    url = "https://users.flatironinstitute.org/~tyerxa/equi_proj/training_checkpoints/final/resnet_50/Barlow/1000_epoch/ep1000-ba625000-rank0"
    fh = urlretrieve(url)
    state_dict = torch.load(fh[0], map_location=torch.device("cpu"))["state"]["model"]
    model = _load_composer_classifier_r50(state_dict)

    return model.eval()


def _load_composer_classifier_r50(sd):
    """
    Load a model saved via Composer into a vanilla torchvision model.
    """
    model = torchvision.models.resnet.resnet50()
    new_sd = OrderedDict()
    for k, v in sd.items():
        if 'lin_cls' in k:
            new_sd['fc.' + k.split('.')[-1]] = v
        if ".f." not in k:
            continue
        parts = k.split(".")
        idx = parts.index("f")
        new_k = ".".join(parts[idx + 1 :])
        new_sd[new_k] = v
    model.load_state_dict(new_sd, strict=True)
    return model