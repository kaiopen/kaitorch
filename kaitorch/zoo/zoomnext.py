r'''
ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object
Detection

'''

import math
import torch

from ..typing import TorchTensor, TorchFloat


def balance_coefficient(p: float) -> float:
    return (1 - math.cos(p * math.pi)) / 2


def uncertainty_aware_loss(
    x: TorchTensor[TorchFloat]
) -> TorchTensor[TorchFloat]:
    return torch.mean(1 - torch.pow(torch.abs(2 * x.sigmoid() - 1), 2))
