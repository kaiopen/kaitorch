r'''
F3Net: Fusion, Feedback and Focus for Salient Object Detection

'''

import torch
import torch.nn.functional as F

from ..typing import TorchTensor, TorchFloat


def structure_loss(
    x: TorchTensor[TorchFloat], t: TorchTensor[TorchFloat]
) -> TorchTensor[TorchFloat]:
    w = 1 + 5 * torch.abs(F.avg_pool2d(t, 31, 1, 15) - t)
    bce = torch.sum(
        w * F.binary_cross_entropy_with_logits(x, t, reduction='none'),
        dim=(2, 3)
    ) / torch.sum(w, dim=(2, 3))

    x = torch.sigmoid(x)
    inter = torch.sum(x * t * w, dim=(2, 3))
    return torch.mean(
        1 - (inter + 1) / (torch.sum((x + t) * w, dim=(2, 3)) - inter + 1)
        + bce
    )
