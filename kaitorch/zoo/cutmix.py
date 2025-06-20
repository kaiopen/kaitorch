from typing import Tuple
import math
import random

import torch
import torch.nn.functional as F

from ..typing import TorchTensor, TorchReal


def cutmix_(
    x: TorchTensor[TorchReal], box: Tuple[int, int, int, int]
) -> TorchTensor[TorchReal]:
    x_0, y_0, x_1, y_1 = box
    x[:, :, y_0: y_1, x_0: x_1] = x.flip(0)[:, :, y_0: y_1, x_0: x_1]
    return x


def reverse_(
    x: TorchTensor[TorchReal], box: Tuple[int, int, int, int]
) -> TorchTensor[TorchReal]:
    x_0, y_0, x_1, y_1 = box
    x[:, :, y_0: y_1, x_0: x_1] = x.flip(0)[:, :, y_0: y_1, x_0: x_1]
    return x


class CutMix:
    def __init__(self, alpha: float = 1., num_category: int = 1000) -> None:
        self._dist = torch.distributions.Beta(
            torch.tensor((alpha,)), torch.tensor((alpha,))
        )
        self._num_cat = num_category

    def __call__(
        self, x: TorchTensor[TorchReal], label: TorchTensor[TorchReal]
    ) -> Tuple[TorchTensor[TorchReal], TorchTensor[TorchReal]]:
        box, lam = self.random_box(*x.shape[2:])
        return cutmix_(x, box), self.mix_label(label, lam)

    def mix_label(
        self, label: TorchTensor[TorchReal], lam: float,
    ) -> TorchTensor[TorchReal]:
        if 1 == label.ndim:
            label = F.one_hot(label, self._num_cat)

        label = label.float()
        return label.flip(0).mul_(1 - lam).add_(label.mul(lam))

    def random_box(
        self, height: int, width: int
    ) -> Tuple[Tuple[int, int, int, int], float]:
        lam = self._dist.sample().item()

        rate = 0.5 * math.sqrt(1. - lam)
        h = int(height * rate)
        w = int(width * rate)
        x = random.randint(0, width)
        y = random.randint(0, height)

        x_0 = max(0, min(x - w, width))
        y_0 = max(0, min(y - h, height))
        x_1 = max(0, min(x + w, width))
        y_1 = max(0, min(y + h, height))

        return (x_0, y_0, x_1, y_1), \
            1 - (x_1 - x_0) * (y_1 - y_0) / (height * width)
