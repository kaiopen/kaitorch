r'''
UPANets: Learning from the Universal Pixel Attention Networks

'''
from typing import Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from ..typing import TorchTensor, TorchFloat
from ..data import tuple_2


class ExC(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        sizes: Sequence[Union[Tuple[int, int], int]],
        *args, **kwargs
    ):
        super().__init__()
        self._spas = nn.ModuleList()
        for s in sizes:
            h, w = tuple_2(s)
            self._spas.append(nn.Linear(h * w, 1))

        c = sum(in_channels)
        self._bn = nn.BatchNorm1d(c)
        self._fc = nn.Linear(c, out_channels)

    def forward(
        self, x: Sequence[TorchTensor[TorchFloat]]
    ) -> TorchTensor[TorchFloat]:
        x = list(x)
        for i, (spa, _x) in enumerate(zip(self._spas, x)):
            x[i] = F.layer_norm(
                spa(_x.flatten(2)).squeeze(-1) + torch.mean(_x, dim=(2, 3)),
                _x.shape[1: 2]
            )  # (B, C)

        return self._fc(self._bn(torch.cat(x, dim=-1)))
