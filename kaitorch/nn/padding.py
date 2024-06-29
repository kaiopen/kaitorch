from typing import Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from .utils import _ntuple


class _CircularPadNd(nn.Module):
    __constants__ = ['padding']
    padding: Sequence[int]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.pad(input, self.padding, 'circular')

    def extra_repr(self) -> str:
        return '{}'.format(self.padding)


class CircularPad2d(_CircularPadNd):
    padding: Tuple[int, int, int, int]

    def __init__(self, padding: Union[int, Tuple[int]]) -> None:
        super(CircularPad2d, self).__init__()
        self.padding = _ntuple(4)(padding)


PAD2D = {
    'zeros': nn.ZeroPad2d,
    'reflect': nn.ReflectionPad2d,
    'replicate': nn.ReplicationPad2d,
    'circular': CircularPad2d
}
