r'''FractalNet: Ultra-Deep Neural Networks without Residuals'''

import torch
from torch import nn

from ..typing import TorchTensor, TorchFloat


class DropPath(nn.Module):
    r'''

    From "FractalNet: Ultra-Deep Neural Networks without Residuals".

    '''
    def __init__(self, p: float = 0.5, *args, **kwargs) -> None:
        super().__init__()
        self._k = 1 - p

    def extra_repr(self):
        return f'p={1 - self._k}'

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        if 1 == self._k or not self.training:
            return x

        r = self._k + torch.rand(
            (x.shape[0],) + (1,) * (x.ndim - 1),
            dtype=x.dtype, device=x.device
        )
        r.floor_()
        return x.div(self._k) * r
