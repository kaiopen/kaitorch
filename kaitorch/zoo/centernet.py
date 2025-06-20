from typing import Sequence, Union

import torch
from torch import nn

from ..typing import TorchTensor, TorchFloat
from ..nn.pooling import max_pool2d


class PseudoNMS(nn.Module):
    r'''

    #### Args:
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: a feature map. Its shape should be `(B, C, H, W)`.

    #### Returns:
    - A feature map. Its shape is `(B, C, H, W)`.

    '''
    def __init__(
        self,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._max_pool2d = max_pool2d(3, 1, 1, padding_mode=padding_mode)

    @torch.no_grad()
    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        x_max = self._max_pool2d(x)
        return x * (x == x_max).float()
