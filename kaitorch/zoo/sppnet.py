r'''
Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition

'''
from typing import Sequence, Union
import torch
from torch import nn

from ..typing import TorchTensor, TorchFloat
from ..nn.pooling import max_pool2d


class SPP(nn.Module):
    r'''

    #### Args:
    - sizes: a sequence of kernel sizes for pooling.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, C, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, C * (len(sizes) + 1), H, W)`.

    '''
    def __init__(
        self,
        sizes: Sequence[int] = (5, 9, 13),
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._pools = nn.ModuleList(
            [
                max_pool2d(k, 1, k // 2, padding_mode=padding_mode)
                for k in sizes
            ]
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return torch.cat([x] + [pool(x) for pool in self._pools], dim=1)
