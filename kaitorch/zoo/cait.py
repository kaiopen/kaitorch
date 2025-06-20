r'''Going deeper with Image Transformers'''

import torch
from torch import nn

from ..typing import TorchTensor, TorchFloat


class LayerScale(nn.Module):
    r'''

    From "Going deeper with Image Transformers".

    #### Args:
    - in_channels
    - scale: initial layer scale.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: a feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - A feature map. Its shape is `(B, in_channels, H, W)`.

    '''
    def __init__(
        self, in_channels: int, scale: float = 1e-4, *args, **kwargs
    ) -> None:
        super().__init__()
        self._scale = nn.Parameter(scale * torch.ones(in_channels))

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._scale * x
