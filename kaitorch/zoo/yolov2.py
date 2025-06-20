import torch
from torch import nn

from ..typing import TorchTensor, TorchFloat


class ReOrg(nn.Module):
    r'''

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, C, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, 4C, H // 2, W // 2)`.

    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return torch.cat(
            (
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2]
            ),
            dim=1
        )
