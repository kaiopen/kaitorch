r'''
Squeeze-and-Excitation Networks

'''
from typing import Any, Dict
from torch import nn

from ..nn.conv import Conv2dBlock
from ..typing import TorchTensor, TorchFloat


class SqueezeExcitation(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - reduction: reduction ratio.
    - activation_1: `relu`, `leakyrelu` or other activation after the first
        linear layer.
    - activation_kw_1: arguments of activation.
    - activation_2: `relu`, `leakyrelu` or other activation after the last
        linear layer.
    - activation_kw_2: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, in_channels, H, W)`.

    '''
    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
        activation_1: str = 'relu',
        activation_kw_1: Dict[str, Any] = {'inplace': True},
        activation_2: str = 'sigmoid',
        activation_kw_2: Dict[str, Any] = {},
        *args, **kwargs
    ) -> None:
        super().__init__()
        c = in_channels // reduction
        self._fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBlock(
                in_channels, c, 1,
                bias=True,
                activation=activation_1,
                activation_kw=activation_kw_1,
                mode='ca'
            ),
            Conv2dBlock(
                c, in_channels, 1,
                bias=True,
                activation=activation_2,
                activation_kw=activation_kw_2
            )
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return x * self._fc(x)
