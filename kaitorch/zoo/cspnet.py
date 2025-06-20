r'''
CSPNet: A New Backbone that can Enhance Learning Capability of CNN

'''
from typing import Any, Dict, Optional, Sequence, Union

import torch
from torch import nn

from ..typing import TorchTensor, TorchFloat
from ..nn.conv import Conv2dBlock
from .sppnet import SPP
from .yolov3 import Bottleneck


class BottleneckCSPB(nn.Module):
    r'''

    #### Args:
    - in_channels
    - out_channels
    - num: the number of the bottleneck from the Darknet.
    - shortcut: Whether to do shortcut.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation function.
    - activation_kw: the arguments to the activation function.

    #### Methods;
    - forward

    ## forward
    #### Args:
    - x: a feature map. Its shape should be `(B, in_channels, H, W)`,

    #### Returns:
    - A feature map. Its shape is `(B, out_channels, H, W)`.

    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num: int = 1,
        shortcut: bool = False,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: Optional[str] = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._conv_0 = Conv2dBlock(
            in_channels, out_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )
        self._bottlenecks = nn.Sequential(
            *[
                Bottleneck(
                    out_channels, out_channels,
                    padding_mode=padding_mode,
                    activation=activation,
                    activation_kw=activation_kw,
                    shortcut=shortcut
                ) for _ in range(num)
            ]
        )
        self._conv_1 = Conv2dBlock(
            out_channels, out_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )
        self._cat = Conv2dBlock(
            2 * out_channels, out_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        x = self._conv_0(x)
        return self._cat(
            torch.cat((self._bottlenecks(x), self._conv_1(x)), dim=1)
        )


class SPPCSPC(nn.Module):
    r'''

    #### Args:
    - in_channels
    - out_channels
    - sizes: kernel sizes.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation function.
    - activation_kw: the arguments to the activation function.

    #### Methods;
    - forward

    ## forward
    #### Args:
    -x: a feature map. Its shape should be `(B, in_channels, H, W)`,

    #### Returns:
    - A feature map. Its shape is `(B, out_channels, H, W)`.

    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sizes: Sequence[int] = (5, 9, 13),
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: Optional[str] = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._i0o1 = nn.Sequential(
            Conv2dBlock(
                in_channels, out_channels, 1, 1, 0,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                out_channels, out_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                out_channels, out_channels, 1, 1, 0,
                activation=activation,
                activation_kw=activation_kw
            ),
            SPP(sizes=sizes, padding_mode=padding_mode),
            Conv2dBlock(
                out_channels * (len(sizes) + 1), out_channels, 1, 1, 0,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                out_channels, out_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._i0o2 = Conv2dBlock(
            in_channels, out_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )
        self._ico3 = Conv2dBlock(
            out_channels * 2, out_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._ico3(torch.cat((self._i0o1(x), self._i0o2(x)), dim=1))
