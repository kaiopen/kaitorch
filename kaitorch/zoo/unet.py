from typing import Any, Dict, Sequence, Union

import torch
from torch import nn

from kaitorch.typing import TorchTensor, TorchFloat

from ..nn.conv import Conv2dBlock


class DownSample(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the number of output channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: Input feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, out_channels, H // 2, W // 2)`,

    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._pool = nn.MaxPool2d(2)
        self._conv = nn.Sequential(
            Conv2dBlock(
                in_channels, out_channels, 3, 1, 1,
                padding_mode=padding_mode,
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

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._conv(self._pool(x))


class Upsample(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the number of output channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x_0: Input feature map. Its shape should be
        `(B, in_channels, H // 2, W // 2)`.
    - x_1: Input feature map. Its shape should be
        `(B, in_channels - in_channels // 2, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, out_channels, H, W)`,

    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2)
        self._conv = nn.Sequential(
            Conv2dBlock(
                in_channels, out_channels, 3, 1, 1,
                padding_mode=padding_mode,
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

    def forward(
        self, x_0: TorchTensor[TorchFloat], x_1: TorchTensor[TorchFloat]
    ) -> TorchTensor[TorchFloat]:
        return self._conv(torch.cat((x_1, self._up(x_0)), dim=1))


class UNet(nn.Module):
    r'''UNet backbone.

    #### Args:
    - in_channels: the number of input channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: Input feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, 64, H, W)`,

    '''
    def __init__(
        self,
        in_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._conv = nn.Sequential(
            Conv2dBlock(
                in_channels, 64, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                64, 64, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._down_0 = DownSample(
            64, 128, padding_mode, activation, activation_kw
        )
        self._down_1 = DownSample(
            128, 256, padding_mode, activation, activation_kw
        )
        self._down_2 = DownSample(
            256, 512, padding_mode, activation, activation_kw
        )
        self._down_3 = DownSample(
            512, 1024, padding_mode, activation, activation_kw
        )

        self._up_i5o4 = Upsample(
            1024, 512, padding_mode, activation, activation_kw
        )
        self._up_i4o3 = Upsample(
            512, 256, padding_mode, activation, activation_kw
        )
        self._up_i3o2 = Upsample(
            256, 128, padding_mode, activation, activation_kw
        )
        self._up_i2o1 = Upsample(
            128, 64, padding_mode, activation, activation_kw
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        p_1 = self._conv(x)
        p_2 = self._down_0(p_1)
        p_3 = self._down_1(p_2)
        p_4 = self._down_2(p_3)
        return self._up_i2o1(
            self._up_i3o2(
                self._up_i4o3(self._up_i5o4(self._down_3(p_4), p_4), p_3), p_2
            ), p_1
        )
