r'''
Path Aggregation Network for Instance Segmentation

'''
from typing import Any, Dict, Sequence, Tuple, Union

import torch
from torch import nn

from ..typing import TorchTensor, TorchFloat
from ..nn.conv import Conv2dBlock


class Upsample(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the number of output channels.
    - factor: interpolating scale factor.
    - mode: interpolating mode.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, in_channels, H, W)`,

    #### Returns:
    - Feature map. Its shape is `(B, out_channels, H * factor, W * factor)`

    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 2,
        mode: str = 'nearest',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._conv = Conv2dBlock(
            in_channels, out_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )
        self._up = nn.Upsample(scale_factor=factor, mode=mode)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._up(self._conv(x))


class UpsampleE(nn.Module):
    r'''Upsample and concatenate with an extral input.

    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the number of output channels.
    - factor: interpolating scale factor.
    - mode: interpolating mode.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: feature map that will be upsampled. Its shape should be
        `(B, in_channels[0], H, W)`,
    - e: feature map. Its shape should be
        `(B, in_channels[1], H * factor, W * factor)`.

    #### Returns:
    - Feature map. Its shape is
        `(B, out_channels // 2 * 2, H * factor, W * factor)`

    '''
    def __init__(
        self,
        in_channels: Tuple[int, int],
        out_channels: int,
        factor: int = 2,
        mode: str = 'nearest',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        out_channels //= 2
        self._upsample = Upsample(
            in_channels[0], out_channels,
            factor=factor,
            mode=mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._conv = Conv2dBlock(
            in_channels[1], out_channels, 1, 1, 0,
            activation=activation,
            activation_kw=activation_kw
        )

    def forward(
        self, x: TorchTensor[TorchFloat], e: TorchTensor[TorchFloat]
    ) -> TorchTensor[TorchFloat]:
        return torch.cat((self._upsample(x), self._conv(e)), dim=1)


def make_layers(
    num_block: int, in_channels: int, out_channels: int,
    padding_mode: Union[str, Sequence[str]] = 'zeros',
    activation: str = 'relu',
    activation_kw: Dict[str, Any] = {'inplace': True}
):
    layers = [
        Conv2dBlock(
            in_channels, out_channels, 1,
            activation=activation,
            activation_kw=activation_kw
        )
    ]
    for _ in range(num_block):
        layers += [
            Conv2dBlock(
                out_channels, in_channels, 3,
                padding=1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                in_channels, out_channels, 1,
                activation=activation,
                activation_kw=activation_kw
            )
        ]
    return nn.Sequential(*layers)


class PAN(nn.Module):
    r'''Path Aggregation Network.

    #### Args:
    - max_channels: maximal input channels.
    - num_map: number of feature maps in the input sequence.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    -x: sequence of feature maps. The shape of the last feature map in the
        sequence should be `(B, max_channels, H, W)`. And the shape of the
        penultimate one should be `(B, max_channels, H * 2, W * 2)`. The
        previous one should be in `(B, max_channels // 2, H * 4, W * 4)` if
        exists. And `(B, max_channels // 4, H * 8, W * 8)` and so on.

    #### Returns:
    - Sequence of feature maps. The last one is in `(B, max_channels, H, W)`.
        The previous one is in `(B, max_channels // 2, H * 2, W * 2)` and so
        on.

    '''
    def __init__(
        self,
        max_channels: int,
        num_map: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
    ) -> None:
        super().__init__()
        self._range_up = range(num_map - 2, -1, -1)
        self._range_down = range(num_map - 2)

        upsamples = []
        convs = []
        cats = []
        for _ in range(num_map - 1):
            half_c = max_channels // 2

            upsamples.append(
                UpsampleE(
                    (max_channels, max_channels), max_channels,
                    activation=activation,
                    activation_kw=activation_kw
                )
            )
            cats.append(
                make_layers(
                    2, max_channels, half_c,
                    padding_mode, activation, activation_kw
                )
            )
            max_channels = half_c
        upsamples.reverse()
        convs.reverse()
        cats.reverse()
        self._upsamples = nn.ModuleList(upsamples)
        self._convs = nn.ModuleList(convs)
        self._cats_up = nn.ModuleList(cats)

        min_channels = max_channels
        downsamples = []
        cats = []
        for _ in range(num_map - 1):
            double_c = min_channels * 2
            downsamples.append(
                Conv2dBlock(
                    min_channels, double_c, 3, 2, 1,
                    padding_mode=padding_mode,
                    activation=activation,
                    activation_kw=activation_kw
                )
            )
            cats.append(
                make_layers(
                    2, double_c * 2, double_c,
                    padding_mode, activation, activation_kw
                )
            )
            min_channels = double_c
        self._downsamples = nn.ModuleList(downsamples)
        self._cats_down = nn.ModuleList(cats)

    def forward(
        self, x: Sequence[TorchTensor[TorchFloat]]
    ) -> TorchTensor[TorchFloat]:
        for i in self._range_up:
            x[i] = self._cats_up[i](self._upsamples[i](x[i + 1], x[i]))

        for i in self._range_down:
            j = i + 1
            x[j] = self._cats_down[i](
                torch.cat((x[j], self._downsamples[i](x[i])), dim=1)
            )
        return x
