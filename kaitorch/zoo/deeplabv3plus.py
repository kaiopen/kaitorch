from typing import Any, Dict, Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from ..typing import TorchTensor, TorchFloat
from ..nn.conv import Conv2dBlock, conv2d
from ..nn.activation import activation as _activation
from ..nn.normalization import normalization
from ..nn.pooling import max_pool2d


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int]] = 0,  # (P_H, P_W)
        dilation: int = 1,
        padding_mode: Union[str, Sequence[str]] = 'zeros',  # (M_H, M_W)
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._layer = nn.Sequential(
            conv2d(
                in_channels, in_channels,
                kernel_size, stride, padding, dilation,
                padding_mode=padding_mode
            ),
            conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._layer(x)


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': False},
        num: int = 1,
        activate_first: bool = True,
        grow_first: bool = True,
        *args, **kwargs
    ) -> None:
        super().__init__()
        if num < 1:
            raise ValueError(
                f'`num` should be larger than 0 but {num} is gotten.'
            )

        if in_channels != out_channels or stride != 1:
            self._shortcut = Conv2dBlock(
                in_channels, out_channels, 1, stride=stride, mode='cn'
            )
        else:
            self._shortcut = nn.Identity()

        layers = []
        if activate_first:
            layers.append(_activation(activation, activation_kw))

        if grow_first:
            layers += [
                SeparableConv2d(
                    in_channels, out_channels, 3, 1,
                    padding=dilation,
                    dilation=dilation,
                    padding_mode=padding_mode
                ),
                normalization('batchnorm2d', channels=out_channels)
            ]
            c = out_channels
        else:
            c = in_channels

        for _ in range(num - 1):
            layers += [
                _activation(activation, activation_kw),
                SeparableConv2d(
                    c, c, 3, 1,
                    padding=dilation,
                    dilation=dilation,
                    padding_mode=padding_mode
                ),
                normalization('batchnorm2d', channels=c)
            ]

        if not grow_first:
            layers += [
                _activation(activation, activation_kw),
                SeparableConv2d(
                    in_channels, out_channels, 3, 1,
                    padding=dilation,
                    dilation=dilation,
                    padding_mode=padding_mode
                ),
                normalization('batchnorm2d', channels=out_channels)
            ]

        if stride != 1:
            layers.append(max_pool2d(3, stride, 1, padding_mode=padding_mode))

        self._layer = nn.Sequential(*layers)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._layer(x) + self._shortcut(x)


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dilations: Sequence[int] = (6, 12, 18),
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': False},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._layers = nn.ModuleList(
            [
                Conv2dBlock(
                    in_channels, 256, 1,
                    activation=activation,
                    activation_kw=activation_kw
                ),
            ]
        )

        for d in dilations:
            self._layers.append(
                Conv2dBlock(
                    in_channels, 256, 3,
                    padding=d,
                    dilation=d,
                    padding_mode=padding_mode,
                    activation=activation,
                    activation_kw=activation_kw
                )
            )

        self._pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBlock(
                in_channels, 256, 1,
                activation=activation,
                activation_kw=activation_kw
            )
        )

        self._project = nn.Sequential(
            Conv2dBlock(
                256 * (len(dilations) + 2), 256, 1,
                activation=activation,
                activation_kw=activation_kw
            ),
            nn.Dropout(0.1)
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        outs = []
        for layer in self._layers:
            outs.append(layer(x))
        outs.append(
            F.interpolate(self._pool(x), size=x.shape[2: 4], mode='bilinear')
        )
        return self._project(torch.cat(outs, dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        in_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': False},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._stem = nn.Sequential(
            Conv2dBlock(
                in_channels, 32, 3, 2, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),  # 1 / 2
            Conv2dBlock(
                32, 64, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._block_1 = Block(
            64, 128,
            stride=2,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw,
            num=2,
            activate_first=False,
            grow_first=True
        )  # 1 / 4
        self._block_2 = Block(
            128, 256,
            stride=2,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw,
            num=2,
            activate_first=True,
            grow_first=True
        )  # 1 / 8
        self._block_3 = Block(
            256, 728,
            stride=2,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw,
            num=2,
            activate_first=True,
            grow_first=True
        )  # 1 / 16

        self._middle = nn.Sequential(
            Block(
                728, 728,
                stride=1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw,
                num=3,
                activate_first=True,
                grow_first=True
            ),
            Block(
                728, 728,
                stride=1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw,
                num=3,
                activate_first=True,
                grow_first=True
            ),
            Block(
                728, 728,
                stride=1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw,
                num=3,
                activate_first=True,
                grow_first=True
            ),
            Block(
                728, 728,
                stride=1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw,
                num=3,
                activate_first=True,
                grow_first=True
            ),
            Block(
                728, 728,
                stride=1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw,
                num=3,
                activate_first=True,
                grow_first=True
            ),
            Block(
                728, 728,
                stride=1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw,
                num=3,
                activate_first=True,
                grow_first=True
            ),
            Block(
                728, 728,
                stride=1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw,
                num=3,
                activate_first=True,
                grow_first=True
            ),
            Block(
                728, 728,
                stride=1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw,
                num=3,
                activate_first=True,
                grow_first=True
            )
        )

        self._exit = nn.Sequential(
            Block(
                728, 1024,
                stride=1,
                dilation=2,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw,
                num=2,
                activate_first=True,
                grow_first=False
            ),
            SeparableConv2d(1024, 1536, 3, 1, 1, padding_mode=padding_mode),
            normalization('batchnorm2d', channels=1536),
            _activation(activation, activation_kw),
            SeparableConv2d(1536, 2048, 3, 1, 1, padding_mode=padding_mode)
        )

    def forward(
        self, x: TorchTensor[TorchFloat]
    ) -> Tuple[TorchTensor[TorchFloat], TorchTensor[TorchFloat]]:
        x = self._block_1(self._stem(x))
        return x, self._exit(self._middle(self._block_3(self._block_2(x))))


class Head(nn.Module):
    def __init__(
        self,
        num_category: int,
        in_channels: Sequence[int] = (128, 2048),
        dilations: Sequence[int] = (6, 12, 18),
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': False},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._project = Conv2dBlock(
            in_channels[0], 48, 1,
            activation=activation,
            activation_kw=activation_kw
        )
        self._aspp = ASPP(
            in_channels[1], dilations, padding_mode, activation, activation_kw
        )

        self._classifier = nn.Sequential(
            Conv2dBlock(
                304, 256, 3, 1, 1,
                activation=activation,
                activation_kw=activation_kw
            ),
            conv2d(256, num_category, 1)
        )

    def forward(
        self, x: Tuple[TorchTensor[TorchFloat], TorchTensor[TorchFloat]]
    ) -> TorchTensor[TorchFloat]:
        x_0, x_1 = x
        return self._classifier(
            torch.cat(
                (
                    self._project(x_0),
                    F.interpolate(
                        self._aspp(x_1), size=x_0.shape[2:], mode='bilinear'
                    )
                ),
                dim=1
            )
        )
