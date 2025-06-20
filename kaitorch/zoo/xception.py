from typing import Any, Dict, Sequence, Union
from torch import nn

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
        padding_mode: Union[str, Sequence[str]] = 'zeros',  # (M_H, M_W)
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._layer = nn.Sequential(
            conv2d(
                in_channels, in_channels, kernel_size, stride, padding,
                padding_mode=padding_mode
            ),
            conv2d(in_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._layer(x)


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
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
                    in_channels, out_channels, 3, 1, 1, padding_mode
                ),
                normalization('batchnorm2d', channels=out_channels)
            ]
            c = out_channels
        else:
            c = in_channels

        for _ in range(num - 1):
            layers += [
                _activation(activation, activation_kw),
                SeparableConv2d(c, c, 3, 1, 1, padding_mode),
                normalization('batchnorm2d', channels=c)
            ]

        if not grow_first:
            layers += [
                _activation(activation, activation_kw),
                SeparableConv2d(
                    in_channels, out_channels, 3, 1, 1, padding_mode
                ),
                normalization('batchnorm2d', channels=out_channels)
            ]

        if stride != 1:
            layers.append(max_pool2d(3, stride, 1, padding_mode=padding_mode))

        self._layer = nn.Sequential(*layers)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._layer(x) + self._shortcut(x)


class Xception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_category: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': False},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._entry = nn.Sequential(
            Conv2dBlock(
                in_channels, 32, 3, 2, 1,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                32, 64, 3,
                activation=activation,
                activation_kw=activation_kw
            ),
            Block(
                64, 128,
                stride=2,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw,
                num=2,
                activate_first=False,
                grow_first=True
            ),
            Block(
                128, 256,
                stride=2,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw,
                num=2,
                activate_first=True,
                grow_first=True
            ),
            Block(
                256, 728,
                stride=2,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw,
                num=2,
                activate_first=True,
                grow_first=True
            )
        )

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
                stride=2,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw,
                num=2,
                activate_first=True,
                grow_first=False
            ),
            SeparableConv2d(1024, 1536, 3, 1, 1, padding_mode),
            normalization('batchnorm2d', channels=1536),
            _activation(activation, activation_kw),
            SeparableConv2d(1536, 2048, 3, 1, 1, padding_mode),
            normalization('batchnorm2d', channels=2048),
            _activation(activation, activation_kw),
            nn.AdaptiveAvgPool2d(1)
        )
        self._linear = nn.Linear(2048, num_category)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        x = self._exit(self._middle(self._entry(x)))
        return self._linear(x.flatten(1))
