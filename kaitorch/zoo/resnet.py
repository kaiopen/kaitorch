r'''
Deep Residual Learning for Image Recognition

'''
from typing import Any, Dict, Optional, Sequence, Union

from torch import nn

from ..nn.activation import activation as _activation
from ..nn.conv import Conv2dBlock
from ..typing import TorchTensor, TorchFloat


class BasicBlock(nn.Module):
    r'''Basic block for ResNet.

    a((cn(cna(x))) + cn(x))

    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the number of output channels. If it is a `None`, it is
        equal to the `in_channels`. If it is a string starting with a "x", the
        number of out channels is `int(float(out_channels[1:]) * in_channels)`.
    - stride: stride of the convolution. Its length should be 2 if it is a
        sequence.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, out_channels, H', W')`, where
        `H' = floor((H -1) / stride) + 1` and
        `W' = floor((W -1) / stride) + 1`.

    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[Union[int, str]] = None,
        stride: Union[int, Sequence[int]] = 1,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        if isinstance(out_channels, str):
            if out_channels.startswith('x'):
                out_channels = int(float(out_channels[1:]) * in_channels)
            else:
                raise ValueError(
                    'The string `out_channels` should start with a "x", e.g.'
                    ' "x2" or "x3".'
                )
        else:
            out_channels = out_channels or in_channels

        self._residue = nn.Sequential(
            Conv2dBlock(
                in_channels, out_channels, 3, stride,
                padding=1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                out_channels, out_channels, 3,
                padding=1,
                padding_mode=padding_mode,
                mode='cn'
            )
        )
        if isinstance(stride, int):
            s_h = s_w = stride
        else:
            s_h, s_w = stride

        if in_channels == out_channels and 1 == s_h and 1 == s_w:
            self._shortcut = nn.Identity()
        else:
            self._shortcut = Conv2dBlock(
                in_channels, out_channels, 1, stride, mode='cn'
            )
        self._act = _activation(activation, activation_kw)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._act(self._shortcut(x) + self._residue(x))


class Bottleneck(nn.Module):
    r'''Bottleneck for ResNet.

    a((cn(cna(cna(x)))) + cn(x))

    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the number of output channels. If it is a `None`, it is
        equal to the `in_channels`. If it is a string starting with a "x", the
        number of out channels is `int(float(out_channels[1:]) * in_channels)`.
    - stride: stride of the convolution. Its length should be 2 if it is a
        sequence.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, out_channels, H', W')`, where
        `H' = floor((H -1) / stride) + 1` and
        `W' = floor((W -1) / stride) + 1`.

    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[Union[int, str]] = None,
        stride: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        if isinstance(out_channels, str):
            if out_channels.startswith('x'):
                out_channels = int(float(out_channels[1:]) * in_channels)
            else:
                raise ValueError(
                    'The string `out_channels` should start with a "x", e.g.'
                    ' "x2" or "x3".'
                )
        else:
            out_channels = out_channels or in_channels

        c = out_channels // 4
        self._residue = nn.Sequential(
            Conv2dBlock(
                in_channels, c, 1,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                c, c, 3, stride,
                padding=1,
                groups=groups,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(c, out_channels, 1, mode='cn')
        )

        if isinstance(stride, int):
            s_h = s_w = stride
        else:
            s_h, s_w = stride
        if in_channels == out_channels and 1 == s_h and 1 == s_w:
            self._shortcut = nn.Identity()
        else:
            self._shortcut = Conv2dBlock(
                in_channels, out_channels, 1, stride, mode='cn'
            )

        self._act = _activation(activation, activation_kw)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._act(self._shortcut(x) + self._residue(x))
