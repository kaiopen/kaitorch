r'''
Searching for MobileNetV3

'''

from typing import Any, Dict, Sequence, Union

from torch import nn

from ..nn.activation import activation as _activation
from ..nn.conv import Conv2dBlock
from ..typing import TorchTensor, TorchFloat
from .senet import SqueezeExcitation


class Bottleneck(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - hidden_channels: the number of the channels of the hidden layers. It
        should be larger than `in_channels` and `out_channels` referred in the
        paper.
    - out_channels: the number of output channels.
    - stride: the stride of the convolution. Its length should be 2 if it is a
        sequence.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: the arguments of the activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: a feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - A feature map. Its shape is `(B, out_channels, floor((H - 1) / stride) +
        1, floor((W - 1) / stride) + 1)`.

    '''
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        stride: int = 1,
        padding_mode: Union[str, Sequence[str]] = 'zeros',  # (M_H, M_W)
        activation: str = 'hardswish',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._residue = nn.Sequential(
            Conv2dBlock(
                in_channels, hidden_channels, 1,
                activation=activation, activation_kw=activation_kw
            ),
            Conv2dBlock(
                hidden_channels, hidden_channels, 3, stride,
                padding=1, groups=hidden_channels, padding_mode=padding_mode,
                mode='cn'
            ),
            Conv2dBlock(
                hidden_channels, out_channels, 1, 1,
                activation=activation, activation_kw=activation_kw
            )
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._residue(x)


class ResBottleneck(Bottleneck):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - hidden_channels: number of channels in the hidden layer. It should be
        larger than `in_channels` and `out_channels` referred in the paper.
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
    - Feature map. Its shape is `(B, in_channels, H, W)`.

    '''
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',  # (M_H, M_W)
        activation: str = 'hardswish',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=in_channels,
            stride=1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw,
            *args, **kwargs
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return x + self._residue(x)


class BottleneckSE(nn.Module):
    r'''

    #### Args:
    - in_channels: number of channels in the input
    - hidden_channels: number of channels in the hidden layer. It should be
        larger than `in_channels` and `out_channels` referred in the paper.
    - out_channels: the number of output channels.
    - stride: stride of the convolution.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    - reduction: reduction ratio for squeeze excitation.
    - activation_se_1: : `relu`, `leakyrelu` or other activation after the
        first linear layer in squeeze excitation.
    - activation_kw_se_1: arguments of activation.
    - activation_se_2: `relu`, `leakyrelu` or other activation after the last
        linear layer in squeeze excitation.
    - activation_kw_se_2: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, out_channels, floor((H - 1) / stride) + 1,
        floor((W - 1) / stride) + 1)`.

    '''
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        stride: int = 1,
        padding_mode: Union[str, Sequence[str]] = 'zeros',  # (M_H, M_W)
        activation: str = 'hardswish',
        activation_kw: Dict[str, Any] = {'inplace': True},

        reduction: int = 4,
        activation_se_1: str = 'relu',
        activation_kw_se_1: Dict[str, Any] = {'inplace': True},
        activation_se_2: str = 'hardsigmoid',
        activation_kw_se_2: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._residue = nn.Sequential(
            Conv2dBlock(
                in_channels, hidden_channels, 1,
                activation=activation, activation_kw=activation_kw
            ),
            Conv2dBlock(
                hidden_channels, hidden_channels, 3, stride,
                padding=1, groups=hidden_channels, padding_mode=padding_mode,
                activation=activation, activation_kw=activation_kw
            ),
            SqueezeExcitation(
                hidden_channels,
                reduction,
                activation_1=activation_se_1,
                activation_kw_1=activation_kw_se_1,
                activation_2=activation_se_2,
                activation_kw_2=activation_kw_se_2
            ),
            _activation(activation, activation_kw),
            Conv2dBlock(hidden_channels, out_channels, 1, 1, mode='cn')
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._residue(x)


class ResBottleneckSE(BottleneckSE):
    r'''

    #### Args:
    - in_channels: number of channels in the input
    - hidden_channels: number of channels in the hidden layer. It should be
        larger than `in_channels` and `out_channels` referred in the paper.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    - reduction: reduction ratio for squeeze excitation.
    - activation_se_1: : `relu`, `leakyrelu` or other activation after the
        first linear layer in squeeze excitation.
    - activation_kw_se_1: arguments of activation.
    - activation_se_2: `relu`, `leakyrelu` or other activation after the last
        linear layer in squeeze excitation.
    - activation_kw_se_2: arguments of activation.

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
        hidden_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',  # (M_H, M_W)
        activation: str = 'hardswish',
        activation_kw: Dict[str, Any] = {'inplace': True},

        reduction: int = 4,
        activation_se_1: str = 'relu',
        activation_kw_se_1: Dict[str, Any] = {'inplace': True},
        activation_se_2: str = 'hardsigmoid',
        activation_kw_se_2: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=in_channels,
            stride=1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw,

            reduction=reduction,
            activation_se_1=activation_se_1,
            activation_kw_se_1=activation_kw_se_1,
            activation_se_2=activation_se_2,
            activation_kw_se_2=activation_kw_se_2,
            *args, **kwargs
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return x + self._residue(x)
