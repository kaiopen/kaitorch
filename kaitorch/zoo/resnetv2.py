r'''
Deep Residual Learning for Image Recognition

'''
from typing import Any, Dict, Sequence, Union

from torch import nn

from ..nn.activation import activation as _activation
from ..nn.conv import Conv2dBlock, conv2d
from ..nn.normalization import normalization as _normalization
from ..typing import TorchTensor, TorchFloat


class Bottleneck(nn.Module):
    r'''Bottleneck for ResNet v2.

    y = na(x)
    c(y) + nac(nac(c(y))) or x + nac(nac(c(y)))

    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the number of output channels.
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
        out_channels: int,
        stride: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        c = out_channels // 4
        self._ba = nn.Sequential(
            _normalization('batchnorm2d', channels=in_channels),
            _activation(activation, activation_kw)
        )
        self._residue = nn.Sequential(
            conv2d(in_channels, c, 1),
            Conv2dBlock(
                c, c, 3, stride,
                padding=1,
                groups=groups,
                padding_mode=padding_mode,
                normalization='batchnorm2d',
                normalization_kw={'num_features': c},
                activation=activation,
                activation_kw=activation_kw,
                mode='nac'
            ),
            Conv2dBlock(
                c, out_channels, 1,
                normalization='batchnorm2d',
                normalization_kw={'num_features': c},
                activation=activation,
                activation_kw=activation_kw,
                mode='nac'
            )
        )
        if 1 != stride or in_channels == out_channels:
            self._shortcut = conv2d(
                in_channels, out_channels, 1, stride
            )
        else:
            self._shortcut = None

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        out = self._ba(x)
        if self._shortcut is None:
            return x + self._residue(x)
        return self._shortcut(out) + self._residue(x)
