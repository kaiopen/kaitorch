r'''
Efficient ConvNet for Real-time Semantic Segmentation

'''
from typing import Any, Dict, Optional, Union, Sequence

from torch import nn

from kaitorch.typing import TorchTensor, TorchFloat
from ..nn.activation import activation as _activation
from ..nn.conv import Conv2dBlock


class NonBottleneck1D(nn.Module):
    r'''

    a((cn(cna(x))) + cn(x))

    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the number of output channels. If it is a `None`, it is
        equal to the `in_channels`. If it is a string starting with a "x", the
        number of out channels is `int(float(out_channels[1:]) * in_channels)`.
    - stride: the stride of the convolution. Its length should be 2 if it is a
        sequence.
    - dilation: the spacing between kernel elements. Its length should be 2 if
        it is a sequence.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation function.
    - activation_kw: the arguments to the activation function.
    - dropout

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: a feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - A feature map. Its shape is `(B, out_channels, floor((H -1) / stride) +
        1, floor((W -1) / stride) + 1)`.

    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[Union[int, str]] = None,
        stride: Union[int, Sequence[int]] = 1,
        dilation: Union[int, Sequence[int]] = 1,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        dropout: float = 0.,
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

        if isinstance(stride, int):
            s_h = s_w = stride
        else:
            s_h, s_w = stride

        if isinstance(dilation, int):
            d_h = d_w = dilation
        else:
            d_h, d_w = dilation

        self._residue = nn.Sequential(
            Conv2dBlock(
                in_channels, out_channels,
                kernel_size=(3, 1),
                stride=(s_h, 1),
                padding=(1, 0),
                bias=True,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw,
                mode='ca'
            ),
            Conv2dBlock(
                out_channels, out_channels,
                kernel_size=(1, 3),
                stride=(1, s_w),
                padding=(0, 1),
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                out_channels, out_channels,
                kernel_size=(3, 1),
                stride=(s_h, 1),
                padding=(1 * d_h, 0),
                dilation=(d_h, 1),
                bias=True,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw,
                mode='ca'
            ),
            Conv2dBlock(
                out_channels, out_channels,
                kernel_size=(1, 3),
                stride=(1, s_w),
                padding=(0, 1 * d_w),
                dilation=(1, d_w),
                padding_mode=padding_mode,
                mode='cn'
            ),
            nn.Dropout2d(dropout)
        )

        if (
            out_channels is None
            or 'x1' == out_channels
            or in_channels == out_channels
        ) and 1 == s_h and 1 == s_w:
            self._shortcut = nn.Identity()
        else:
            self._shortcut = Conv2dBlock(
                in_channels, out_channels, 1, stride,
                mode='cn'
            )

        self._act = _activation(activation, activation_kw)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._act(self._shortcut(x) + self._residue(x))
