from typing import Any, Dict, Optional, Sequence, Union

from torch import nn

from ..typing import TorchFloat, TorchTensor
from ..data.utils import tuple_2
from .activation import activation as _activation
from .normalization import normalization as _normalization
from .padding import PAD2D


def conv1d(
    in_channels: int,
    out_channels: Optional[Union[int, str]] = None,
    kernel_size: Union[int, Sequence[int]] = 1,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,  # (P_L,)
    dilation: Union[int, Sequence[int]] = 1,
    groups: int = 1,
    bias: bool = False,
    padding_mode: Union[str, Sequence[str]] = 'zeros',  # (M_L,)
    device: Optional[Any] = None,
    dtype: Optional[Any] = None
) -> nn.Module:
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the number of output channels produced by the convolution.
        If it is a `None`, it is equal to the `in_channels`. If it is a string
        starting with a "x", the number of out channels is
        `int(float(out_channels[1:]) * in_channels)`.
    - kernel_size: the size of the convolution kernel. Its length should be 1
        if it is a sequence.
    - stride: the stride of the convolution. Its length should be 1 if it is a
        sequence.
    - padding: the padding added to the dimensions `H` and `W` of the input.
        Its length should be 1 if it is a sequence.
    - dilation: the spacing between kernel elements. Its length should be 1 if
        it is a sequence.
    - groups: the number of blocked connections from input channels to output
        channels.
    - bias: Whether to add a learnable bias to the output.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` working on
        dimension `L` of the input. Its length should be 1 if it is a sequence.

    #### Returns:
    - 1D convolution.

    '''
    if isinstance(out_channels, str):
        if out_channels.startswith('x'):
            out_channels = int(float(out_channels[1:]) * in_channels)
        else:
            raise ValueError(
                'The string `out_channels` should start with a "x", e.g. "x2"'
                ' or "x3".'
            )
    else:
        out_channels = out_channels or in_channels

    if isinstance(padding_mode, (tuple, list)):
        if 1 == len(padding_mode):
            padding_mode = padding_mode[0]
        else:
            raise ValueError(
                'The length of the sequence `padding_mode` should be 1.'
            )

    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        device=device,
        dtype=dtype
    )


def conv2d(
    in_channels: int,
    out_channels: Optional[Union[int, str]] = None,
    kernel_size: Union[int, Sequence[int]] = 1,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,  # (P_H, P_W)
    dilation: Union[int, Sequence[int]] = 1,
    groups: int = 1,
    bias: bool = False,
    padding_mode: Union[str, Sequence[str]] = 'zeros',  # (M_H, M_W)
    device: Optional[Any] = None,
    dtype: Optional[Any] = None
) -> nn.Module:
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the number of output channels produced by the convolution.
        If it is a `None`, it is equal to the `in_channels`. If it is a string
        starting with a "x", the number of out channels is
        `int(float(out_channels[1:]) * in_channels)`.
    - kernel_size: the size of the convolution kernel. Its length should be 2
        if it is a sequence.
    - stride: the stride of the convolution. Its length should be 2 if it is a
        sequence.
    - padding: the padding added to dimensions `H` and `W` of the input. Its
        length should be less than or equal to 2 if it is a sequence.
    - dilation: the spacing between kernel elements. Its length should be 2 if
        it is a sequence.
    - groups: the number of blocked connections from input channels to output
        channels.
    - bias: Whether to add a learnable bias to the output.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.

    #### Returns:
    - 2D convolution.

    '''
    if isinstance(out_channels, str):
        if out_channels.startswith('x'):
            out_channels = int(float(out_channels[1:]) * in_channels)
        else:
            raise ValueError(
                'The string `out_channels` should start with a "x", e.g. "x2"'
                ' or "x3".'
            )
    else:
        out_channels = out_channels or in_channels

    if isinstance(padding_mode, (tuple, list)):
        len_m = len(padding_mode)
        if 2 == len_m:
            padding = tuple_2(padding)
            if len(padding) > 2:
                raise ValueError(
                    'The length of the sequence `padding` should be less than'
                    ' or equal to 2.'
                )
            return nn.Sequential(
                PAD2D[padding_mode[1]](
                    (padding[1], padding[1], 0, 0)  # (P_L, P_R, P_T, P_B)
                ),  # pad left and right first
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride,
                    padding=(padding[0], 0),  # (P_H, P_W) pad top and bottom
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    padding_mode=padding_mode[0],
                    device=device,
                    dtype=dtype
                )
            )
        if 1 == len_m:
            padding_mode = padding_mode[0]
        else:
            raise ValueError(
                'The length of the sequence `padding_mode` should be less than'
                ' or equal to 2.'
            )

    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        device=device,
        dtype=dtype
    )


class Conv1dBlock(nn.Module):
    r'''1D Convolution block comprised of some 1D convolution layers, 1D
    normalization layers or activation layers.

    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the number of output channels produced by the convolution.
        If it is a `None`, it is equal to the `in_channels`. If it is a string
        starting with a "x", the number of out channels is
        `int(float(out_channels[1:]) * in_channels)`.
    - kernel_size: the size of the convolution kernel. Its length should be 1
        if it is a sequence.
    - stride: the stride of the convolution. Its length should be 1 if it is a
        sequence.
    - padding: the padding added to dimension `L` of the input. Its length
        should be 1 if it is a sequence.
    - dilation: the spacing between kernel elements. Its length should be 1 if
        it is a sequence.
    - groups: the number of blocked connections from input channels to output
        channels.
    - bias: Whether to add a learnable bias to the output.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` working on
        dimension `L` of the input. Its length should be 1 if it is a sequence.

    - normalization: `batchnorm1d`, `layernorm` or `groupnrom`.
    - normalization_kw: the arguments to the normalization function.

    - activation: `relu`, `leakyrelu` or other activation function .
    - activation_kw: the arguments to the activation function.

    - mode: `cna`, `cn`, `nac`, `c`, `cnacna` or other combination of `c`
        (convolution), `n` (normalization) or `a` (activation).

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: a tensor. Its shape should be `(B, L, in_channels)`.

    #### Returns:
    - A tensor. Its shape is `(B, L, out_channels)`.

    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[Union[int, str]] = None,
        kernel_size: Union[int, Sequence[int]] = 1,
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int]] = 0,  # (P_L,)
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: Union[str, Sequence[str]] = 'zeros',  # (M_L,)

        normalization: str = 'batchnorm1d',
        normalization_kw: Optional[Dict[str, Any]] = None,

        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},

        mode: str = 'cna',  # Convolution-Normalization-Activation
        device: Optional[Any] = None,
        dtype: Optional[Any] = None,
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

        self._layers = nn.Sequential()
        for m in mode:
            if 'c' == m:  # Conv1d
                self._layers.append(
                    conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                        padding_mode=padding_mode,
                        device=device,
                        dtype=dtype
                    )
                )
            elif 'n' == m:
                self._layers.append(
                    _normalization(
                        name=normalization,
                        kw=normalization_kw,
                        in_channels=out_channels,
                        device=device,
                        dtype=dtype
                    )
                )
            elif 'a' == m:
                self._layers.append(_activation(activation, activation_kw))
            else:
                raise ValueError(
                    'A `mode` comprised of "c", "n" and "a" is acceptable.'
                )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._layers(x)


class Conv2dBlock(nn.Module):
    r'''2D Convolution block comprised of some 2D convolution layers, 2D
    normalization layers or activation layers.

    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the number of output channels produced by the convolution.
        If it is a `None`, it is equal to the `in_channels`. If it is a string
        starting with a "x", the number of out channels is
        `int(float(out_channels[1:]) * in_channels)`.
    - kernel_size: the size of the convolution kernel. Its length should be 2
        if it is a sequence.
    - stride: the stride of the convolution. Its length should be 2 if it is a
        sequence.
    - padding: the padding added to dimensions `H` and `W` of the input. Its
        length should be less than or equal to 2 if it is a sequence.
    - dilation: the spacing between kernel elements. Its length should be 2 if
        it is a sequence.
    - groups: the number of blocked connections from input channels to output
        channels.
    - bias: Whether to add a learnable bias to the output.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.

    - normalization: `batchnorm2d`, `layernorm` or `groupnrom`.
    - normalization_kw: the arguments to the normalization function.

    - activation: `relu`, `leakyrelu` or other activation function.
    - activation_kw: the arguments to the activation function.

    - mode: `cna`, `cn`, `nac`, `c`, `cnacna` or other combination of `c`
        (convolution), `n` (normalization) or `a` (activation).

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: a feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - A feature map. Its shape is `(B, out_channels, H, W)`.

    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[Union[int, str]] = None,
        kernel_size: Union[int, Sequence[int]] = 1,
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int]] = 0,  # (P_H, P_W)
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: Union[str, Sequence[str]] = 'zeros',  # (M_H, M_W)

        normalization: str = 'batchnorm2d',
        normalization_kw: Optional[Dict[str, Any]] = None,

        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},

        mode: str = 'cna',  # Convolution-Normalization-Activation
        device: Optional[Any] = None,
        dtype: Optional[Any] = None,
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

        self._layers = nn.Sequential()
        for m in mode:
            if 'c' == m:  # Conv2d
                self._layers.append(
                    conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                        padding_mode=padding_mode,
                        device=device,
                        dtype=dtype
                    )
                )
            elif 'n' == m:
                self._layers.append(
                    _normalization(
                        name=normalization,
                        kw=normalization_kw,
                        in_channels=out_channels,
                        device=device,
                        dtype=dtype
                    )
                )
            elif 'a' == m:
                self._layers.append(_activation(activation, activation_kw))
            else:
                raise ValueError(
                    'A `mode` comprised of "c", "n" and "a" is acceptable.'
                )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._layers(x)
