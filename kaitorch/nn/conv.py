from typing import Any, Dict, Optional, Sequence, Union

from torch import nn

from ..typing import TorchFloat, TorchTensor
from .activation import activation as _activation
from .normalization import normalization as _normalization
from .padding import PAD2D
from .utils import _ntuple


def conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,  # (P_L,)
    dilation: Union[int, Sequence[int]] = 1,
    groups: int = 1,
    bias: bool = False,
    padding_mode: Union[str, Sequence[str]] = 'zeros',  # (M_L,)
) -> nn.Module:
    r'''

    ### Args:
        - in_channels: number of channels in the input.
        - out_channels: number of channels produced by the convolution.
        - kernel_size: size of the convolving kernel. Its length should be
            1 if it is a sequence.
        - stride: stride of the convolution. Its length should be 1 if it
            is a sequence.
        - padding: padding added to dimension of `H` and `W` of the input.
            Its length should be 1 if it is a sequence.
        - dilation: spacing between kernel elements. Its length should be 1
            if it is a sequence.
        - groups: number of blocked connections from input channels to
            output channels.
        - bias: Whether to add a learnable bias to the output.
        - padding_mode: `zeros`, `reflect`, `replicate`, `circular` working on
            dimension `L` of the input. Its length should be 1 if it is a
            sequence.

    ### Returns:
        - 1D convolution.

    '''
    if isinstance(padding_mode, (tuple, list)):
        if 1 == len(padding_mode):
            padding_mode = padding_mode[0]
        else:
            raise ValueError(
                'padding_mode in sequence with length == 1 or a string wanted.'
            )
    return nn.Conv1d(
        in_channels, out_channels,
        kernel_size, stride, padding,
        dilation, groups, bias, padding_mode
    )


def conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,  # (P_H, P_W)
    dilation: Union[int, Sequence[int]] = 1,
    groups: int = 1,
    bias: bool = False,
    padding_mode: Union[str, Sequence[str]] = 'zeros'  # (M_H, M_W)
) -> nn.Module:
    r'''

    ### Args:
        - in_channels: number of channels in the input.
        - out_channels: number of channels produced by the convolution.
        - kernel_size: size of the convolving kernel. Its length should be
            2 if it is a sequence.
        - stride: stride of the convolution. Its length should be 2 if it
            is a sequence.
        - padding: padding added to dimension of `H` and `W` of the input.
            Its length should be 2 if it is a sequence.
        - dilation: spacing between kernel elements. Its length should be 2
            if it is a sequence.
        - groups: number of blocked connections from input channels to
            output channels.
        - bias: Whether to add a learnable bias to the output.
        - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
            combination working on dimension `H` and `W` of the input. Its
            length should be less than or equal to 2 if it is a sequence.

    ### Returns:
        - 2D convolution.

    '''
    if isinstance(padding_mode, (tuple, list)):
        len_m = len(padding_mode)
        if 2 == len_m:
            padding = _ntuple(2)(padding)
            if len(padding) > 2:
                raise ValueError(
                    'padding in sequence with length <= 2 or a number wanted.'
                )
            return nn.Sequential(
                PAD2D[padding_mode[1]](
                    (padding[1], padding[1], 0, 0)  # (P_L, P_R, P_T, P_B)
                ),  # pad left and right first
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride,
                    padding=(padding[0], 0),  # (P_H, P_W) pad top and bottm
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    padding_mode=padding_mode[0]
                )
            )
        if 1 == len_m:
            padding_mode = padding_mode[0]
        else:
            raise ValueError(
                'padding_mode in sequence with length <= 2 or a string wanted.'
            )
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size, stride, padding,
        dilation, groups, bias, padding_mode
    )


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
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

        mode: str = 'cna'  # Convolution-Normalization-Activation
    ) -> None:
        r'''
        Block consists of 1D convolution layer, 1D normalization layer or
        activation layer.

        ### Args:
            - in_channels: number of channels in the input.
            - out_channels: number of channels produced by the convolution.
            - kernel_size: size of the convolving kernel. Its length should be
                1 if it is a sequence.
            - stride: stride of the convolution. Its length should be 1 if it
                is a sequence.
            - padding: padding added to dimension of `L` of the input. Its
                length should be 1 if it is a sequence.
            - dilation: spacing between kernel elements. Its length should be 1
                if it is a sequence.
            - groups: number of blocked connections from input channels to
                output channels.
            - bias: Whether to add a learnable bias to the output.
            - padding_mode: `zeros`, `reflect`, `replicate`, `circular` working
                on dimension `L` of the input. Its length should be 1 if it is
                a sequence.
            - normalization: `batchnorm1d`, `layernorm` or `groupnrom`.
            - normalization_kw: arguments of normalization.

            - activation: `relu`, `leakyrelu` or other activation.
            - activation_kw: arguments of activation.

            - mode: `cna`, `cn`, `nac`, `c`, `cnacna` or other combination of
                `c` (convolution), `n` (normalization) or `a` (activation).

        ### Methods:
            - forward

        '''
        super().__init__()
        layers = []
        for m in mode:
            if 'c' == m:  # Conv1d
                layers.append(
                    conv1d(
                        in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias, padding_mode
                    )
                )
            elif 'n' == m:
                layers.append(
                    _normalization(
                        normalization, normalization_kw, out_channels
                    )
                )
            elif 'a' == m:
                layers.append(_activation(activation, activation_kw))
            else:
                raise ValueError('mode comprised of "c", "n" and "a" wanted')
        self._layers = nn.Sequential(*layers)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._layers(x)


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
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

        mode: str = 'cna'  # Convolution-Normalization-Activation
    ) -> None:
        r'''
        Block consists of 2D convolution layer, 2D normalization layer or
        activation layer.

        ### Args:
            - in_channels: number of channels in the input.
            - out_channels: number of channels produced by the convolution.
            - kernel_size: size of the convolving kernel. Its length should be
                2 if it is a sequence.
            - stride: stride of the convolution. Its length should be 2 if it
                is a sequence.
            - padding: padding added to dimension of `H` and `W` of the input.
                Its length should be 2 if it is a sequence.
            - dilation: spacing between kernel elements. Its length should be 2
                if it is a sequence.
            - groups: number of blocked connections from input channels to
                output channels.
            - bias: Whether to add a learnable bias to the output.
            - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or
                their combination working on dimension `H` and `W` of the
                input. Its length should be less than or equal to 2 if it is a
                sequence.

            - normalization: `batchnorm2d`, `layernorm` or `groupnrom`.
            - normalization_kw: arguments of normalization.

            - activation: `relu`, `leakyrelu` or other activation.
            - activation_kw: arguments of activation.

            - mode: `cna`, `cn`, `nac`, `c`, `cnacna` or other combination of
                `c` (convolution), `n` (normalization) or `a` (activation).

        ### Methods:
            - forward

        '''
        super().__init__()
        layers = []
        for m in mode:
            if 'c' == m:  # Conv2d
                layers.append(
                    conv2d(
                        in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias, padding_mode
                    )
                )
            elif 'n' == m:
                layers.append(
                    _normalization(
                        normalization, normalization_kw, out_channels
                    )
                )
            elif 'a' == m:
                layers.append(_activation(activation, activation_kw))
            else:
                raise ValueError('mode comprised of "c", "n" and "a" wanted')
        self._layers = nn.Sequential(*layers)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._layers(x)
