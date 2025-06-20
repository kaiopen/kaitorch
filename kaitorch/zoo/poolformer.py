r'''
MetaFormer is Actually What You Need for Vision

'''
from typing import Any, Dict, Optional, Sequence, Union

from torch import nn

from ..nn.conv import Conv2dBlock, conv2d
from ..nn.pooling import avg_pool2d
from ..nn.normalization import normalization as _normalization
from ..typing import TorchTensor, TorchFloat
from .cait import LayerScale
from .droppath import DropPath


class PatchEmbedding(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the number of output channels.
    - kernel_size: size of the convolving kernel. Its length should be 2 if it
        is a sequence.
    - stride: stride of the convolution. Its length should be 2 if it is a
        sequence.
    - padding: padding added to dimension of `H` and `W` of the input. Its
        length should be 2 if it is a sequence.
    - dilation: spacing between kernel elements. Its length should be 2 if it
        is a sequence.
    - groups: number of blocked connections from input channels to output
        channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, out_channels, H', W')`, where
        `H' = floor((H + 2 * padding - kernel_size) / stride) + 1` and
        `W' = floor((W + 2 * padding - kernel_size) / stride) + 1`.

    '''
    def __init__(
        self,
        in_channels,
        out_channels,
        size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int]] = 0,  # (P_H, P_W)
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        padding_mode: Union[str, Sequence[str]] = 'zeros',  # (M_H, M_W)
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._emb = conv2d(
            in_channels, out_channels,
            size, stride, padding,
            dilation, groups, False, padding_mode
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._emb(x)


class Pooling(nn.Module):
    r'''

    #### Args:
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.

    #### Methods;
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, C, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, C, H, W)`.

    '''
    def __init__(
        self,
        padding_mode: Union[str, Sequence[str]] = 'zeros',  # (M_H, M_W)
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._pool = avg_pool2d(
            3, 1, 1,
            count_include_pad=False,
            padding_mode=padding_mode
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._pool(x) - x


class MLP(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - hidden_channels: number of hidden channels.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.
    - dropout: dropout probability.

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
        activation: str = 'gelu',
        activation_kw: Dict[str, Any] = {},
        dropout: float = 0.,
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._mlp = nn.Sequential(
            Conv2dBlock(
                in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                activation=activation,
                activation_kw=activation_kw,
                mode='ca'
            ),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, in_channels, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._mlp(x)


class PoolFormerBlock(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.
    - ratio: expansion ratio on hidden channels in MLP.
    - scale: initial layer scale. No layer scale will be used if `None`.
    - dropout: dropout probability.
    - droppath: droppath probability.

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
        padding_mode: Union[str, Sequence[str]] = 'zeros',  # (M_H, M_W)
        activation: str = 'gelu',
        activation_kw: Dict[str, Any] = {},
        ratio: int = 4,
        scale: Optional[float] = 1e-5,
        dropout: float = 0.,
        droppath: float = 0.,
        *args, **kwargs
    ) -> None:
        super().__init__()
        norm_kw = {
            'num_groups': 1,
            'num_channels': in_channels
        }

        self._norm_att = _normalization('groupnorm', norm_kw)
        self._pool = Pooling(in_channels, padding_mode, scale)
        self._scale_att = nn.Identity() \
            if scale is None else LayerScale(in_channels, scale)

        self._norm_mlp = _normalization('groupnorm', norm_kw)
        self._mlp = MLP(
            in_channels, in_channels * ratio,
            activation, activation_kw,
            dropout
        )
        self._scale_mlp = nn.Identity() \
            if scale is None else LayerScale(in_channels, scale)
        self._droppath = DropPath(droppath)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        x = x + self._droppath(
            self._scale_att(
                self._pool(self._norm_att(x)).permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)
        )
        return x + self._droppath(
            self._scale_mlp(
                self._mlp(self._norm_mlp(x)).permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)
        )
