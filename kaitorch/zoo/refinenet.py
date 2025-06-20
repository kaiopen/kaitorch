r'''
ReﬁneNet: Multi-Path Reﬁnement Networks for High-Resolution Semantic
Segmentation

'''
from typing import Any, Dict, Optional, Sequence, Union
import math

from torch import nn

from ..typing import TorchTensor, TorchFloat
from ..nn.conv import Conv2dBlock


class Refine(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the number of output channels.
    - factors: interpolating scale factor. Its length should be
        `len(in_channels) - 1`. If it is a single number `f`, the factors used
        will be `[f^i for i in range(1, len(in_channels))]`.
    - mode: interpolating mode.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods;
    - forward

    ## forward
    #### Args:
    -x: feature maps. Their shapes should be `(B, in_channels[0], H, W)`,
        `(B, in_channels[1], H // factors[0], W // factors[0])` ... Feature
        maps will be upsampled except the first one.

    #### Returns:
    - Feature map. Its shape is `(B, out_channels, H, W)`.

    '''
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        factors: Union[int, Sequence[int]] = 2,
        mode: str = 'bilinear',
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: Optional[str] = 'silu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        if not isinstance(factors, (tuple, list)):
            f = int(factors)
            factors = [math.pow(f, i) for i in range(1, len(in_channels))]

        self._conv = Conv2dBlock(
            in_channels[0], out_channels, 3, 1, 1,
            padding_mode=padding_mode,
            activation=activation,
            activation_kw=activation_kw
        )
        self._refines = nn.ModuleList()
        self._refines.append(nn.Identity())
        for c, f in zip(in_channels[1:], factors):
            self._refines.append(
                nn.Sequential(
                    Conv2dBlock(
                        c, out_channels, 3, 1, 1,
                        padding_mode=padding_mode,
                        activation=activation,
                        activation_kw=activation_kw
                    ),
                    nn.Upsample(scale_factor=f, mode=mode)
                )
            )

        self._range = range(1, len(in_channels))

    def forward(
        self, x: Sequence[TorchTensor[TorchFloat]]
    ) -> TorchTensor[TorchFloat]:
        out = self._conv(x[0])
        for i in self._range:
            out = out + self._refines[i](x[i])
        return out
