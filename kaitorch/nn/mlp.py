from typing import Any, Dict, Optional

from torch import nn

from ..typing import TorchTensor, TorchFloat
from .activation import activation as _activation
from .normalization import normalization as _normalization


class Linear1dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        normalization: str = 'batchnorm1d',
        normalization_kw: Optional[Dict[str, Any]] = None,
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        mode: str = 'lna'  # Linear-Normalization-Activation
    ) -> None:
        r'''
        Block consists of Linear, 1D normalization layer or activation layer.

        ### Args:
            - in_channels: number of channels in the input.
            - out_channels: number of channels produced by the linear.
            - bias: Whether to add a learnable bias to the output.
            - normalization: `batchnorm1d`, `layernorm` or `groupnrom`.
            - normalization_kw: arguments of normalization.
            - activation: `relu`, `leakyrelu` or other activation.
            - activation_kw: arguments of activation.
            - mode: `lna`, `ln`, `nal`, `l`, `lnalna` or other combination of
                `l` (linear), `n` (normalization) or `a` (activation).

        ### Methods:
            - forward

        '''
        super().__init__()
        layers = []
        for m in mode:
            if 'l' == m:
                layers.append(
                    nn.Linear(in_channels, out_channels, bias)
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
                raise ValueError('mode comprised of "l", "n" and "a" wanted')
        self._layers = nn.Sequential(*layers)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._layers(x)
