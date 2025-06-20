from typing import Any, Dict, Optional, Union

from torch import nn

from ..typing import TorchTensor, TorchFloat
from .activation import activation as _activation
from .normalization import normalization as _normalization


class LinearBlock(nn.Module):
    r'''Linear block comprised of some linear layers, 1D normalization layers
    or activation layers.

    #### Args:
    - in_channels: the number of input channels.
    - out_channels: the number of output channels produced by the linear. If it
        is a `None`, it is equal to the `in_channels`. If it is a string
        starting with a "x", the number of out channels is
        `int(float(out_channels[1:]) * in_channels)`.
    - bias: Whether to add a learnable bias to the output.
    - normalization: `batchnorm1d`, `layernorm` or `groupnrom`.
    - normalization_kw: the arguments to the normalization function.
    - activation: `relu`, `leakyrelu` or other activation function.
    - activation_kw: the arguments to the activation function.
    - mode: `lna`, `ln`, `nal`, `l`, `lnalna` or other combination of `l`
        (linear), `n` (normalization) or `a` (activation).

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
        bias: bool = False,
        normalization: str = 'batchnorm1d',
        normalization_kw: Optional[Dict[str, Any]] = None,
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        mode: str = 'lna',  # Linear-Normalization-Activation
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
            if 'l' == m:
                self._layers.append(
                    nn.Linear(
                        in_features=in_channels,
                        out_features=out_channels,
                        bias=bias,
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
                    'A `mode` comprised of "l", "n" and "a" is acceptable.'
                )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._layers(x)
