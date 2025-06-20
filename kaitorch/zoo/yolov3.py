r'''
YOLOv3: An Incremental Improvement

'''
from typing import Any, Dict, Optional, Sequence, Union

from torch import nn

from ..typing import TorchTensor, TorchFloat
from ..nn.conv import Conv2dBlock


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: Optional[str] = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        shortcut: bool = False,
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._residue = nn.Sequential(
            Conv2dBlock(
                in_channels, out_channels, 1, 1, 0,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                out_channels, out_channels, 3, 1, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )
        self._add = shortcut and in_channels == out_channels

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return x + self._residue(x) if self._add else self._residue(x)
