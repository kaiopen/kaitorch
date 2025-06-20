from typing import Any, Dict, Sequence, Union

from torch import nn

from ..typing.utils import TorchTensor, TorchFloat
from ..nn.activation import activation as _activation
from ..nn.conv import Conv2dBlock


class RepConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        if in_channels == out_channels:
            self._identity = nn.BatchNorm2d(out_channels)
        else:
            self._identity = nn.Identity()

        self._dense = Conv2dBlock(
            in_channels, out_channels, 3, 1, 1,
            padding_mode=padding_mode,
            mode='cn'
        )
        self._one = Conv2dBlock(
            in_channels, out_channels, 1, 1, 0,
            padding_mode=padding_mode,
            mode='cn'
        )

        self._act = _activation(activation, activation_kw)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._act(self._dense(x) + self._one(x) + self._identity(x))
