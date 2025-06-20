from typing import Sequence, Union

from torch import nn

from ..typing import TorchTensor, TorchFloat
from ..nn.conv import conv2d


class FFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expansion: int = 4,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        *args, **kwargs
    ):
        super().__init__()
        c = in_channels * expansion
        self._ffn = nn.Sequential(
            nn.Conv2d(in_channels, c, 1),
            conv2d(c, c, 3, 1, 1, bias=True, padding_mode=padding_mode),
            nn.GELU(),
            nn.Conv2d(c, in_channels, 1)
        )

    def forward(self, x: TorchTensor[TorchFloat]):
        return self._ffn(x)
