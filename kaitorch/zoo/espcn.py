from typing import Optional, Sequence, Union

from torch import nn

from ..typing import TorchTensor, TorchFloat
from ..nn.conv import conv2d


class SubPixel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        upscale_factor: int,
        out_channels: Optional[int] = None,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._layer = nn.Sequential(
            conv2d(
                in_channels,
                (out_channels or in_channels) * upscale_factor ** 2,
                3, 1, 1,
                padding_mode=padding_mode
            ),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._layer(x)
