from typing import Sequence, Union

import torch
from torch import nn
import torch.nn.functional as F

from kaitorch.typing import TorchTensor, TorchFloat
from kaitorch.nn.conv import conv2d


class SRU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        groups: int,
        threshold: float = 0.5,
        *args, **kwargs
    ):
        super().__init__()
        self._gn = nn.GroupNorm(num_groups=groups, num_channels=in_channels)
        self._t = threshold

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        g = self._gn(x)
        w = self._gn.weight / sum(self._gn.weight)
        w = F.sigmoid(g * w.view(1, -1, 1, 1))

        mask = w > self._t
        c = x.shape[1] // 2
        return torch.cat(
            (
                torch.add(
                    *torch.split(
                        torch.where(mask, torch.ones_like(w), w) * x, c, dim=1
                    )
                ),
                torch.add(
                    *torch.split(
                        torch.where(mask, torch.zeros_like(w), w) * x, c, dim=1
                    )
                )
            ),
            dim=1
        )


class CRU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Sequence[int]] = 3,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        *args, **kwargs
    ):
        super().__init__()
        self._in_channel = in_channels
        up = in_channels // 2
        low = in_channels - up
        self._c = (up, low)

        up_sq = up // 2
        low_sq = low // 2

        self._sq_up = nn.Conv2d(up, up_sq, 1, 1, 0, bias=False)
        self._sq_low = nn.Conv2d(low, low_sq, 1, 1, 0, bias=False)

        self._gwc = conv2d(
            up_sq, in_channels, kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=2,
            padding_mode=padding_mode
        )
        self._pwc_up = nn.Conv2d(up_sq, in_channels, 1, 1, 0, bias=False)

        self._pwc_low = nn.Conv2d(
            low_sq, in_channels - low_sq, 1, 1, 0, bias=False
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        up, low = torch.split(x, self._c, dim=1)
        up = self._sq_up(up)
        low = self._sq_low(low)

        x = torch.cat(
            (
                self._gwc(up) + self._pwc_up(up),
                torch.cat((self._pwc_low(low), low), dim=1)
            ),
            dim=1
        )
        return torch.add(
            *torch.split(
                F.softmax(torch.mean(x, dim=(2, 3), keepdim=True), dim=1) * x,
                self._in_channel,
                dim=1
            )
        )


class SCConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Sequence[int]] = 3,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        groups: int = 4,
        threshold: float = 0.5,
        *args, **kwargs
    ):
        super().__init__()
        self._sru = SRU(in_channels, groups, threshold)
        self._cru = CRU(in_channels, kernel_size, padding_mode)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._cru(self._sru(x))
