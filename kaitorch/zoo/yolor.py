r'''
You Only Learn One Representation: Unified Network for Multiple Tasks

'''
import torch
from torch import nn

from kaitorch.typing import TorchTensor, TorchFloat


class ImplicitA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mean: float = 0,
        std: float = 0.02,
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._implicit = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        nn.init.normal_(self._implicit, mean=mean, std=std)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._implicit + x


class ImplicitM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mean: float = 1,
        std: float = 0.02,
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._implicit = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        nn.init.normal_(self._implicit, mean=mean, std=std)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._implicit * x
