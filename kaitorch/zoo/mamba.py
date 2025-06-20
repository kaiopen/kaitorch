import math

import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm import Mamba as _Mamba

from ..typing import TorchTensor, TorchFloat


class MambaBlock(nn.Module):
    r'''A modification of https://github.com/alxndrTL/mamba.py.

    #### Args:
    - in_channels: the number of input channels.
    - state_channels: the number of the channels of the implicit latent state.
    - expansion: expansion factor.
    - kernel_size: the size of the convolution kernel

    '''
    def __init__(
        self,
        in_channels: int,
        state_channels: int = 16,
        expansion: int = 2,
        kernel_size: int = 4,
        *args, **kwargs
    ):
        super().__init__()
        self._c = in_channels * expansion  # ED
        self._state = state_channels  # N
        self._rank = math.ceil(in_channels / 16)

        # Projects block input from D to 2 * ED (two branches)
        self._proj_inp = nn.Linear(in_channels, 2 * self._c, bias=False)
        # Projects block output from ED back to D
        self._proj_out = nn.Linear(self._c, in_channels, bias=False)

        self._conv = nn.Conv1d(
            self._c, self._c, kernel_size, 1,
            padding=kernel_size - 1,
            groups=self._c,
            bias=True
        )

        self._A = nn.Parameter(
            torch.log(
                torch.arange(
                    1, state_channels + 1, dtype=torch.float
                ).repeat(self._c, 1)
            )
        )
        self._A._no_weight_decay = True

        # See Appendix A for detail.
        self._D = nn.Parameter(torch.ones(self._c))
        self._D._no_weight_decay = True

        # Projects x to input-dependent delta, B, C.
        self._proj_x = nn.Linear(
            self._c, self._rank + 2 * state_channels, bias=False
        )
        self._proj_delta = nn.Linear(self._rank, self._c, bias=True)

        bound = self._rank ** -0.5
        nn.init.uniform_(self._proj_delta.weight, -bound, bound)

        dt = torch.exp(
            torch.rand(self._c) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        with torch.no_grad():
            self._proj_delta.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        x, res = self._proj_inp(x).chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)
        return self._proj_out(
            self.ssm(
                F.silu(
                    self._conv(
                        x.transpose(1, 2)
                    )[:, :, :x.shape[1]].transpose(1, 2)
                )
            ) * F.silu(res)
        )  # (B, L, D)

    def ssm(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        delta, B, C = torch.split(
            self._proj_x(x), [self._rank, self._state, self._state], dim=-1
        )  # (B, L, R), (B, L, state_channels), (B, L, state_channels)
        # (B, L, ED, 1)
        delta = F.softplus(self._proj_delta(delta)).unsqueeze(-1)

        # (B, L, ED, 1) * (ED, state_channels) => (B, L, ED, state_channels)
        A = torch.exp(-delta * torch.exp(self._A))
        # (B, L, ED, 1) * (B, L, 1, N) * (B, L, ED, 1) => (B, L, ED, N)
        B = delta * B.unsqueeze(2) * x.unsqueeze(-1)

        h = torch.zeros((x.shape[0], self._c, self._state), device=A.device)
        # h = B[:, 0]
        hs = [h,]
        for t in range(1, x.shape[1]):
            h = A[:, t] * h + B[:, t]
            hs.append(h)
        # (B, L, ED, state_channels) @ (B, L, state_channels, 1)
        # -> (B, L, ED, 1)
        return (
            torch.stack(hs, dim=1) @ C.unsqueeze(-1)
        ).squeeze(3) + self._D * x


class Mamba(_Mamba):
    r'''The official interface.

    #### Args:
    - in_channels: the number of input channels.
    - state_channels: the number of the channels of the implicit latent state.
    - expansion: expansion factor.
    - kernel_size: the size of the convolution kernel

    '''
    def __init__(
        self,
        in_channels: int,
        state_channels: int = 16,
        expansion: int = 2,
        kernel_size: int = 4,
        *args, **kwargs
    ) -> None:
        super().__init__(
            d_model=in_channels,
            d_state=state_channels,
            d_conv=kernel_size,
            expand=expansion
        )
