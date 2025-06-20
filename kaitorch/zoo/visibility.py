r'''
What You See is What You Get: Exploiting Visibility for 3D Object Detection

'''

from typing import Sequence

import torch
from torch import nn

from ..typing import TorchTensor, TorchFloat, TorchInt64


class Visibility(nn.Module):
    r'''

    #### Args:
    - size: size of visibility map. It shouLd be in the form of `(C, R, T)`
        where `R` is the number of distance groups and `T` is the number of
        angle groups.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - rho: rhos of occupied points. Its shape should be `(N,)`.
    - theta: thetas of occupied points. Its shape should be `(N,)`.
    - f: features of occupied points. Its shape should be `(N,)`.

    #### Returns:
    - Visibility map. Its shape is `(C, R, T)`. For each unit of the map, it is
        free if it is a positive number. It is occupied if a zero. Otherwise,
        it is occluded. The number is from `-(R - 1)` to `R`. The absolute
        value is the distance from the occupied.

    '''
    def __init__(self, size: Sequence[int]) -> None:
        super().__init__()
        self._size = size
        r = size[1]
        self.register_buffer(
            '_rp', torch.arange(r + 1, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            '_r', torch.arange(r, dtype=torch.float), persistent=False
        )

    def forward(
        self,
        rho: TorchTensor[TorchInt64],  # (N,)
        theta: TorchTensor[TorchInt64],  # (N,)
        f: TorchTensor[TorchInt64]  # (N,)
    ) -> TorchTensor[TorchFloat]:
        occ = torch.full(self._size, -1, device=rho.device)
        occ[f, rho, theta] = rho

        ids = torch.argmax(occ, dim=1, keepdim=True)  # the farest points
        ids[torch.all(-1 == occ, dim=1, keepdim=True)] = -1

        # (C, 1, T) - (1, R, 1) = (C, R, T)
        return self._rp[ids] - self._r.reshape(1, -1, 1)
