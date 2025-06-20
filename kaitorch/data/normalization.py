from typing import Optional

import torch

from ..typing import TorchTensor, TorchFloat, TorchReal


def min_max_norm(
    x: TorchTensor[TorchReal],
    lower_bound: Optional[TorchTensor[TorchReal]] = None,
    upper_bound: Optional[TorchTensor[TorchReal]] = None,
    eps: float = 1e-8
) -> TorchTensor[TorchFloat]:
    r'''

    #### Args:
    - x: data to be normalized. Its shape should be `(B, L)`.
    - lower_bound: lower bound of the data. Its shape should be `(B,)` or
        `(B, L)`. If not provided, the lower bound will be calculated according
        to the input data.
    - upper_bound: upper bound of the data. Its shape should be `(B,)` or
        `(B, L)`. If not provided, the upper bound will be calculated according
        to the input data.
    - eps

    #### Returns:
    - Normalized data. Its shape is `(B, L)`.

    '''
    if lower_bound is None:
        lower_bound = torch.min(x, dim=-1)[0].view(-1, 1)
    else:
        match lower_bound.ndim:
            case 1:
                lower_bound = lower_bound.view(-1, 1)
            case 2:
                pass
            case _:
                raise ValueError(
                    '`lower_bound` with shape `(B,)` or `(B, L)` wanted'
                )

    if upper_bound is None:
        upper_bound = torch.max(x, dim=-1)[0].view(-1, 1)
    else:
        match upper_bound.ndim:
            case 1:
                upper_bound = upper_bound.view(-1, 1)
            case 2:
                pass
            case _:
                raise ValueError(
                    '`upper_bound` with shape `(B,)` or `(B, L)` wanted'
                )

    return (x - lower_bound) / (upper_bound - lower_bound + eps)
