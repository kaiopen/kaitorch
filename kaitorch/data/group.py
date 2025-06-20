from typing import Optional, Tuple, Union

import torch
from torch import nn

from ..typing import TorchDevice, TorchTensor, TorchTensorLike, \
    TorchBool, TorchFloat, TorchInt64, TorchReal, Bool, Real


class Group:
    r'''Group data.

    A group is calculated as follows.
            group = int((a - lower_bound) / cell + error)

    #### Args:
    - lower_bound: lower bound of the data. Its length should be `C`.
    - cell: size of a cell for grouping. Its length should be `C`.
    - error: tolerant error. Its shape should be `(C,)` or its length
        should be larger than 1. An error should be within [0, 1).
    - closed: Whether do loop closing. Its shape should be `(C,)` or
        its length should be larger than 1. If `True`, the last group
        will be merged into the first group.
    - upper_bound: upper bound of the data. Its length should be `C`.
        Used for calculating the maximal group ID if do loop closing.
        If not provided, the maximal group ID will be calculated
        according to the input data.
    - return_offset: Whether to return offsets. An offset is calculated
        as follows.
            offset = (a - lower_bound) / cell + error - group
    - device

    #### Methods:
    - __call__

    ## __call__
    #### Args:
    - x: data to be grouped. Its shape should be `([*,] C)`.

    #### Returns:
    - Groups. Its shape is `([*,] C)`.
    - (optional) Offsets. Its shape is `([*,] C)`. Returned if `return_offset`
        is `True`.

        '''
    def __init__(
        self,
        lower_bound: TorchTensorLike[Real],
        cell: TorchTensorLike[Real],
        error: TorchTensorLike[Real] = torch.as_tensor(0),
        closed: TorchTensorLike[Bool] = torch.as_tensor(False),
        upper_bound: Optional[TorchTensorLike[Real]] = None,
        return_offset: bool = False,
        device: TorchDevice = torch.device('cpu')
    ) -> None:
        super().__init__()
        self._lower_bound = torch.as_tensor(lower_bound, device=device)
        self._cell = torch.as_tensor(cell, device=device)
        self._error = torch.as_tensor(
            error, device=device
        ).expand_as(self._cell)
        closed = torch.as_tensor(closed, device=device)

        if torch.any(closed):
            self._closed = closed
            self._close_ = self._close_loop_
            if upper_bound is None:
                self._max_group = self._max_group_no_bound
            else:
                self._mg = torch.ceil(
                    (
                        torch.as_tensor(upper_bound, device=device)[closed] -
                        self._lower_bound[closed]
                    ) / self._cell[closed] + self._error[closed]
                ).long() - 1,
                self._max_group = self._max_group_with_bound
        else:
            self._close_ = nn.Identity()

        self._ret = self._return_offset if return_offset else self._return

    def __call__(
        self, x: TorchTensor[TorchReal]
    ) -> Union[
        TorchTensor[TorchInt64],
        Tuple[TorchTensor[TorchInt64], TorchTensor[TorchFloat]]
    ]:
        return self._ret(x)

    def _close_loop_(
        self, groups: TorchTensor[TorchInt64]
    ) -> TorchTensor[TorchInt64]:
        s = groups.shape
        closed_broadcasted = self._closed.expand(s)
        g = groups[closed_broadcasted].reshape(*s[:-1], -1)
        g[self._max_group(g) == g] = 0
        groups[closed_broadcasted] = g.flatten()
        return groups

    @staticmethod
    def _max_group_no_bound(
        g: TorchTensor[TorchInt64]
    ) -> TorchTensor[TorchInt64]:
        return g.max(dim=-2, keepdim=True)

    def _max_group_with_bound(
        self, g: TorchTensor[TorchInt64]
    ) -> TorchTensor[TorchInt64]:
        return self._mg.reshape([1] * (g.ndim - 1) + [-1])

    def _return(self, x: TorchTensor[TorchReal]) -> TorchTensor[TorchInt64]:
        return self._close_(
            ((x - self._lower_bound) / self._cell + self._error).long()
        )

    def _return_offset(
        self, x: TorchTensor[TorchReal]
    ) -> Tuple[TorchTensor[TorchInt64], TorchTensor[TorchFloat]]:
        a_in_group = (x - self._lower_bound) / self._cell + self._error
        groups = a_in_group.long()
        offsets = a_in_group - groups
        return self._close_(groups), offsets


class ReverseGroup:
    def __init__(
        self,
        lower_bound: TorchTensorLike[Real],
        cell: TorchTensorLike[Real],
        error: TorchTensorLike[Real] = torch.as_tensor(0),
        device: TorchDevice = torch.device('cpu')
    ) -> None:
        r'''Reverse operation of grouping.

        #### Args:
        - lower_bound: lower bound of the data. Its length should be `C`.
        - cell: size of a cell for grouping. Its length should be `C`.
        - error: tolerant error. Its shape should be `(C,)` or its length
            should be larger than 1. An error should be within [0, 1).
        - device

        #### Methods:
        - forward

        ## forward
        #### Args:
        - x: groups. Its shape should be `([*,] C)`.

        #### Returns:
        - Coordinates. Its shape is `([*,] C)`.

        '''
        super().__init__()
        self._lower_bound = torch.as_tensor(lower_bound, device=device)
        self._cell = torch.as_tensor(cell, device=device)
        self._error = torch.as_tensor(error, device=device)

    def __call__(self, x: TorchTensor[Real]) -> TorchTensor[TorchReal]:
        return (x - self._error) * self._cell + self._lower_bound


def group(
    a: TorchTensor[TorchReal],
    lower_bound: TorchTensor[TorchReal],
    cell: TorchTensor[TorchReal],
    error: TorchTensor[TorchReal] = torch.as_tensor(0),
    closed: TorchTensor[TorchBool] = torch.as_tensor(False),
    upper_bound: Optional[TorchTensor[TorchReal]] = None,
    return_offset: bool = False
) -> Union[
    TorchTensor[TorchInt64],
    Tuple[TorchTensor[TorchInt64], TorchTensor[TorchFloat]]
]:
    r'''Group data.

    A group is calculated as follows.
            group = int((a - lower_bound) / cell + error)

    #### Args:
    - a: data to be grouped. Its shape should be `([*,] C)`.
    - lower_bound: lower bound of the data. Its shape should be `(C,)`.
    - cell: size of a cell for grouping. Its shape should be `(C,)`.
    - error: tolerant error. Its shape should be `(C,)` or its length should be
        larger than 1. An error should be within [0, 1).
    - closed: Whether do loop closing. Its shape should be `(C,)` or its length
        should be larger than 1. If `True`, the last group will be merged into
        the first group.
    - upper_bound: upper bound of the data. Its length should be `C`. Used for
        calculating the maximal group ID if do loop closing. If not provided,
        the maximal group ID will be calculated according to the input data.
    - return_offset: Whether return offsets. An offset is calculated as
        follows.
            offset = (a - lower_bound) / cell + error - group

    #### Returns:
    - Groups. Its shape is `([*,] C)`.
    - (optional) Offsets. Its shape is `([*,] C)`. Returned if `return_offset`
        is `True`.

    '''
    error = error.expand_as(cell)
    a_in_group = (a - lower_bound) / cell + error
    groups = a_in_group.long()

    if return_offset:
        ret = (groups, a_in_group - groups)
    else:
        ret = groups

    if torch.any(closed):
        s = groups.shape
        closed_broadcasted = closed.expand(s)
        g = groups[closed_broadcasted].reshape(*s[:-1], -1)
        if upper_bound is None:
            max_group = g.max(dim=-2, keepdims=True)
        else:
            max_group = torch.ceil(
                (
                    torch.as_tensor(upper_bound)[closed] - lower_bound[closed]
                ) / cell[closed] + error[closed]
            ).reshape([1] * (g.ndim - 1) + [-1]).long() - 1

        g[g == max_group] = 0
        groups[closed_broadcasted] = g.flatten()
    return ret


def reverse_group(
    a: TorchTensor[TorchReal],
    lower_bound: TorchTensor[TorchReal],
    cell: TorchTensor[TorchReal],
    error: TorchTensor[TorchReal] = torch.as_tensor(0)
) -> TorchTensor[TorchReal]:
    r'''Reverse operation of grouping.

    #### Args:
    - a: groups. Its shape should be `([*,] C)`.
    - lower_bound: lower bound of the data. Its length should be `C`.
    - cell: size of a cell for grouping. Its length should be `C`.
    - error: tolerant error. Its shape should be `(C,)` or its length should be
        larger than 1. An error should be within [0, 1).

    #### Returns:
    - Coordinates. Its shape is `([*,] C)`.

    '''
    return (a - error) * cell + lower_bound


def cell_from_size(
    lower_bound: TorchTensor[TorchReal],
    upper_bound: TorchTensor[TorchReal],
    size: TorchTensor[TorchReal],
    error: TorchTensor[TorchReal] = torch.as_tensor(0),
    closed: TorchTensor[TorchBool] = torch.as_tensor(False)
) -> TorchTensor[TorchFloat]:
    r'''Size of a cell.

    #### Args:
    - lower_bound: lower bound of data. Its shape should be `(C,)`.
    - upper_bound: upper bound of data. Its shape should be `(C,)`.
    - size: size of grouped data. Its shape should be `(C,)`.
    - error: tolerant error. Its shape should be `(C,)` or its length should be
        larger than 1. An error should be within [0, 1).
    - closed: Whether do loop closing. Its shape should be `(C,)` or its length
        should be larger than 1. If `True`, the last group will be merged into
        the first group.

    #### Returns:
    - Size of a cell. Its shape is `(C,)`.

    '''
    return (upper_bound - lower_bound) / (
        size - torch.logical_and(torch.logical_not(closed), 0 != error).float()
    )


def size_from_cell(
    lower_bound: TorchTensor[TorchReal],
    upper_bound: TorchTensor[TorchReal],
    cell: TorchTensor[TorchReal],
    error: TorchTensor[TorchReal] = torch.as_tensor(0),
    closed: TorchTensor[TorchBool] = torch.as_tensor(False)
) -> TorchTensor[TorchInt64]:
    r'''Size of grouped data.

    #### Args:
    - lower_bound: lower bound of data. Its shape should be `(C,)`.
    - upper_bound: upper bound of data. Its shape should be `(C,)`.
    - cell: size of a cell. Its shape should be `(C,)`.
    - error: tolerant error. Its shape should be `(C,)` or its length should be
        larger than 1. An error should be within [0, 1).
    - closed: Whether do loop closing. Its shape should be `(C,)` or its length
        should be larger than 1. If `True`, the last group will be merged into
        the first group.

    #### Returns:
    - Size of grouped data. Its shape is `(C,)`.

    '''
    return torch.ceil(
        (upper_bound - lower_bound) / cell + error
    ).long() - closed.long()
