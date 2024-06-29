import collections
from itertools import repeat

from torch import nn


def _ntuple(n: int):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


class NoneGenerator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        r'''Do nothing but return a `None`.

        '''
        super().__init__()

    def forward(self, *args, **kwargs):
        return None
