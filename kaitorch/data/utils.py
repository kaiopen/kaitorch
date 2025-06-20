import collections
from itertools import repeat


def ntuple(n: int):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


tuple_2 = ntuple(2)
