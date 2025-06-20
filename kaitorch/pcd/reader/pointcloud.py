from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import warnings

import torch
from ...typing import TorchTensor, TorchReal, TorchDType


def _get_dtype(t: str) -> TorchDType:
    if 'U1' == t:
        return torch.uint8
    if 'I1' == t:
        return torch.int8
    if t in ('I2', 'U2'):
        return torch.int16
    if t in ('I4', 'U4'):
        return torch.int32
    if t in ('I8', 'U8'):
        return torch.int64
    if t in ('F4', 'F1', 'F2'):
        return torch.float32
    if 'F8' == t:
        return torch.float64


def _set_point(
    fields: Sequence[str], types: Sequence[str], values: Sequence[str]
) -> Dict[str, TorchReal]:
    d = {}
    # e.x.
    # {
    #   'x': torch.tensor(123., dtype=torch.float32),
    #   'y': torch.tensor(123., dtype=torch.float32),
    #   'z': torch.tensor(123., dtype=torch.float32)
    # }
    for i, v in enumerate(values):
        d[fields[i]] = torch.tensor(float(v), dtype=_get_dtype(types[i]))
    return d


class PointCloudReader:
    r'''Abstract class for a point cloud.

    Read PCD in ASCII and version 0.7 only.

    #### Args:
    - f: path to a PCD file.
    - fields: fields to be loaded.

    #### Methods:
    - __iter__
    - __getitem__
    - __len__
    - __next__
    - is_empty: Whether the point cloud is empty.

    '''
    def __init__(
        self,
        f: Union[Path, str],
        fields: Optional[Sequence[str]] = None,
        *args, **kwargs
    ) -> None:
        if isinstance(f, str):
            f = Path(f)
        elif not isinstance(f, Path):
            raise ValueError('f should be a str or Path.')

        lines = f.read_text().splitlines()
        if lines[10].strip().split()[1] != 'ascii':
            raise TypeError('PCD in ASCII required.')

        if float(lines[1].strip().split()[1]) != 0.7:
            warnings.warn('This lib is for PCD 0.7.')

        f_field = lines[2].strip().split()[1:]
        num_field = len(f_field)

        f_points = []
        for line in lines[11:]:
            p = line.strip().split()
            if len(p) != num_field:
                raise IOError('Dirty file caused by points.')
            f_points.append(p)

        sizes = lines[3].strip().split()[1:]
        types = lines[4].strip().split()[1:]
        if len(sizes) != num_field and len(types) != num_field:
            raise IOError('Dirty file caused by fields, sizes or types.')

        if fields is not None:
            if not set(fields).issubset(f_field):
                raise ValueError('Fields are not included in f.')

            if len(fields) != len(f_field):
                s = []
                t = []
                ids = []
                for f in fields:
                    i = f_field.index(f)
                    s.append(sizes[i])
                    t.append(types[i])
                    ids.append(i)
                sizes = s
                types = t
                for i, p in enumerate(f_points):
                    new_p = []
                    for idx in ids:
                        new_p.append(p[idx])
                    f_points[i] = new_p
        else:
            fields = f_field

        types = [t + s for t, s in zip(types, sizes)]

        self._points: List[Dict[str, TorchReal]] = []
        for p in f_points:
            self._points.append(_set_point(fields, types, p))

        self._num_point = int(lines[9].strip().split()[1])
        if len(self._points) != self._num_point:
            raise IOError('Dirty file caused by the number of points.')

        self.__i: int = 0

    def __iter__(self):
        return self

    def __getitem__(self, index: int) -> Dict[str, TorchReal]:
        return self._points[index]

    def __len__(self) -> int:
        return self._num_point

    def __next__(self) -> Dict[str, TorchReal]:
        if self.__i < self._num_point:
            data = self[self.__i]
            self.__i += 1
            return data
        self.__i = 0
        raise StopIteration

    def _get_points(self, fields: Sequence[str]) -> TorchTensor[TorchReal]:
        return torch.tensor([[p[f] for f in fields] for p in self])

    def is_empty(self) -> bool:
        r'''Whether the point cloud is empty.

        #### Return:
        - Empty or not.

        '''
        return 0 == self._num_point
