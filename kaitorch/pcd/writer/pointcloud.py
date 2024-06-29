from typing import Sequence, Union
from pathlib import Path

from ..torch import PointClouds


class PointCloudWriter:
    def __init__(
        self,
        fields: Sequence[str],
        types: Sequence[str],
        *args, **kwargs
    ) -> None:
        r'''

        '''
        dim = len(fields)
        if len(types) != dim:
            raise ValueError('Conflicted fields and types.')

        self._head = [
            '# .PCD v0.7 - Point Cloud Data file format',
            'VERSION 0.7',
            'FIELDS ' + ' '.join(fields)
        ]
        ts = ['SIZE']
        ss = ['TYPE']
        for t in types:
            ts.append(t[0])
            ss.append(t[1])
        self._head.append(' '.join(ss))
        self._head.append(' '.join(ts))
        self._head.append('COUNT ' + ' '.join(['1'] * dim))
        self._head.append('WIDTH ')
        self._head.append('HEIGHT 1')
        self._head.append('VIEWPOINT 0 0 0 1 0 0 0')
        self._head.append('POINTS ')
        self._head.append('DATA ascii')

    def __call__(self, pcd: PointClouds, path: Union[str, Path]) -> None:
        return self.write(pcd, path)

    def write(self, pcd: PointClouds, path: Union[str, Path]) -> None:
        raise NotImplementedError
