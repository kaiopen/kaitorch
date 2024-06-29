from pathlib import Path
from typing import Optional, Sequence, Union

from ...typing import TorchTensor, TorchReal
from .pointcloudxyz import PointCloudReaderXYZ


class PointCloudReaderXYZR(PointCloudReaderXYZ):
    def __init__(
        self, f: Union[Path, str],
        fields: Sequence[str] = ('x', 'y', 'z', 'ring'),
        *args, **kwargs
    ) -> None:
        r'''
        Read PCD in ASCII and version 0.7 only.

        ### Args:
            - f: path to a PCD file.
            - fields: fields to be loaded.

        ### Properties:
            - ring: rings. Its shape is `(N >= 0, 1)`.
            - ring_: rings. Its shape is `(N >= 0, 1)`.
            - xyz: 3D coordinates in the 3D rectangular coordinate system. Its
                shape is `(N >= 0, 3)`. A coordiante is in the form of
                `(x, y, z)`.
            - xyz_: 3D coordinates in the 3D rectangular coordinate system. Its
                shape is `(N >= 0, 3)`. A coordiante is in the form of
                `(x, y, z)`.

        ### Methods:
            - __iter__
            - __getitem__
            - __len__
            - __next__
            - is_empty: Whether the point cloud is empty.

        '''
        super().__init__(f, fields, *args, **kwargs)
        self._ring_: Optional[TorchTensor[TorchReal]] = None

    @property
    def ring_(self) -> TorchTensor[TorchReal]:
        if self._ring_ is None:
            self._ring_ = self._get_points(('ring',))
        return self._ring_

    @property
    def ring(self) -> TorchTensor[TorchReal]:
        r'''
        Rings. Its shape is `(N >= 0, 1)`.

        This is a copy of the data stored.

        '''
        return self.ring_.clone()
