from pathlib import Path
from typing import Optional, Sequence, Union

from ...typing import TorchTensor, TorchReal
from .pointcloudxyz import PointCloudReaderXYZ


class PointCloudReaderXYZI(PointCloudReaderXYZ):
    def __init__(
        self, f: Union[Path, str],
        fields: Sequence[str] = ('x', 'y', 'z', 'intensity'),
        *args, **kwargs
    ) -> None:
        r'''
        Read PCD in ASCII and version 0.7 only.

        ### Args:
            - f: path to a PCD file.
            - fields: fields to be loaded.

        ### Properties:
            - intensity: intensities. Its shape is `(N >= 0, 1)`.
            - intensity_: intensities. Its shape is `(N >= 0, 1)`.
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
        self._intensity_: Optional[TorchTensor[TorchReal]] = None

    @property
    def intensity_(self) -> TorchTensor[TorchReal]:
        if self._intensity_ is None:
            self._intensity_ = self._get_points(('intensity',))
        return self._intensity_

    @property
    def intensity(self) -> TorchTensor[TorchReal]:
        r'''
        Intensities. Its shape is `(N >= 0, 1)`.

        This is a copy of the data stored.

        '''
        return self.intensity_.clone()
