from pathlib import Path
from typing import Sequence, Union

from .pointcloudxyzi import PointCloudReaderXYZI
from .pointcloudxyztime import PointCloudReaderXYZTime


class PointCloudReaderXYZITime(PointCloudReaderXYZI, PointCloudReaderXYZTime):
    def __init__(
        self, f: Union[Path, str],
        fields: Sequence[str] = ('x', 'y', 'z', 'intensity', 'time'),
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
            - timestamp: timestamps. Its shape is `(N >= 0, 1)`.
            - timestamp_: timestamps. Its shape is `(N >= 0, 1)`.
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
