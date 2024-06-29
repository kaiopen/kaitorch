from pathlib import Path
from typing import Sequence, Union

from .pointcloudxyzid import PointCloudReaderXYZID
from .pointcloudxyzr import PointCloudReaderXYZR
from .pointcloudxyztime import PointCloudReaderXYZTime


class PointCloudReaderXYZRTimeID(
    PointCloudReaderXYZR, PointCloudReaderXYZTime, PointCloudReaderXYZID
):
    def __init__(
        self, f: Union[Path, str],
        fields: Sequence[str] = ('x', 'y', 'z', 'ring', 'time', 'id'),
        *args, **kwargs
    ) -> None:
        r'''
        Read PCD in ASCII and version 0.7 only.

        ### Args:
            - f: path to a PCD file.
            - fields: fields to be loaded.

        ### Properties:
            - id: IDs. Its shape is `(N >= 0, 1)`.
            - id_: IDs. Its shape is `(N >= 0, 1)`.
            - ring: rings. Its shape is `(N >= 0, 1)`.
            - ring_: rings. Its shape is `(N >= 0, 1)`.
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
