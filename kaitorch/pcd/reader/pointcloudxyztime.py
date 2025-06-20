from pathlib import Path
from typing import Optional, Sequence, Union

from ...typing import TorchTensor, TorchReal
from .pointcloudxyz import PointCloudReaderXYZ


class PointCloudReaderXYZTime(PointCloudReaderXYZ):
    r'''Read PCD in ASCII and version 0.7 only.

    #### Args:
    - f: path to a PCD file.
    - fields: fields to be loaded.

    #### Properties:
    - timestamp: timestamps. Its shape is `(N >= 0, 1)`.
    - timestamp_: timestamps. Its shape is `(N >= 0, 1)`.
    - xyz: 3D coordinates in the 3D rectangular coordinate system. Its shape is
        `(N >= 0, 3)`. A coordinate is in the form of `(x, y, z)`.
    - xyz_: 3D coordinates in the 3D rectangular coordinate system. Its  shape
        is `(N >= 0, 3)`. A coordinate is in the form of `(x, y, z)`.

    #### Methods:
    - __iter__
    - __getitem__
    - __len__
    - __next__
    - is_empty: Whether the point cloud is empty.

    '''
    def __init__(
        self, f: Union[Path, str],
        fields: Sequence[str] = ('x', 'y', 'z', 'timestamp'),
        *args, **kwargs
    ) -> None:
        super().__init__(f, fields, *args, **kwargs)
        self._timestamp_: Optional[TorchTensor[TorchReal]] = None

    @property
    def timestamp_(self) -> TorchTensor[TorchReal]:
        if self._timestamp_ is None:
            self._timestamp_ = self._get_points(('timestamp',))
        return self._timestamp_

    @property
    def timestamp(self) -> TorchTensor[TorchReal]:
        r'''Timestamps. Its shape is `(N >= 0, 1)`.

        This is a copy of the data stored.

        '''
        return self.timestamp_.clone()
