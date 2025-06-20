from pathlib import Path
from typing import Optional, Sequence, Union

from ...typing import TorchTensor, TorchReal
from .pointcloudxyz import PointCloudReaderXYZ


class PointCloudReaderXYZID(PointCloudReaderXYZ):
    r'''Read PCD in ASCII and version 0.7 only.

    #### Args:
    - f: path to a PCD file.
    - fields: fields to be loaded.

    #### Properties:
    - id: IDs. Its shape is `(N >= 0, 1)`.
    - id_: IDs. Its shape is `(N >= 0, 1)`.
    - xyz: 3D coordinates in the 3D rectangular coordinate system. Its shape is
        `(N >= 0, 3)`. A coordinate is in the form of `(x, y, z)`.
    - xyz_: 3D coordinates in the 3D rectangular coordinate system. Its shape
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
        fields: Sequence[str] = ('x', 'y', 'z', 'id'),
        *args, **kwargs
    ) -> None:
        super().__init__(f, fields, *args, **kwargs)
        self._id_: Optional[TorchTensor[TorchReal]] = None

    @property
    def id_(self) -> TorchTensor[TorchReal]:
        if self._id_ is None:
            self._id_ = self._get_points(('id',))
        return self._id_

    @property
    def id(self) -> TorchTensor[TorchReal]:
        r'''IDs. Its shape is `(N >= 0, 1)`.

        This is a copy of the data stored.

        '''
        return self.id_.clone()
