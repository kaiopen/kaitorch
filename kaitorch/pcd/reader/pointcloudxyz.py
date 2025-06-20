from pathlib import Path
from typing import Optional, Sequence, Union

from ...typing import TorchTensor, TorchReal

from .pointcloud import PointCloudReader


class PointCloudReaderXYZ(PointCloudReader):
    r'''Read PCD in ASCII and version 0.7 only.

    #### Args:
    - f: path to a PCD file.
    - fields: fields to be loaded.

    #### Properties:
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
        self,
        f: Union[Path, str],
        fields: Sequence[str] = ('x', 'y', 'z'),
        *args, **kwargs
    ) -> None:
        super().__init__(f, fields, *args, **kwargs)
        self._xyz_: Optional[TorchTensor[TorchReal]] = None

    @property
    def xyz_(self) -> TorchTensor[TorchReal]:
        if self._xyz_ is None:
            self._xyz_ = self._get_points(('x', 'y', 'z'))
        return self._xyz_

    @property
    def xyz(self) -> TorchTensor[TorchReal]:
        r'''3D coordinates in the 3D rectangular coordinate system. Its shape
        is `(N >= 0, 3)`. A coordinate is in the form of `(x, y, z)`.

        This is a copy of the data stored.

        '''
        return self.xyz_.clone()
