from typing import Sequence, Union
from copy import deepcopy
from pathlib import Path

from ..torch import PointCloudTimes
from .pointcloud import PointCloudWriter


class PointCloudWriterXYZTime(PointCloudWriter):
    def __init__(
        self,
        fields: Sequence[str] = ('x', 'y', 'z', 'timestamp'),
        types: Sequence[str] = ('F4', 'F4', 'F4', 'F8'),
        *args, **kwargs
    ) -> None:
        super().__init__(fields, types, *args, **kwargs)

    def write(self, pcd: PointCloudTimes, path: Union[str, Path]) -> None:
        data = deepcopy(self._head)
        num = str(len(pcd))
        data[6] += num
        data[9] += num

        for xyz, t in zip(
            pcd.xyz_.tolist(), pcd.timestamp_.squeeze().tolist()
        ):
            data.append(f'{xyz[0]} {xyz[1]} {xyz[2]} {t}')

        path = Path(path).with_suffix('.pcd')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(data))
