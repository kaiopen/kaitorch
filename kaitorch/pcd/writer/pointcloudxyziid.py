from typing import Sequence, Union
from copy import deepcopy
from pathlib import Path

from ..torch import PointCloudIIDs
from .pointcloud import PointCloudWriter


class PointCloudWriterXYZIID(PointCloudWriter):
    def __init__(
        self,
        fields: Sequence[str] = ('x', 'y', 'z', 'intensity', 'id'),
        types: Sequence[str] = ('F4', 'F4', 'F4', 'F4', 'I8'),
        *args, **kwargs
    ) -> None:
        super().__init__(fields, types, *args, **kwargs)

    def write(self, pcd: PointCloudIIDs, path: Union[str, Path]) -> None:
        data = deepcopy(self._head)
        num = str(len(pcd))
        data[6] += num
        data[9] += num

        for xyz, i, id in zip(
            pcd.xyz_.tolist(),
            pcd.intensity_.squeeze().tolist(),
            pcd.id_.squeeze().tolist()
        ):
            data.append(f'{xyz[0]} {xyz[1]} {xyz[2]} {i} {id}')

        path = Path(path).with_suffix('.pcd')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(data))
