from typing import Sequence, Union
from copy import deepcopy
from pathlib import Path

from ..torch import PointCloudIRs
from .pointcloud import PointCloudWriter


class PointCloudWriterXYZIR(PointCloudWriter):
    def __init__(
        self,
        fields: Sequence[str] = ('x', 'y', 'z', 'intensity', 'ring'),
        types: Sequence[str] = ('F4', 'F4', 'F4', 'F4', 'U2'),
        *args, **kwargs
    ) -> None:
        super().__init__(fields, types, *args, **kwargs)

    def write(self, pcd: PointCloudIRs, path: Union[str, Path]) -> None:
        data = deepcopy(self._head)
        num = str(len(pcd))
        data[6] += num
        data[9] += num

        for xyz, i, r in zip(
            pcd.xyz_.tolist(),
            pcd.intensity_.squeeze().tolist(),
            pcd.ring_.squeeze().tolist()
        ):
            data.append(f'{xyz[0]} {xyz[1]} {xyz[2]} {i} {r}')

        path = Path(path).with_suffix('.pcd')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(data))
