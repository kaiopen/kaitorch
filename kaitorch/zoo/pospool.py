r'''
A Closer Look at Local Aggregation Operators in Point Cloud Analysis

'''
import torch

from .pointnet2 import NeighborOperator
from ..typing import TorchTensor, TorchFloat


class PosPoolNeighborOperator(NeighborOperator):
    r'''PosPool operator to extract features of neighbor points.

    #### args:
    - position_channels: number of channels of neighbors' positions.
    - features_channels: number of channels of neighbors' features.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - positions: coordinates of neighbor points. Its shape should be
        `(N, position_channels)`.
    - features: features of neighbor points. Its shape should be
        `(N, feature_channels)`.
    - centers: coordinates of center points for neighbor points. Its shape
        should be `(N, position_channels)`.

    #### Returns:
    - New features of neighbor points. Its shape is `(N, feature_channels)`.

    '''
    def __init__(
        self,
        position_channels: int,
        feature_channels: int,
        *args, **kwargs
    ) -> None:
        super().__init__(out_channels=feature_channels, *args, **kwargs)
        self.register_buffer(
            '_d',
            torch.pow(
                1000,
                torch.arange(
                    0, 1,
                    step=2 * position_channels / feature_channels
                )
            ),
            persistent=False
        )

    def forward(
        self,
        positions: TorchTensor[TorchFloat],
        features: TorchTensor[TorchFloat],
        centers: TorchTensor[TorchFloat]
    ) -> TorchTensor[TorchFloat]:
        w = torch.div(
            torch.unsqueeze(100 * (positions - centers), -1),
            self._d
        )
        # (N, position_channels, feature_channels // (2 * position_channels))
        return features * \
            torch.cat(
                (torch.sin(w), torch.cos(w)),
                -1
            ).reshape(-1, self.out_channels)
