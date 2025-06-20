r'''
PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space

'''
from typing import Any, Dict, Sequence

import torch
from torch import nn

from ..nn.activation import ACTIVATIONS
from ..nn.conv import Conv1dBlock
from ..typing import TorchTensor, TorchBool, TorchFloat, TorchInt64, TorchReal


def farthest_point_sample(
    points: TorchTensor[TorchReal], k: int
) -> TorchTensor[TorchInt64]:
    '''
    Define the minimal distance between a point and samples as the distance
    between the point and the sample set. Select the one that is the
    farthest away from existed sample set as a new sample and add it to the
    sample set. The first sample is selected randomly.

    Note that all attributes of a point will be considered when calculating the
    distance.

    #### Args:
    - points: points. Its shape should be `(B, C, N)` where `C` is the number
        of attributes a point have.
    - k: number of samples wanted.

    #### Returns:
    - Indices of samples. Its shape is `(B, k)`.

    '''
    batch_size, num_point, _ = points.shape
    i_b = torch.arange(batch_size, dtype=torch.long, device=points.device)
    ret = torch.zeros((batch_size, k), dtype=torch.long, device=points.device)
    distances = torch.ones(
        (batch_size, num_point), device=points.device
    ) * 1e10
    # Distances between points and sample set.

    farthest = torch.randint(
        0, num_point, (batch_size,),
        dtype=torch.long, device=points.device
    )
    # The first sample is selected randomly.
    for i in range(k):
        ret[:, i] = farthest
        sample = points[i_b, farthest].unsqueeze(1)  # (B, 1, C)
        d = torch.sum((points - sample) ** 2, -1)  # (B, N)
        mask = d < distances
        distances[mask] = d[mask]
        farthest = torch.argmax(distances, -1)
    return ret


def query(excluded: TorchTensor[TorchBool]) -> TorchTensor[TorchInt64]:
    r'''

    Warning: This function may take up a lot of memory.

    #### Args:
    - excluded: relation between center points and queried points. Its shape
        should be `(B, M, N)`, where `M` is the number of the center points and
        `N` is the number of the queried points. Center points must be included
        in queried points. If `True`, a point is not what a center point wants.

    #### Returns:
    - Indices of selected points. Its shape is `(B, M, N)`. The excluded points
        are marked by `N`.

    '''
    batch_size, num_center, num_point = excluded.shape
    ret = torch.arange(
        num_point, dtype=torch.long, device=excluded.device
    ).view(1, 1, num_point).repeat(batch_size, num_center, 1)  # (B, M, N)
    ret[excluded] = num_point
    return ret.sort(dim=-1)[0]


def ball_query(
    distances: TorchTensor[TorchReal],
    radius: float,
    k: int,
) -> TorchTensor[TorchInt64]:
    '''Select top k points in a range.

    Warning: This function may take up a lot of memory.

    Note: The order of selection depends on the order of points. Therefore, it
    is recommended to shuffle the points before.

    #### Args:
    - distances: distances for querying. Its shape should be `(B, M, N)`, where
        `M` is the number of center points and `N` is the number of queried
        points. Center points must be included in queried points (their
        distances are zero).
    - radius: maximal query radius.
    - k: number of points to be selected.

    #### Returns:
    - Indices of selected points. Its shape is `(B, M, k)`. If the number of
        points can be selected is less than k, the indices will be filled with
        `N`.

    '''
    return query(distances >= radius ** 2)[:, :, :k]


def dilated_ball_query(
    distances: TorchTensor[TorchReal],
    min_radius: float,
    max_radius: float,
    k: int,
) -> TorchTensor[TorchInt64]:
    '''Select top k points in a range.

    Warning: This function may take up a lot of memory.

    Note: The order of selection depends on the order of points. Therefore, it
    is recommended to shuffle the points before.

    #### Args:
    - distances: distances for querying. Its shape should be `(B, M, N)`, where
        `M` is the number of center points and `N` is the number of queried
        points. Center points must be included in queried points (their
        distances are zero).
    - min_radius: minimal query radius.
    - max_radius: maximal query radius.
    - k: number of points to be selected.

    #### Returns:
    - Indices of selected points. Its shape is `(B, M, k)`. If the number of
        points can be selected is less than k, the indices will be filled with
        `N`.

    '''
    return query(
        (distances >= max_radius ** 2) | (distances < min_radius ** 2)
    )[:, :, :k]


class NeighborOperator(nn.Module):
    def __init__(self, out_channels: int, *args, **kwargs) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(
        self,
        positions: TorchTensor[TorchReal],
        features: TorchTensor[TorchReal],
        centers: TorchTensor[TorchReal],
        *args, **kwargs
    ) -> TorchTensor[TorchFloat]:
        raise NotImplementedError


class PointNetNeighborOperator(NeighborOperator):
    r'''PointNet2 operator to extract features of neighbor points.

    #### Args:
    - in_channels: the number of input channels.
    - mlps: number of channels for MLP including input channels.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - positions: coordinates of neighbor points. Its shape should be `(N, D)`.
    - features: features of neighbor points. Its shape should be `(N, C)`.
    - centers: coordinates of center points for neighbor points. Its shape
        should be `(N, D)`.

    #### Returns:
    - New features of neighbor points. Its shape is `(N, C')`.

    '''
    def __init__(
        self,
        in_channels: int,
        mlp: Sequence[int],
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__(out_channels=mlp[-1], *args, **kwargs)
        self._mlp = []
        for c in mlp:
            self._mlp += [
                nn.Linear(in_channels, c, bias=False),
                nn.BatchNorm1d(c),
                ACTIVATIONS[activation](**activation_kw)
            ]
            in_channels = c
        self._mlp = nn.Sequential(*self._mlp)

    def forward(
        self,
        positions: TorchTensor[TorchReal],
        features: TorchTensor[TorchReal],
        centers: TorchTensor[TorchReal],
        *args, **kwargs
    ) -> TorchTensor[TorchFloat]:
        return self._mlp(torch.cat((positions - centers, features), dim=-1))


class PointSampler(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(
        self,
        positions: TorchTensor[TorchReal],
        features: TorchTensor[TorchReal],
        centers: TorchTensor[TorchReal],
        *args, **kwargs
    ) -> TorchTensor[TorchFloat]:
        raise NotImplementedError


class PointNetSampler(PointSampler):
    r'''

    #### Args:
    - batch_size
    - num_center: number of center points.
    - radius: radius of query.
    - num_neighbor: number of neighbors for query.
    - operator: instance of `NeighborOperator`.
    - out_channels: the number of output channels.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - positions: coordinates of points. Its shape should be `(B, N, D)`.
    - features: features of points. Its shape should be `(B, N, C)`.
    - centers: coordinates of center points. Its shape should be `(B, M, D)`.
    - distances: distances between center points and points. Its shape should
        be `(B, M, N)`.

    #### Returns:
    - Features of center points. Its shape is `(B, M, out_channels)`.

    '''
    def __init__(
        self,
        batch_size: int,
        num_center: int,
        radius: float,
        num_neighbor: int,
        operator: NeighborOperator,
        out_channels: int,
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._radius = radius
        self._num_neighbor = num_neighbor
        self._operator = operator

        self.register_buffer(
            '_ids',
            torch.arange(
                batch_size
            ).reshape(-1, 1, 1).expand(batch_size, num_center, num_neighbor),
            persistent=False
        )
        self.register_buffer(
            '_i_c',
            (
                torch.arange(num_center).reshape(1, -1, 1) +
                torch.arange(batch_size).reshape(-1, 1, 1) * num_center
            ).expand(batch_size, num_center, num_neighbor),
            persistent=False
        )
        self.register_buffer(
            '_ids_neg',
            torch.full((batch_size, num_center, num_neighbor), -1),
            persistent=False
        )
        self.register_buffer(
            '_filler',
            torch.arange(batch_size * num_center * num_neighbor),
            persistent=False
        )
        operator_channels = operator.out_channels
        self.register_buffer(
            '_zeros', torch.zeros(1, operator_channels), persistent=False
        )

        self._agg = Conv1dBlock(
            operator_channels, out_channels, 1,
            activation=activation,
            activation_kw=activation_kw
        )

    def forward(
        self,
        positions: TorchTensor[TorchReal],
        features: TorchTensor[TorchReal],
        centers: TorchTensor[TorchReal],
        distances: TorchTensor[TorchReal],
        *args, **kwargs
    ) -> TorchTensor[TorchFloat]:
        num_point = positions.shape[1]

        positions = positions.flatten(end_dim=1)  # (BN, D)
        features = features.flatten(end_dim=1)  # (BN, C)
        centers = centers.flatten(end_dim=1)  # (BM, D)

        ids = ball_query(distances, self._radius, self._num_neighbor)
        # (B, M, k)
        mask = ids != num_point
        ids = ids[mask] + self._ids[mask] * num_point  # (X,)
        # Add `self._ids[mask] * num_point` for batch.

        f = self._operator(
            positions[ids], features[ids], centers[self._i_c[mask]]
        )  # (X, C') X <= BMk

        # MAX
        ids = self._ids_neg.clone()
        # If the number of neighbors is less than k, fill zeros to align.
        ids[mask] = self._filler[:len(f)]

        return self._agg(
            torch.max(
                torch.cat((f, self._zeros), dim=0)[ids],  # (B, M, k, C')
                dim=2
            )[0].permute(0, 2, 1)  # (B, C', M)
        ).permute(0, 2, 1)


class PointNet2Sampler(PointSampler):
    r'''

    #### Args:
    - batch_size
    - num_center: number of center points.
    - radii: max radii of dilated query.
    - num_neighbor: number of neighbors for each query.
    - operators: instances of `NeighborOperator`.
    - out_channels: the number of output channels.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - positions: coordinates of points. Its shape should be `(B, N, D)`.
    - features: features of points. Its shape should be `(B, N, C)`.
    - centers: coordinates of center points. Its shape should be `(B, M, D)`.
    - distances: distances between center points and points. Its shape should
        be `(B, M, N)`.

    ### Returns:
    - Features of center points. Its shape is `(B, M, C')`.

    '''
    def __init__(
        self,
        batch_size: int,
        num_center: int,
        radii: Sequence[float],
        num_neighbor: Sequence[int],
        operators: Sequence[NeighborOperator],
        out_channels: int,
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self.register_buffer(
            '_ids',
            torch.arange(batch_size).reshape(-1, 1, 1),
            persistent=False
        )

        i_c = torch.arange(num_center).reshape(1, -1, 1) + \
            torch.arange(batch_size).reshape(-1, 1, 1) * num_center

        names_i_c = []
        names_ids_neg = []
        names_filler = []
        for i, n in enumerate(num_neighbor):
            name = f'_i_c_{i:d}'
            self.register_buffer(
                name,
                i_c.expand(batch_size, num_center, n),
                persistent=False
            )
            names_i_c.append(name)

            name = f'_ids_neg_{i:d}'
            self.register_buffer(
                name,
                torch.full((batch_size, num_center, n), -1),
                persistent=False
            )
            names_ids_neg.append(name)

            name = f'_filler_{i:d}'
            self.register_buffer(
                '_filler',
                torch.arange(batch_size * num_center * n),
                persistent=False
            )
            names_filler.append(name)

        in_channels = 0
        names_zero = []
        for i, op in enumerate(operators):
            name = f'_zeros_{i:d}'
            operator_channels = op.out_channels
            self.register_buffer(
                name, torch.zeros(1, operator_channels), persistent=False
            )
            in_channels += operator_channels
            names_zero.append(name)

        self._loop = zip(
            radii, num_neighbor, names_i_c, operators,
            names_ids_neg, names_filler, names_zero
        )

        self._agg = Conv1dBlock(
            in_channels, out_channels, 1,
            activation=activation, activation_kw=activation_kw
        )

    def forward(
        self,
        positions: TorchTensor[TorchReal],
        features: TorchTensor[TorchReal],
        centers: TorchTensor[TorchReal],
        distances: TorchTensor[TorchReal]
    ) -> TorchTensor[TorchFloat]:
        num_point = positions.shape[1]
        _ids = self._ids * num_point

        positions = positions.flatten(end_dim=1)  # (BN, D)
        features = features.flatten(end_dim=1)  # (BN, C)
        centers = centers.flatten(end_dim=1)  # (BM, D)

        min_radius = -1
        fs = []
        for (
            max_radius, k,
            name_i_c, op,
            name_ids_neg, name_filler, name_zero
        ) in self._loop:
            indices = dilated_ball_query(
                distances, min_radius, max_radius, k
            )  # (B, M, k)
            min_radius = max_radius
            mask = indices != num_point
            ids = indices[mask] + _ids[mask]

            f = op(
                positions[ids],
                features[ids],
                centers[self.get_buffer(name_i_c)[mask]]
            )  # (X, C')

            #  MAX
            ids = self.get_buffer(name_ids_neg).clone()
            # If the number of neighbors is less than k, fill zeros to align.
            ids[mask] = self.get_buffer(name_filler)[:len(f)]
            fs.append(
                torch.max(
                    torch.cat(
                        (f, self.get_buffer(name_zero)),
                        dim=0
                    )[ids],  # (B, M, k, C')
                    dim=2
                )[0].permute(0, 2, 1)  # (B, C', M)
            )
        return self._agg(torch.cat(fs, dim=1)).permute(0, 2, 1)
