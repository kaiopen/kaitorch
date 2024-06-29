import torch

from ..typing import TorchTensor, TorchFloat


def euclidean_distance(
    a: TorchTensor[TorchFloat], b: TorchTensor[TorchFloat]
) -> TorchTensor[TorchFloat]:
    r'''Euclidean distances between a and b.

    ### Args:
        - a: coordinates. Its shape should be `([*,] C)`.
        - b: coordinates. Its shape should be `([*,] C)`.

    ### Returns:
        - Distances between a and b. Its shape is `([*,])`.

    '''
    return torch.linalg.norm(a - b, dim=-1)


def euclidean_distance_polar(
    a: TorchTensor[TorchFloat], b: TorchTensor[TorchFloat]
) -> TorchTensor[TorchFloat]:
    r'''Euclidean distances between polar coordinates a and b.

    ### Args:
        - a: coordinates in a polar coordiante system. Its shape should be
            `([*,] 2 [+ C])`.
        - b: coordinates in a polar coordinate system. Its shape should be
            `([*,] 2 [+ C])`.

    ### Returns:
        - Distances between a and b. Its shape is `([*,])`.

    '''
    return torch.sqrt(squared_euclidean_distance_polar(a, b))


def squared_euclidean_distance(
    a: TorchTensor[TorchFloat], b: TorchTensor[TorchFloat]
) -> TorchTensor[TorchFloat]:
    r''''Squared Euclidean distances between a and b.

    ### Args:
        - a: coordinates. Its shape should be `([*,] C)`.
        - b: coordinates. Its shape should be `([*,] C)`.

    ### Returns:
        - Squared distances between a and b. Its shape is `([*,])`.

    '''
    return torch.sum(torch.pow(a - b, 2), dim=-1)


def squared_euclidean_distance_polar(
    a: TorchTensor[TorchFloat], b: TorchTensor[TorchFloat]
) -> TorchTensor[TorchFloat]:
    r'''Euclidean distances between polar coordinates a and b.

    ### Args:
        - a: coordinates in a polar coordiante system. Its shape should be
            `([*,] 2 [+ C])`.
        - b: coordinates in a polar coordinate system. Its shape should be
            `([*,] 2 [+ C])`.

    ### Returns:
        - Squared distances between a and b. Its shape is `([*,])`.

    '''
    r_1 = a[..., 0]
    t_1 = a[..., 1]
    r_2 = b[..., 0]
    t_2 = b[..., 1]
    d = r_1 * r_1 + r_2 * r_2 - 2 * r_1 * r_2 * torch.cos(t_1 - t_2)
    d[d < 0] = 0  # correct computing error
    return d
