from ..typing import TorchTensor, TorchFloat, TorchReal


def min_max_normalize(
    a: TorchTensor[TorchReal],
    lower_bound: TorchTensor[TorchReal],
    upper_bound: TorchTensor[TorchReal]
) -> TorchTensor[TorchFloat]:
    r'''

    ### Args:
        - a: data to be normalized. Its shape should be `(N, C)`.
        - lower_bound: lower bound of the data. Its shape should be `(C,)`.
        - upper_bound: upper bound of the data. Its shape should be `(C,)`.

    ### Returns:
        - Normalized data. Its shape is `(N, C)`.

    '''
    return (a - lower_bound) / (upper_bound - lower_bound)


def z_score(
    a: TorchTensor[TorchReal],
    mean: TorchTensor[TorchReal],
    std: TorchTensor[TorchReal]
) -> TorchTensor[TorchFloat]:
    r'''

    ### Args:
        - a: data to be normalized. Its shape should be `(N, C)`.
        - mean: average of the data. Its shape should be `(N, C)`.
        - std: standard deviation of the data. Its shape should be `(N, C)`.

    ### Returns:
        - Normalized data. Its shape is `(N, C)`.

    '''
    return (a - mean) / std
