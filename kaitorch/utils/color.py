import torch

from ..typing import TorchTensor, TorchFloat


PALETTE_30 = (
    (0.4118, 0.4118, 0.4118),
    (0.3333, 0.4196, 0.1843),
    (0.1333, 0.5451, 0.1333),
    (0.4980, 0.0000, 0.0000),
    (0.2824, 0.2392, 0.5451),
    (0.0000, 0.5451, 0.5451),
    (0.0000, 0.0000, 0.5020),
    (0.8235, 0.4118, 0.1176),
    (0.5020, 0.0000, 0.5020),
    (0.6902, 0.1882, 0.3765),
    (1.0000, 0.0000, 0.0000),
    (1.0000, 0.6471, 0.0000),
    (1.0000, 1.0000, 0.0000),
    (0.0000, 1.0000, 0.0000),
    (0.5412, 0.1686, 0.8863),
    (0.0000, 1.0000, 0.4980),
    (0.8627, 0.0784, 0.2353),
    (0.0000, 1.0000, 1.0000),
    (0.0000, 0.7490, 1.0000),
    (0.0000, 0.0000, 1.0000),
    (1.0000, 0.5020, 0.5020),
    (0.6784, 1.0000, 0.1843),
    (0.8471, 0.7490, 0.8471),
    (1.0000, 0.0000, 1.0000),
    (0.1176, 0.5647, 1.0000),
    (0.5647, 0.9333, 0.5647),
    (1.0000, 0.0784, 0.5765),
    (0.4824, 0.4078, 0.9333),
    (0.9333, 0.5098, 0.9333),
    (1.0000, 0.8549, 0.7255)
)


def pseudo_colors(a: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
    r'''Generate pseudo colors according to the data.

    ### Args:
        - a: data. Its shape should be `(N)` and value should be from 0 to 1.

    ### Returns:
        - Pseudo colors from 0 to 1. Its shape is `(N, 3)`.

    '''
    a = a * 255

    colors = torch.zeros((len(a), 3), dtype=a.dtype)

    mask = a < 64
    d = a[mask]
    colors[mask] = torch.stack(
        (torch.zeros_like(d), (256 - 4 * d) / 256, torch.ones_like(d)), dim=-1
    )

    mask = (a >= 64) & (a < 128)
    d = a[mask]
    colors[mask] = torch.stack(
        (torch.zeros_like(d), (4 * d - 256) / 256, (512 - 4 * d) / 256), dim=-1
    )

    mask = (a >= 128) & (a < 192)
    d = a[mask]
    colors[mask] = torch.stack(
        ((4 * d - 512) / 256, torch.ones_like(d), torch.zeros_like(d)), dim=-1
    )

    mask = a >= 192
    d = a[mask]
    colors[mask] = torch.stack(
        (torch.ones_like(d), (1024 - 4 * d) / 256, torch.zeros_like(d)), dim=-1
    )

    return colors
