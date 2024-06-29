from typing import Any, Dict, Optional
from torch import nn


NORMALIZATION = {
    'batchnorm1d': nn.BatchNorm1d,
    'batchnorm2d': nn.BatchNorm2d,
    'batchnorm3d': nn.BatchNorm3d,
    'layernorm': nn.LayerNorm,
    'groupnorm': nn.GroupNorm
}


def normalization(
    name: str,
    kw: Optional[Dict[str, Any]] = None,
    channels: Optional[int] = None,
) -> nn.Module:
    if kw is None:
        if channels is None:
            raise ValueError(
                'Either `kw` or `channels` should not be `None`.'
            )
        if name.startswith('batchnorm'):
            kw = {
                'num_features': channels
            }
        elif 'layernorm' == name:
            kw = {
                'normalized_shape': channels
            }
        elif 'groupnorm' == name:
            kw = {
                'num_groups': 32,
                'num_channels': channels
            }
    return NORMALIZATION[name](**kw)
