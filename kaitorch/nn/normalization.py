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
    in_channels: Optional[int] = None,
    device: Optional[Any] = None,
    dtype: Optional[Any] = None
) -> nn.Module:
    r'''

    #### Args:
    - name: the name of a normalization function.
    - kw: the arguments to the normalization function.
    - in_channels: the number of input channels.

    #### Returns:
    - A normalization function.

    '''
    if kw is None:
        if in_channels is None:
            raise ValueError(
                'Either the `kw` or the `in_channels` should not be a `None`.'
            )
        if name.startswith('batchnorm'):
            kw = {
                'num_features': in_channels
            }
        elif 'layernorm' == name:
            kw = {
                'normalized_shape': in_channels
            }
        elif 'groupnorm' == name:
            kw = {
                'num_groups': 32,
                'num_channels': in_channels
            }

    if device is not None and getattr(kw, 'device', None) is None:
        kw['device'] = device
    if dtype is not None and getattr(kw, 'dtype', None) is None:
        kw['dtype'] = dtype

    return NORMALIZATION[name](**kw)
