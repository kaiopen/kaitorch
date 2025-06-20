from typing import Any, Dict
from torch import nn


ACTIVATIONS = {
    'gelu': nn.GELU,
    'hardswish': nn.Hardswish,
    'hardsigmoid': nn.Hardsigmoid,
    'leakyrelu': nn.LeakyReLU,
    'mish': nn.Mish,
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'sigmoid': nn.Sigmoid,
    'silu': nn.SiLU
}


def activation(name: str, kw: Dict[str, Any] = {}) -> nn.Module:
    r'''

    #### Args:
    - name: the name of a activation function.
    - kw: the arguments to the activation function.

    #### Returns:
    - An activation function.

    '''
    return ACTIVATIONS[name](**kw)
