"""GreyML neural network utilities.
Python-side layers and helpers that wrap the C backend.
"""

from ..tensor import Tensor
from . import layers, activation, container, dropout, functional, init, normalization, pooling, rnn, attention

def uniform(tensor: Tensor, low: float = 0.0, high: float = 1.0) -> None:
    # Would call ga_init_uniform
    tensor.fill_(low)

def normal(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    # Would call ga_init_normal
    pass

def xavier_uniform(tensor: Tensor) -> None:
    # Would call ga_init_xavier_uniform
    pass

def kaiming_uniform(tensor: Tensor) -> None:
    # Would call ga_init_kaiming_uniform
    import math
    fan_in = tensor.shape[1] if len(tensor.shape) > 1 else tensor.shape[0]
    bound = math.sqrt(1.0 / fan_in)
    uniform(tensor, -bound, bound)

__all__ = [
    "uniform",
    "normal",
    "xavier_uniform",
    "kaiming_uniform",
    "layers",
    "activation",
    "container",
    "dropout",
    "functional",
    "init",
    "normalization",
    "pooling",
    "rnn",
    "attention",
]
