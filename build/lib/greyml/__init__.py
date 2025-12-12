"""GreyML Python bindings.
Bridges Python code to the C backend and autograd system.
"""

__version__ = "0.1.0"

from .tensor import (
    Tensor,
    zeros,
    ones,
    randn,
    load_csv,
    mse_loss,
    l1_loss,
    binary_cross_entropy,
    huber_loss,
    cross_entropy,
)
from . import nn
from . import optim
from . import ml
from . import ops
from .utils.io import save, load

__all__ = [
    "Tensor",
    "zeros",
    "ones",
    "randn",
    "load_csv",
    "mse_loss",
    "l1_loss",
    "binary_cross_entropy",
    "huber_loss",
    "cross_entropy",
    "nn",
    "optim",
    "ml",
    "ops",
    "save",
    "load",
]
