"""Parameter initialization helpers.
Implements Xavier/Kaiming and related initializers for GreyML tensors.
"""

import numpy as np
from ..tensor import Tensor


def uniform_(tensor: Tensor, low: float, high: float):
    arr = np.random.uniform(low, high, size=tensor.shape).astype(tensor.dtype)
    tensor._lib.ga_tensor_copy_from(tensor._c_ptr, arr.ctypes.data)
    return tensor


def xavier_uniform_(tensor: Tensor):
    fan_in = tensor.shape[-1]
    fan_out = tensor.shape[0]
    bound = np.sqrt(6.0 / float(fan_in + fan_out))
    return uniform_(tensor, -bound, bound)


def kaiming_uniform_(tensor: Tensor):
    fan_in = tensor.shape[-1]
    bound = np.sqrt(1.0 / float(fan_in))
    return uniform_(tensor, -bound, bound)


__all__ = ["uniform_", "xavier_uniform_", "kaiming_uniform_"]
