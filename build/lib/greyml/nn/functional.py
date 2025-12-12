"""Functional NN API.
Stateless layer-style helpers mirroring PyTorch-style functional ops.
"""

from ..tensor import Tensor
import greyml.ops as ops

def relu(x: Tensor) -> Tensor:
    return ops.relu(x)

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    return ops.softmax(x, dim)

def mse_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    reduction_map = {"none": 0, "mean": 1, "sum": 2}
    return ops.mse_loss(pred, target, reduction_map[reduction])