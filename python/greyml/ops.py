"""Python-side operation helpers for GreyML.
Wraps Tensor methods to provide a simple functional API (mirrors C ops)."""

from .tensor import Tensor

def relu(x: Tensor) -> Tensor:
    return x.relu()

def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    return x.softmax(dim=dim)

def add(a: Tensor, b: Tensor) -> Tensor:
    return a + b

def mul(a: Tensor, b: Tensor) -> Tensor:
    return a * b

def matmul(a: Tensor, b: Tensor) -> Tensor:
    return a.matmul(b)

def sum(x: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    return x.sum(dim=dim, keepdim=keepdim)

__all__ = ["relu", "sigmoid", "softmax", "add", "mul", "matmul", "sum"]
