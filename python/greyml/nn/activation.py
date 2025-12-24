"""Activation helpers for GreyML NN.
Python wrappers around core activation ops used by layers and functional API.
"""

from .module import Module
from ..tensor import Tensor
import greyml.ops as ops

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.sigmoid(x)

class Softmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        return ops.softmax(x, self.dim)