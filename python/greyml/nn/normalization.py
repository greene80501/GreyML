"""Normalization layers.
Batch/Layer normalization wrappers with training/eval handling.
"""

import numpy as np
from .module import Module
from ..tensor import Tensor


class BatchNorm2d(Module):
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.running_mean = np.zeros((num_features,), dtype=np.float32)
        self.running_var = np.ones((num_features,), dtype=np.float32)
        self.weight = Tensor(np.ones((num_features,), dtype=np.float32), requires_grad=True)
        self.bias = Tensor(np.zeros((num_features,), dtype=np.float32), requires_grad=True)
        self._parameters["weight"] = self.weight
        self._parameters["bias"] = self.bias

    def forward(self, x: Tensor) -> Tensor:
        # x: (N,C,H,W)
        N, C, H, W = x.shape
        x_reshaped = x.reshape((N, C, H * W))
        mean = x_reshaped.mean(dim=2, keepdim=True).mean(dim=0)  # (C,)
        diff = x_reshaped - mean.reshape((1, C, 1))
        var = (diff * diff).mean(dim=2, keepdim=True).mean(dim=0)

        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.numpy()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.numpy()
            mean_use = mean
            var_use = var
        else:
            mean_use = Tensor(self.running_mean, dtype=np.float32)
            var_use = Tensor(self.running_var, dtype=np.float32)

        x_centered = x - mean_use.reshape((1, C, 1, 1))
        try:
            denom = (var_use.reshape((1, C, 1, 1)) + Tensor(np.array(self.eps, dtype=np.float32))).sqrt()
            x_norm = x_centered / denom
        except AttributeError:
            denom_np = np.sqrt(var_use.numpy().reshape((1, C, 1, 1)) + self.eps)
            x_norm = Tensor(x_centered.numpy() / denom_np.astype(np.float32), dtype=np.float32)
        out = x_norm * self.weight.reshape((1, C, 1, 1)) + self.bias.reshape((1, C, 1, 1))
        return out


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Tensor(np.ones(self.normalized_shape, dtype=np.float32), requires_grad=True)
        self.bias = Tensor(np.zeros(self.normalized_shape, dtype=np.float32), requires_grad=True)
        self._parameters["weight"] = self.weight
        self._parameters["bias"] = self.bias

    def forward(self, x: Tensor) -> Tensor:
        # normalize over last len(normalized_shape) dims
        dims = len(self.normalized_shape)
        x_view = x.reshape((-1, int(np.prod(self.normalized_shape))))
        mean = x_view.mean(dim=1, keepdim=True)
        diff = x_view - mean
        var = (diff * diff).mean(dim=1, keepdim=True)
        try:
            x_norm = diff / (var + Tensor(np.array(self.eps, dtype=np.float32))).sqrt()
        except AttributeError:
            x_norm = Tensor(diff.numpy() / np.sqrt(var.numpy() + self.eps), dtype=np.float32)
        x_norm = x_norm.reshape(x.shape)
        return x_norm * self.weight + self.bias


__all__ = ["BatchNorm2d", "LayerNorm"]
