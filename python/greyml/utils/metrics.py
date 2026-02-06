"""Metrics utilities.
Simple evaluation metrics implemented with GreyML tensors.
"""

import numpy as np
from ..tensor import Tensor


def accuracy(logits: Tensor, targets: Tensor) -> float:
    """Compute classification accuracy."""
    preds = logits.numpy()
    if preds.ndim > 1:
        preds = preds.argmax(axis=-1)
    targets_np = targets.numpy()
    return float((preds == targets_np).mean())


def mse(pred: Tensor, target: Tensor) -> float:
    p = pred.numpy()
    t = target.numpy()
    return float(np.mean((p - t) ** 2))


def mae(pred: Tensor, target: Tensor) -> float:
    p = pred.numpy()
    t = target.numpy()
    return float(np.mean(np.abs(p - t)))


__all__ = ["accuracy", "mse", "mae"]
