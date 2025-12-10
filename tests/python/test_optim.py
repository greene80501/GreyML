"""Tests for optim.
Covers expected behaviors to guard against regressions in the GreyML stack.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from greyml import Tensor, mse_loss  # noqa: E402
from greyml.optim import SGD, Adam  # noqa: E402


def _quad_loss(param: Tensor) -> Tensor:
    """Simple quadratic loss helper."""
    return (param * param).sum()


def test_sgd_updates_and_zero_grad():
    w = Tensor([1.0, -1.0], dtype=np.float32, requires_grad=True)
    opt = SGD([w], lr=0.1)

    loss1 = _quad_loss(w)
    loss1.backward()
    opt.step()
    # Parameters should have moved toward zero
    assert np.all(np.abs(w.numpy()) < np.array([1.0, 1.0], dtype=np.float32))

    # zero_grad clears accumulated gradients
    opt.zero_grad()
    assert w.grad is None


def test_adam_converges_on_quadratic():
    w = Tensor([2.0, -3.0], dtype=np.float32, requires_grad=True)
    opt = Adam([w], lr=0.05)
    initial = _quad_loss(w).numpy()
    for _ in range(10):
        opt.zero_grad()
        loss = _quad_loss(w)
        loss.backward()
        opt.step()
    final = _quad_loss(w).numpy()
    assert final < initial
