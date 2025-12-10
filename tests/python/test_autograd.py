"""Tests for autograd.
Covers expected behaviors to guard against regressions in the GreyML stack.
"""

import os
import sys
import numpy as np
import pytest

# Ensure local package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from greyml import Tensor, cross_entropy, mse_loss  # noqa: E402
from greyml.tensor import no_grad  # noqa: E402


def test_add_mul_backward():
    x = Tensor([[1.0, 2.0]], dtype=np.float32, requires_grad=True)
    y = Tensor([[3.0, 4.0]], dtype=np.float32, requires_grad=True)
    z = (x + y) * (x + y)  # (x+y)^2
    loss = z.sum()
    loss.backward()
    expected = 2 * (x.numpy() + y.numpy())
    np.testing.assert_allclose(x.grad.numpy(), expected)
    np.testing.assert_allclose(y.grad.numpy(), expected)


def test_mean_backward_broadcast():
    x = Tensor([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32, requires_grad=True)
    y = Tensor([[2.0, 1.0]], dtype=np.float32, requires_grad=True)
    out = (x * y).mean(dim=0)
    out.sum().backward()
    # d/dx of mean over batch => y / N
    np.testing.assert_allclose(x.grad.numpy(), np.array([[1.0, 0.5], [1.0, 0.5]], dtype=np.float32))
    np.testing.assert_allclose(y.grad.numpy(), np.array([[2.0, 3.5]], dtype=np.float32))


def test_softmax_cross_entropy_grad():
    logits = Tensor([[2.0, 0.5]], dtype=np.float32, requires_grad=True)
    target = Tensor([0], dtype=np.int64)
    target_idx = int(target.numpy()[0])
    loss = cross_entropy(logits, target, reduction="mean")
    loss.backward()
    probs = logits.softmax(dim=-1).numpy()
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(one_hot.shape[0]), target_idx] = 1.0
    expected_grad = (probs - one_hot) / float(probs.size)
    np.testing.assert_allclose(logits.grad.numpy(), expected_grad, rtol=1e-5, atol=1e-5)


def test_no_grad_context_disables_tracking():
    x = Tensor([1.0, 2.0], dtype=np.float32, requires_grad=True)
    with no_grad():
        y = x * 3.0
    y_sum = y.sum()
    y_sum.backward()  # Should be a no-op because y has no history
    assert x.grad is None
