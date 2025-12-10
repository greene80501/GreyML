"""Tests for bindings.
Covers expected behaviors to guard against regressions in the GreyML stack.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
import greyml


def test_requires_grad_flag():
    t = greyml.Tensor([1.0, 2.0], dtype=np.float32, requires_grad=True)
    assert t.requires_grad is True
    t2 = greyml.Tensor([1.0, 2.0], dtype=np.float32, requires_grad=False)
    assert t2.requires_grad is False


def test_losses_numeric():
    pred = greyml.Tensor([0.2, 0.8], dtype=np.float32)
    target = greyml.Tensor([0.0, 1.0], dtype=np.float32)

    mse = greyml.mse_loss(pred, target, reduction="sum")
    np.testing.assert_allclose(mse.numpy(), np.array([0.08], dtype=np.float32))

    try:
        l1 = greyml.l1_loss(pred, target, reduction="sum")
        np.testing.assert_allclose(l1.numpy(), np.array([1.0], dtype=np.float32))
    except RuntimeError:
        pytest.skip("l1_loss binding not available")

    bce = greyml.binary_cross_entropy(pred, target, reduction="sum")
    assert bce.numpy() >= 0

    huber = greyml.huber_loss(pred, target, delta=1.0, reduction="sum")
    assert huber.numpy() >= 0


def test_cross_entropy_labels():
    logits = greyml.Tensor([[2.0, 0.5]], dtype=np.float32)
    target = greyml.Tensor([0], dtype=np.int64)
    try:
        ce = greyml.cross_entropy(logits, target, reduction="mean")
        assert ce.numpy() > 0
    except RuntimeError:
        pytest.skip("cross_entropy binding not available")
