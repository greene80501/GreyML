"""Tests for integration.
Covers expected behaviors to guard against regressions in the GreyML stack.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from greyml import Tensor, mse_loss  # noqa: E402
from greyml.optim import SGD  # noqa: E402


def test_linear_regression_sgd_training():
    # y = 2x + 1 synthetic data
    x = Tensor([[0.0], [1.0], [2.0]], dtype=np.float32)
    y_true = Tensor([[1.0], [3.0], [5.0]], dtype=np.float32)

    w = Tensor([[0.0]], dtype=np.float32, requires_grad=True)
    b = Tensor([[0.0]], dtype=np.float32, requires_grad=True)
    opt = SGD([w, b], lr=0.1)

    initial_loss = mse_loss(x @ w + b, y_true, reduction="mean").numpy()
    for _ in range(5):
        opt.zero_grad()
        preds = x @ w + b
        loss = mse_loss(preds, y_true, reduction="mean")
        loss.backward()
        opt.step()
    final_loss = mse_loss(x @ w + b, y_true, reduction="mean").numpy()
    assert final_loss < initial_loss
