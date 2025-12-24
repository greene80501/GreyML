import numpy as np
import pytest

from greyml import Tensor
from greyml.utils.metrics import accuracy, mse, mae


def test_accuracy_with_logits():
    logits = Tensor(np.array([[1.0, 0.1], [0.2, 1.5]], dtype=np.float32))
    targets = Tensor(np.array([0, 1], dtype=np.int64))
    acc = accuracy(logits, targets)
    assert acc == 1.0


def test_mse_and_mae():
    pred = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    target = Tensor(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    assert mse(pred, target) == pytest.approx(5.0 / 3.0)
    assert mae(pred, target) == pytest.approx(1.0)
