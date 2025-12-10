"""Tests for nn.
Covers expected behaviors to guard against regressions in the GreyML stack.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from greyml.nn import layers  # noqa: E402
from greyml import Tensor  # noqa: E402


def test_module_parameters_and_modes():
    lin = layers.Linear(2, 3, bias=True)
    params = lin.parameters()
    assert len(params) in (1, 2)  # bias may be optional
    lin.train(False)
    assert lin.training is False
    lin.eval()
    assert lin.training is False
    lin.train(True)
    assert lin.training is True


def test_linear_backward_autograd():
    lin = layers.Linear(2, 1, bias=True)
    # Enable gradients on parameters
    lin.weight._requires_grad = True
    lin.bias._requires_grad = True
    x = Tensor(np.array([[1.0, 2.0]], dtype=np.float32), requires_grad=True)
    out = lin(x)
    loss = out.mean()
    loss.backward()
    assert x.grad is not None
    assert lin.weight.grad is not None
    assert lin.bias.grad is not None
