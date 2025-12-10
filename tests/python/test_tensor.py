"""Tests for tensor.
Covers expected behaviors to guard against regressions in the GreyML stack.
"""

import pytest
import numpy as np
from greyml import Tensor

def test_tensor_creation():
    t = Tensor([1, 2, 3])
    assert t.shape == (3,)

def test_tensor_addition():
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a + b
    np.testing.assert_array_equal(c.numpy(), [5, 7, 9])

def test_tensor_matmul():
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    c = a @ b
    expected = np.array([[19, 22], [43, 50]])
    np.testing.assert_array_almost_equal(c.numpy(), expected)