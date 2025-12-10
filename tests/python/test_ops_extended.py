"""Tests for ops extended.
Covers expected behaviors to guard against regressions in the GreyML stack.
"""

import os
import sys
import numpy as np
import pytest

# Ensure local package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from greyml import Tensor


def test_add_sub_mul_div_basic():
    a = Tensor([[1, 2], [3, 4]], dtype=np.float32)
    b = Tensor([[5, 6], [7, 8]], dtype=np.float32)

    print("\n[add_sub_mul_div] a:", a.numpy(), "b:", b.numpy())

    np.testing.assert_array_equal((a + b).numpy(), np.array([[6, 8], [10, 12]], dtype=np.float32))
    np.testing.assert_array_equal((b - a).numpy(), np.array([[4, 4], [4, 4]], dtype=np.float32))
    np.testing.assert_array_equal((a * b).numpy(), np.array([[5, 12], [21, 32]], dtype=np.float32))
    np.testing.assert_array_equal((b / a).numpy(), np.array([[5.0, 3.0], [7.0 / 3.0, 2.0]], dtype=np.float32))


def test_matmul_and_transpose():
    a = Tensor([[1, 2], [3, 4]], dtype=np.float32)
    b = Tensor([[5, 6], [7, 8]], dtype=np.float32)
    expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
    print("\n[matmul] a:", a.numpy(), "b:", b.numpy())
    np.testing.assert_array_almost_equal((a @ b).numpy(), expected)

    t = a.transpose()
    print("[transpose] result:", t.numpy())
    np.testing.assert_array_equal(t.numpy(), np.array([[1, 3], [2, 4]], dtype=np.float32))


def test_relu_sigmoid_softmax():
    x = Tensor([[-1.0, 0.0, 1.0]], dtype=np.float32)
    print("\n[activations] x:", x.numpy())

    relu_out = x.relu().numpy()
    np.testing.assert_array_equal(relu_out, np.array([[0.0, 0.0, 1.0]], dtype=np.float32))

    sigmoid_out = x.sigmoid().numpy()
    expected_sigmoid = np.array([[0.26894142, 0.5, 0.73105858]], dtype=np.float32)
    print("[sigmoid] out:", sigmoid_out)
    np.testing.assert_allclose(sigmoid_out, expected_sigmoid, rtol=1e-5)

    sm = x.softmax(dim=-1).numpy()
    print("[softmax] out:", sm)
    assert sm.shape == (1, 3)
    assert np.all(np.isfinite(sm))
    assert np.all(sm >= 0)


def test_sum_and_mean_dim():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    print("\n[reduce] x:", x.numpy())
    np.testing.assert_array_equal(x.sum(dim=0).numpy(), np.array([4.0, 6.0], dtype=np.float32))
    np.testing.assert_array_equal(x.sum(dim=1).numpy(), np.array([3.0, 7.0], dtype=np.float32))

    np.testing.assert_array_equal(x.mean(dim=0).numpy(), np.array([2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(x.mean(dim=1).numpy(), np.array([1.5, 3.5], dtype=np.float32))


def test_reshape_unsqueeze_squeeze():
    x = Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    print("\n[shape] x:", x.numpy(), "shape:", x.shape)
    r = x.reshape(3, 2)
    print("[reshape] ->", r.numpy())
    np.testing.assert_array_equal(r.numpy(), np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32))

    u = x.unsqueeze(0)
    print("[unsqueeze] shape:", u.shape)
    assert u.shape == (1, 2, 3)

    s = u.squeeze(0)
    print("[squeeze] shape:", s.shape)
    assert s.shape == (2, 3)


if __name__ == "__main__":
    # Allow direct execution: run pytest with verbose output and live prints
    import pytest
    raise SystemExit(pytest.main(["-s", __file__]))
