import numpy as np
import pytest

from greyml import Tensor


def test_tensor_creation_and_numpy_roundtrip():
    t = Tensor([1, 2, 3], dtype=np.float32)
    assert t.shape == (3,)
    assert t.ndim == 1
    assert t.dtype == np.float32
    np.testing.assert_array_equal(t.numpy(), np.array([1, 2, 3], dtype=np.float32))


def test_basic_arithmetic_and_matmul():
    a = Tensor(np.ones((2, 2), dtype=np.float32))
    b = Tensor(np.full((2, 2), 2.0, dtype=np.float32))
    add = a + b
    np.testing.assert_allclose(add.numpy(), np.full((2, 2), 3.0, dtype=np.float32))

    c = a @ b
    expected = np.ones((2, 2), dtype=np.float32) @ np.full((2, 2), 2.0, dtype=np.float32)
    np.testing.assert_allclose(c.numpy(), expected)


def test_reshape_unsqueeze_squeeze_transpose():
    t = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    r = t.reshape((3, 2))
    assert r.shape == (3, 2)

    u = t.unsqueeze(0)
    assert u.shape == (1, 2, 3)
    s = u.squeeze(0)
    assert s.shape == (2, 3)

    tr = t.transpose()
    np.testing.assert_allclose(tr.numpy(), t.numpy().T)


def test_softmax_and_sum_keepdim():
    t = Tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    sm = t.softmax(dim=-1)
    np.testing.assert_allclose(sm.numpy().sum(axis=-1), np.ones((1,), dtype=np.float32), rtol=1e-4, atol=1e-4)

    summed = t.sum(dim=1, keepdim=True)
    assert summed.shape == (1, 1)
    np.testing.assert_allclose(summed.numpy(), np.array([[6.0]], dtype=np.float32))


def test_autograd_backward_on_simple_graph():
    x_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = Tensor(x_data, requires_grad=True)
    y = x * x
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), 2 * x_data, rtol=1e-4, atol=1e-4)


def test_clone_detach_and_grad_flag():
    x = Tensor(np.array([5.0], dtype=np.float32), requires_grad=True)
    clone = x.clone()
    assert clone.requires_grad

    det = clone.detach()
    assert not det.requires_grad
    np.testing.assert_allclose(det.numpy(), clone.numpy())
