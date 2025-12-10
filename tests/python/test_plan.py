"""
Comprehensive test plan scaffold for future coverage.

Most tests are marked skipped because the Python bindings and implementations
are not yet complete. They document intended behavior so we can unskip as
features land.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))


@pytest.mark.skip(reason="Autograd graph/backward not implemented in Python bindings yet")
def test_autograd_add_mul_backward():
    from greyml import Tensor

    x = Tensor([[1.0, 2.0]], dtype=np.float32, requires_grad=True)
    y = Tensor([[3.0, 4.0]], dtype=np.float32, requires_grad=True)
    z = (x + y) * (x + y)
    z.backward()
    # Expected gradients: dz/dx = 2*(x+y), dz/dy = 2*(x+y)
    np.testing.assert_allclose(x.grad.numpy(), 2 * (x.numpy() + y.numpy()))
    np.testing.assert_allclose(y.grad.numpy(), 2 * (x.numpy() + y.numpy()))


@pytest.mark.skip(reason="Conv2d not exposed in Python bindings yet")
def test_conv2d_forward_shape():
    from greyml.nn import layers
    conv = layers.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
    x = np.random.randn(1, 1, 8, 8).astype(np.float32)
    out = conv(Tensor(x))
    assert out.shape == (1, 2, 8, 8)


@pytest.mark.skip(reason="Pooling ops not exposed in Python bindings yet")
def test_pooling_forward_shape():
    from greyml.nn import functional as F
    x = Tensor(np.random.randn(1, 1, 8, 8).astype(np.float32))
    y = F.max_pool2d(x, kernel_size=2, stride=2)
    assert y.shape == (1, 1, 4, 4)


@pytest.mark.skip(reason="Loss bindings not implemented")
def test_losses_ce_bce_huber():
    from greyml import loss
    logits = Tensor([[2.0, 0.5]], dtype=np.float32)
    target = Tensor([0], dtype=np.int64)
    ce = loss.cross_entropy(logits, target)
    assert ce.item() > 0

    pred = Tensor([0.2, 0.8], dtype=np.float32)
    target = Tensor([0.0, 1.0], dtype=np.float32)
    bce = loss.binary_cross_entropy(pred, target)
    assert bce.item() > 0

    huber = loss.huber(pred, target, delta=1.0)
    assert huber.item() >= 0


@pytest.mark.skip(reason="Optimizers not wired to Python yet")
def test_sgd_and_adam_step():
    from greyml import Tensor
    from greyml import optim

    w = Tensor([1.0, -1.0], dtype=np.float32, requires_grad=True)
    opt = optim.SGD([w], lr=0.1, momentum=0.0)
    loss = (w * w).sum()
    loss.backward()
    opt.step()
    assert not np.allclose(w.numpy(), np.array([1.0, -1.0], dtype=np.float32))


@pytest.mark.skip(reason="Classical ML bindings not ready")
def test_tree_and_svm_basic():
    from greyml import ml
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    clf = ml.DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == y.shape


@pytest.mark.skip(reason="IO functions not exposed")
def test_tensor_save_load_roundtrip(tmp_path):
    from greyml import Tensor
    t = Tensor([[1, 2], [3, 4]], dtype=np.float32)
    path = tmp_path / "t.gat"
    t.save(str(path))
    loaded = Tensor.load(str(path))
    np.testing.assert_array_equal(t.numpy(), loaded.numpy())
