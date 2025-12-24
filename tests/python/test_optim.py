import numpy as np

from greyml import Tensor, mse_loss
from greyml.optim import SGD, Adam


def _toy_setup():
    w = Tensor(np.array([[1.0]], dtype=np.float32), requires_grad=True)
    x = Tensor(np.array([[2.0]], dtype=np.float32))
    target = Tensor(np.array([[0.0]], dtype=np.float32))
    pred = x @ w
    loss = mse_loss(pred, target)
    return w, loss


def test_sgd_updates_parameter():
    w, loss = _toy_setup()
    loss.backward()
    before = w.numpy().copy()
    opt = SGD([w], lr=0.1)
    opt.step()
    assert not np.allclose(w.numpy(), before)


def test_adam_updates_parameter():
    w, loss = _toy_setup()
    loss.backward()
    before = w.numpy().copy()
    opt = Adam([w], lr=0.01)
    opt.step()
    assert not np.allclose(w.numpy(), before)
