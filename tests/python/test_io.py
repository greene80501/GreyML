import numpy as np
import pytest

from greyml import Tensor
from greyml.nn.layers import Linear
from greyml.utils import io
from greyml.tensor import _lib


def test_tensor_save_load_roundtrip(tmp_path):
    if not hasattr(_lib, "ga_tensor_save") or not hasattr(_lib, "ga_tensor_load"):
        pytest.skip("Tensor save/load not available in this build")
    t = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    path = tmp_path / "tensor.gat"
    io.save(t, str(path))
    loaded = io.load(str(path))
    np.testing.assert_allclose(loaded.numpy(), t.numpy())


def test_module_state_dict_roundtrip(tmp_path):
    model = Linear(3, 2)
    path = tmp_path / "model.npz"
    io.save(model, str(path))
    restored = io.load(str(path), Linear(3, 2))
    assert restored.weight.shape == model.weight.shape
    assert restored.bias.shape == model.bias.shape
