"""Tests for io.
Covers expected behaviors to guard against regressions in the GreyML stack.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
import greyml  # noqa: E402
from greyml.tensor import _ga_tensor_save, _ga_tensor_load  # noqa: E402


@pytest.mark.skipif(_ga_tensor_save is None or _ga_tensor_load is None, reason="Tensor save/load not available")
def test_tensor_save_load_roundtrip(tmp_path):
    t = greyml.Tensor([[1, 2], [3, 4]], dtype=np.float32)
    path = tmp_path / "tensor.gat"
    greyml.save(t, str(path))
    loaded = greyml.load(str(path))
    np.testing.assert_array_equal(t.numpy(), loaded.numpy())
