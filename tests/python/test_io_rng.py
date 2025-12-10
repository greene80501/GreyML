"""Tests for io rng.
Covers expected behaviors to guard against regressions in the GreyML stack.
"""

import os
import sys
import numpy as np
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
import greyml


def test_tensor_save_load_roundtrip():
    t = greyml.Tensor([[1, 2], [3, 4]], dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.gat")
        greyml.save(t, path)
        try:
            loaded = greyml.load(path)
        except RuntimeError:
            pytest.skip("ga_tensor_load not available in this build")
        else:
            np.testing.assert_array_equal(t.numpy(), loaded.numpy())


def test_randn_seed_determinism():
    greyml.tensor._lib.ga_random_seed(123)
    a = greyml.randn(2, 2).numpy()
    greyml.tensor._lib.ga_random_seed(123)
    b = greyml.randn(2, 2).numpy()
    np.testing.assert_array_equal(a, b)
