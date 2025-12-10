"""Tests for ml.
Covers expected behaviors to guard against regressions in the GreyML stack.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from greyml.ml import cluster  # noqa: E402
import greyml  # noqa: E402


def test_kmeans_smoke():
    km = cluster.KMeans(n_clusters=2)
    X = greyml.Tensor(np.array([[0.0], [0.1], [2.0], [2.1]], dtype=np.float32))
    labels = km.fit_predict(X)
    assert labels.shape[0] == 4
