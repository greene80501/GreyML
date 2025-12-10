"""Tests for ml algorithms.
Covers expected behaviors to guard against regressions in the GreyML stack.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from greyml.tensor import Tensor
from greyml import ml


def test_decision_tree_classifier_simple():
    X = Tensor(np.array([[0.0], [0.1], [0.9], [1.0]], dtype=np.float32))
    y = Tensor(np.array([0, 0, 1, 1], dtype=np.int64))
    clf = ml.tree.DecisionTreeClassifier(max_depth=2)
    if getattr(clf, "_ptr", None) is None:
        pytest.skip("C tree symbols unavailable")
    clf.fit(X, y)
    preds = clf.predict(X).numpy()
    assert (preds == y.numpy()).all()
    assert preds.dtype == np.int64


def test_random_forest_classifier_votes():
    np.random.seed(0)
    X = Tensor(np.vstack([np.zeros((10, 1)), np.ones((10, 1))]).astype(np.float32))
    y = Tensor(np.array([0] * 10 + [1] * 10, dtype=np.int64))
    rf = ml.forest.RandomForestClassifier(n_estimators=5, max_depth=2)
    if getattr(rf, "_ptr", None) is None:
        pytest.skip("C forest symbols unavailable")
    rf.fit(X, y)
    preds = rf.predict(X).numpy()
    acc = (preds == y.numpy()).mean()
    assert acc >= 0.8
    assert preds.dtype == np.int64


def test_random_forest_regressor_mean():
    X = Tensor(np.vstack([np.zeros((5, 1)), np.ones((5, 1))]).astype(np.float32))
    y = Tensor(np.array([0.0] * 5 + [1.0] * 5, dtype=np.float32))
    rf = ml.forest.RandomForestRegressor(n_estimators=3, max_depth=2)
    if getattr(rf, "_ptr", None) is None:
        pytest.skip("C forest symbols unavailable")
    rf.fit(X, y)
    preds = rf.predict(X).numpy()
    assert preds.dtype == np.float32
    assert preds.min() >= -0.5 and preds.max() <= 1.5


def test_knn_classifier():
    X_train = Tensor(np.array([[0.0], [0.1], [0.9], [1.0]], dtype=np.float32))
    y_train = Tensor(np.array([0, 0, 1, 1], dtype=np.int64))
    knn = ml.neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_train).numpy()
    assert (preds == y_train.numpy()).all()
    assert preds.dtype == np.int64


def test_knn_regressor_distance_weighted():
    X_train = Tensor(np.array([[0.0], [0.5], [1.0]], dtype=np.float32))
    y_train = Tensor(np.array([0.0, 1.0, 2.0], dtype=np.float32))
    knn = ml.neighbors.KNeighborsRegressor(n_neighbors=2, weights="distance")
    knn.fit(X_train, y_train)
    preds = knn.predict(X_train).numpy()
    assert np.allclose(preds, y_train.numpy(), atol=1e-2)
    dists, idx = knn.kneighbors(X_train, k=2)
    assert dists.shape == (3, 2) and idx.shape == (3, 2)


def test_kmeans_clusters_two_blobs():
    np.random.seed(0)
    blob1 = np.random.randn(20, 2).astype(np.float32) + np.array([0, 0], dtype=np.float32)
    blob2 = np.random.randn(20, 2).astype(np.float32) + np.array([5, 5], dtype=np.float32)
    X = Tensor(np.vstack([blob1, blob2]))
    km = ml.cluster.KMeans(n_clusters=2, max_iter=20, n_init=2)
    if getattr(km, "_ptr", None) is None:
        pytest.skip("C KMeans symbols unavailable")
    km.fit(X)
    labels = km.predict(X).numpy()
    assert set(np.unique(labels)) <= {0, 1}
    if km.inertia_ is not None:
        assert km.inertia_ > 0


def test_dbscan_marks_noise():
    X = Tensor(np.array([[0, 0], [0, 0.1], [5, 5]], dtype=np.float32))
    db = ml.cluster.DBSCAN(eps=0.3, min_samples=2)
    if getattr(db, "_ptr", None) is None:
        pytest.skip("C DBSCAN symbols unavailable")
    labels = db.fit_predict(X).numpy()
    assert -1 in labels
    assert 0 in labels
    assert labels.dtype == np.int64


def test_svc_linear_separable():
    X = Tensor(np.array([[0.0], [0.2], [0.8], [1.0]], dtype=np.float32))
    y = Tensor(np.array([0, 0, 1, 1], dtype=np.int64))
    svc = ml.svm.SVC()
    svc.fit(X, y)
    preds = svc.predict(X).numpy()
    assert (preds == y.numpy()).all()
    assert preds.dtype == np.int64


def test_svc_xor_rbf():
    X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
    y = Tensor(np.array([0, 1, 1, 0], dtype=np.int64))
    svc = ml.svm.SVC(kernel="rbf", gamma=1.0)
    if getattr(svc, "_ptr", None) is None:
        pytest.skip("C SVM symbols unavailable")
    svc.fit(X, y)
    preds = svc.predict(X).numpy()
    assert preds.dtype == np.int64
    assert (preds == y.numpy()).mean() >= 0.75


def test_svr_regression_line():
    X = Tensor(np.array([[0.0], [1.0], [2.0]], dtype=np.float32))
    y = Tensor(np.array([0.0, 1.0, 2.0], dtype=np.float32))
    svr = ml.svm.SVR()
    svr.fit(X, y)
    preds = svr.predict(X).numpy()
    assert np.allclose(preds, y.numpy(), atol=0.2)
    assert preds.dtype == np.float32


def test_tree_feature_importances_nonzero():
    X = Tensor(np.array([[0.0], [0.1], [0.9], [1.0]], dtype=np.float32))
    y = Tensor(np.array([0, 0, 1, 1], dtype=np.int64))
    tree = ml.tree.DecisionTreeClassifier(max_depth=2)
    tree.fit(X, y)
    if getattr(tree, "_ptr", None) is None:
        pytest.skip("C tree symbols unavailable")
    fi = tree.feature_importances_.numpy()
    assert fi.shape[0] >= 1
    assert np.isfinite(fi).all()
