import numpy as np

from greyml import Tensor
from greyml.ml import tree, svm, cluster, neighbors


def test_decision_tree_classifier_predict_shape():
    X = Tensor(np.array([[0.0], [1.0], [2.0]], dtype=np.float32))
    y = Tensor(np.array([0, 1, 1], dtype=np.int64))
    clf = tree.DecisionTreeClassifier(max_depth=2).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (X.shape[0],)


def test_svc_predict_shape():
    X = Tensor(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    y = Tensor(np.array([0, 1, 1, 0], dtype=np.int64))
    clf = svm.SVC(kernel="linear").fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (4,)


def test_kmeans_fit_predict_shape():
    data = Tensor(np.random.randn(10, 2).astype(np.float32))
    km = cluster.KMeans(n_clusters=2, max_iter=5)
    labels = km.fit_predict(data)
    assert labels.shape == (10,)


def test_knn_predict_returns_labels():
    # Simple two-class toy dataset
    train = Tensor(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    labels = Tensor(np.array([0, 1], dtype=np.int64))
    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(train, labels)
    preds = knn.predict(train)
    assert preds.shape == (2,)
