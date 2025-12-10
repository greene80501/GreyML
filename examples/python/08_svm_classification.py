"""SVM classification on XOR pattern with RBF kernel."""

import sys
sys.path.insert(0, "../../python")
import numpy as np  # noqa: E402
import greyml  # noqa: E402
from greyml.ml import svm  # noqa: E402


def main():
    X = greyml.Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
    y = greyml.Tensor(np.array([0, 1, 1, 0], dtype=np.int64))
    clf = svm.SVC(kernel="rbf", gamma=1.0)
    clf.fit(X, y)
    preds = clf.predict(X).numpy()
    acc = (preds == y.numpy()).mean()
    print("SVM XOR accuracy:", acc, "preds:", preds)


if __name__ == "__main__":
    main()
