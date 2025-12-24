"""Decision tree demo on a toy OR dataset."""

import sys
sys.path.insert(0, "../../python")
from greyml.ml import tree  # noqa: E402
import numpy as np
import greyml  # noqa: E402


def main():
    X = greyml.Tensor(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    y = greyml.Tensor(np.array([0, 1, 1, 1], dtype=np.int64))
    clf = tree.DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)
    preds = clf.predict(X).numpy()
    acc = (preds == y.numpy()).mean()
    print("Tree predictions:", preds, "accuracy:", acc)
    if getattr(clf, "_ptr", None):
        print("Feature importances:", clf.feature_importances_.numpy())


if __name__ == "__main__":
    main()
