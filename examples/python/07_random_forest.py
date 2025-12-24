"""Random forest example on a noisy 1D dataset."""

import sys
sys.path.insert(0, "../../python")
import numpy as np  # noqa: E402
import greyml  # noqa: E402
from greyml.ml import forest  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.2, size=(20, 1)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    X_t = greyml.Tensor(X)
    y_t = greyml.Tensor(y)
    clf = forest.RandomForestClassifier(n_estimators=5, max_depth=3)
    clf.fit(X_t, y_t)
    preds = clf.predict(X_t).numpy()
    acc = (preds == y).mean()
    print("RF accuracy:", acc, "preds:", preds[:5])


if __name__ == "__main__":
    main()
