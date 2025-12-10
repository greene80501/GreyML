"""K-Means clustering on two blobs."""

import sys
sys.path.insert(0, "../../python")
import numpy as np  # noqa: E402
import greyml  # noqa: E402
from greyml.ml import cluster  # noqa: E402


def main():
    rng = np.random.default_rng(1)
    blob1 = rng.normal(0, 0.2, size=(10, 2)).astype(np.float32)
    blob2 = rng.normal(3, 0.2, size=(10, 2)).astype(np.float32)
    X = greyml.Tensor(np.vstack([blob1, blob2]))
    km = cluster.KMeans(n_clusters=2, n_init=3, max_iter=50)
    labels = km.fit_predict(X).numpy()
    print("KMeans labels:", labels)
    if km.inertia_ is not None:
        print("Inertia:", km.inertia_, "iters:", km.n_iter_)


if __name__ == "__main__":
    main()
