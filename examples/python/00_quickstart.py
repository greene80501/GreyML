"""
Quickstart example that runs with the current Python-only stack.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
import greyml


def main():
    print("GreyML quickstart")
    x = greyml.Tensor([[1.0, 2.0], [3.0, 4.0]])
    w = greyml.Tensor([[0.5, -0.25], [0.1, 0.2]])
    y = x @ w
    print("Matmul result:\n", y.numpy())

    lin = greyml.nn.layers.Linear(2, 3)
    out = lin(x)
    print("Linear output shape:", out.shape)

    km = greyml.ml.cluster.KMeans(n_clusters=2, max_iter=10)
    km.fit(greyml.Tensor(np.array([[0, 0], [0.1, 0.2], [5, 5], [5.1, 5.2]], dtype=np.float32)))
    labels = km.predict(greyml.Tensor(np.array([[0, 0], [5, 5]], dtype=np.float32)))
    print("KMeans labels:", labels.numpy())


if __name__ == "__main__":
    main()
