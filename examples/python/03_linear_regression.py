"""GreyML example: 03 linear regression.
Shows a minimal usage pattern you can copy into your own experiments.
"""

\"\"\"Placeholder linear regression example.\"\"\"

import numpy as np
import sys
sys.path.insert(0, \"../../python\")
import greyml  # noqa: E402
from greyml.nn import layers  # noqa: E402


def main():
    model = layers.Linear(1, 1)
    x = greyml.Tensor(np.array([[1.0], [2.0], [3.0]], dtype=np.float32), requires_grad=True)
    y = greyml.Tensor(np.array([[2.0], [4.0], [6.0]], dtype=np.float32))
    pred = model(x)
    loss = ((pred - y) * (pred - y)).mean()
    loss.backward()
    print(\"pred:\", pred.numpy(), \"loss:\", loss.numpy())


if __name__ == \"__main__\":
    main()
