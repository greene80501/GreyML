"""GreyML example: 02 autograd intro.
Shows a minimal usage pattern you can copy into your own experiments.
"""

\"\"\"Placeholder autograd intro example.\"\"\"

import numpy as np
import sys
sys.path.insert(0, \"../../python\")
import greyml  # noqa: E402


def main():
    x = greyml.Tensor(np.array([2.0, 3.0], dtype=np.float32), requires_grad=True)
    y = (x * x).sum()
    y.backward()
    print(\"x:\", x.numpy(), \"grad:\", x.grad.numpy())


if __name__ == \"__main__\":
    main()
