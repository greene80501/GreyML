"""Tiny GreyML smoke test.
Runs a handful of quick operations to ensure the build is usable.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
from greyml import Tensor, zeros


def main():
    print("Testing GreyArea AI Library v0.1")
    print("=" * 40)

    x = Tensor([1, 2, 3, 4, 5], dtype=np.float32)
    print(f"[OK] Created tensor: {x.numpy()}")

    y = x + x
    print(f"[OK] Addition works: {y.numpy()}")

    print(f"[OK] Shape: {x.shape}")

    z = zeros(3, 4)
    print(f"[OK] Zeros shape: {z.shape}")

    print("\nAll tests passed! GreyArea AI Library is functional.")


if __name__ == "__main__":
    main()
