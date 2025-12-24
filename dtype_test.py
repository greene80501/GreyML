"""Ad-hoc dtype smoke test.
Validates tensor dtype handling against a few quick scenarios.
"""

import sys
import os
import numpy as np

# Make sure local "python/greyml" is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
from greyml import Tensor


def main():
    print("Dtype test")
    print("=" * 20)

    # float64
    x64 = Tensor([1.0, 2.0, 3.0], dtype=np.float64)
    arr64 = x64.numpy()
    print("float64 tensor:", arr64, arr64.dtype)
    assert arr64.dtype == np.float64

    # int32
    x32 = Tensor([1, 2, 3], dtype=np.int32)
    arr32 = x32.numpy()
    print("int32 tensor:", arr32, arr32.dtype)
    assert arr32.dtype == np.int32

    print("\n🎉 dtype_test.py passed.")


if __name__ == "__main__":
    main()
