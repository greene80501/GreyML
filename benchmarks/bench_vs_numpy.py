"""
Simple benchmarks comparing greyml Tensor ops to numpy.

Run:
    python benchmarks/bench_vs_numpy.py
"""
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from greyml import Tensor


def bench_matmul(n=256, reps=5):
    a_np = np.random.randn(n, n).astype(np.float32)
    b_np = np.random.randn(n, n).astype(np.float32)

    # numpy
    t0 = time.time()
    for _ in range(reps):
        _ = a_np @ b_np
    numpy_time = (time.time() - t0) / reps

    # greyml
    a = Tensor(a_np)
    b = Tensor(b_np)
    t1 = time.time()
    for _ in range(reps):
        _ = a @ b
    grey_time = (time.time() - t1) / reps

    return numpy_time, grey_time


def bench_conv2d(n=32, c=8, k=3, reps=3):
    # Placeholder: just measure Tensor creation for now if conv not wired
    arr = np.random.randn(1, c, n, n).astype(np.float32)
    t0 = time.time()
    for _ in range(reps):
        Tensor(arr)
    return (time.time() - t0) / reps


def main():
    np_time, ga_time = bench_matmul()
    print(f"Matmul 256x256: numpy {np_time:.4f}s, greyml {ga_time:.4f}s (per run)")
    conv_time = bench_conv2d()
    print(f"Conv2d placeholder (Tensor creation) avg: {conv_time:.6f}s")


if __name__ == "__main__":
    main()
