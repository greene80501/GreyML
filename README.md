# GreyML (C17 + Python, Windows-first ML runtime)

GreyML is a small, inspectable machine-learning stack: a C17 core (`greyarea.dll`) that implements tensors, math kernels, autograd, neural-network layers, and classic ML; plus a thin Python API (`greyml`) that mirrors the core while staying NumPy-friendly. Everything here is tuned for Windows/MSVC first and keeps dependencies to a minimum (NumPy only on the Python side).

## Highlights
- C backend you can read: tensors, elementwise/reduction ops, matmul, conv/pooling, autograd tape, NN/ML kernels, SIMD helpers.
- Python ergonomics: `Tensor` with autograd, NN/optim/ml modules, dataset/dataloader utilities, and fallbacks to NumPy when C symbols are missing.
- Classical ML included: decision trees/forests, SVM (class/regression), KNN, KMeans, DBSCAN.
- Windows-first build: CMake presets for MSVC + Ninja; CPU-only today with optional BLAS/OpenMP flags.
- Batteries included for demos: quick smoke scripts and the end-to-end demo; full C/Python test suites are omitted in this trimmed copy (grab from the original repo if needed).

## Getting Started
1) **Prereqs**: Windows 11, Python 3.10+, NumPy, CMake 3.20+, MSVC 2022 (Build Tools) or Clang-CL. Optional: Ninja, OpenBLAS.
2) **Build the DLL** (copies into `python/greyml/`):
   ```batch
   scripts\build.bat
   ```
3) **Install Python package (editable)**:
   ```bash
   pip install -e .
   ```
4) **Smoke tests**:
   ```bash
   python quick_test.py
   python dtype_test.py
   python demo_greyml.py
   ```
5) **Full suites**: Not included here; copy `tests/` from the original repo if you need pytest/ctest coverage.

## Repository Layout (essentials)
- `src/`: C17 backend (core tensor/memory/random, ops, autograd, NN, optim, loss, ML, IO, SIMD).
- `include/`: public headers (`greyarea.h` umbrella, GA_* component APIs).
- `python/greyml/`: Python bindings and API surface (`tensor.py`, `ops.py`, `nn/`, `optim/`, `ml/`, `data/`, `utils/`, `_lib_loader.py`, `_bindings.py`).
- `examples/`: Python and C samples (tensor basics, autograd, NN, ML, XOR/MNIST placeholders).
- `benchmarks/`: Matmul/conv microbenchmarks and NumPy comparison.
- `scripts/`: Build helper and MNIST downloader placeholder.
- `docs/`: Stubs plus a Windows setup note; see `main.md` in this repo for detailed docs.
- `tests/`: Not included in this trimmed snapshot; copy from the original repo if you need them.
- Generated artifacts to ignore: `build/`, `dist/`, `__pycache__/`, `.pytest_cache/`.

## Usage Snippets
**Tensor math & autograd**
```python
from greyml import Tensor, zeros
import numpy as np

x = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
y = x * 3 + 1
z = y.sum()
z.backward()
print("z:", z.item(), "grad:", x.grad.numpy())
```

**Neural layers + optimizer (Python-side)**
```python
from greyml import Tensor, mse_loss
from greyml.nn.layers import Linear
from greyml.optim import SGD
import numpy as np

model = Linear(2, 1)
model.weight._requires_grad = True
model.bias._requires_grad = True
opt = SGD(model.parameters(), lr=0.1)

x = Tensor(np.array([[1.0, 2.0]], dtype=np.float32), requires_grad=True)
y = Tensor(np.array([[5.0]], dtype=np.float32))
pred = model(x)
loss = mse_loss(pred, y)
loss.backward()
opt.step()
```

**Classical ML**
```python
from greyml import Tensor
from greyml.ml import svm, cluster
import numpy as np

xor = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32))
labels = Tensor(np.array([0,1,1,0], dtype=np.int64))
clf = svm.SVC(kernel="rbf", gamma=1.0).fit(xor, labels)
print("preds:", clf.predict(xor).numpy())

km = cluster.KMeans(n_clusters=2, max_iter=20).fit(Tensor(np.random.randn(20, 2).astype(np.float32)))
print("kmeans labels shape:", km.predict(Tensor(np.random.randn(5, 2).astype(np.float32))).shape)
```

**C entry point** (after building the DLL)
```c
#include "greyarea/greyarea.h"
int main(void) {
    int64_t shape[2] = {2, 2};
    GATensor* t = ga_tensor_ones(2, shape, GA_FLOAT32);
    ga_tensor_print(t);
    ga_tensor_release(t);
    return 0;
}
```

## Testing & Benchmarks
- Smoke: `python quick_test.py`, `python dtype_test.py`, or `python demo_greyml.py`.
- Full suites: copy `tests/` from the original repo, then `python -m pytest tests/python` or `ctest` in `build/release`.
- Benchmarks: `python benchmarks/bench_vs_numpy.py` or compile/run `benchmarks/bench_matmul.c`.

## Current Limits / Work in Progress
- Autograd coverage is partial on the C side; some gradients run only in Python.
- IO/model save/load and CSV loader are stubs; tensor save/load may be missing in some builds.
- CPU-only; SIMD exists but no GPU/offload yet. BLAS/OpenMP are optional and off by default.
- Several examples (C MNIST/XOR, CNN demo) are placeholders awaiting full kernels/datasets.

For a deeper dive into every module, data flow, and file role, see `main.md` in the repo root.
