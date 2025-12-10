<!-- GreyML doc: README. Added context for collaborators. -->

﻿GreyML: Windows-First AI Stack
==============================

Made and developed by GreyArea Labs — https://greyarea.icu

GreyML is a compact AI library with a C17 core (`greyarea.dll`) and a Python wrapper (`greyml`). It is built to be small, transparent, and Windows-focused while still covering modern deep learning layers and classic ML algorithms. This README consolidates the technical plan and the practical getting-started guide.

What GreyML is (and why it exists)
----------------------------------
- Own your stack: hand-rolled C tensor/ops engine you can read and extend.
- Python-first ergonomics: NumPy interoperability and a clean API exposed from `python/greyml`.
- Modern neural pieces without heavy baggage: Linear, Conv2D, BatchNorm, Dropout, Embedding, RNN/LSTM/GRU, Multi-head Attention; optimizers (SGD, Adam) and schedulers.
- Classical ML included: decision trees, random forests, SVMs, KNN, KMeans, DBSCAN.
- Dependency-light and offline-friendly: NumPy is the only Python requirement; everything else is built in C.
- Windows-first: targets Windows 11 (x64), MSVC/Clang-CL, CMake; CPU-only today with SIMD helpers.

Architecture (high level)
-------------------------
- Core engine (C17): sources in `src/` build into `greyarea.dll`. Handles tensors, math ops, parts of autograd, and CPU execution with SIMD support.
- Python bindings: `python/greyml/_bindings.py` and `_lib_loader.py` load the DLL via `ctypes`, expose the `Tensor` class (`python/greyml/tensor.py`), and surface neural, optim, and ML helpers.
- Autograd: early-stage. Some gradients run in C; others fall back to Python-side tracking. Expect gaps until full C autograd coverage is finished.
- Scope today: CPU-only; some ops may temporarily fall back to NumPy when a matching C symbol is missing.

Install and run (quick start)
-----------------------------
1) Requirements: Windows 11, Python 3.10+, NumPy, MSVC build tools (for DLL rebuild).
2) From `kimi_k2/`, install locally:
   ```bash
   pip install -e .
   ```
3) Ensure the DLL exists at `python/greyml/greyarea.dll`. If not, build it:
   ```batch
   scripts\build.bat
   ```
4) Smoke test:
   ```bash
   python quick_test.py
   ```
5) Feature tour (tensors, LSTM/GRU, attention, simple MLP):
   ```bash
   python demo_greyml.py
   ```

Usage snapshots
---------------
Create and use tensors:
```python
from greyml import Tensor, zeros
import numpy as np

x = Tensor([1, 2, 3], dtype=np.float32)
y = x + x
z = zeros(2, 2)
print(y.numpy(), z.shape)
```

Simple feedforward example (see `demo_greyml.py` for more):
```python
from greyml.nn.layers import Linear
import numpy as np
from greyml import Tensor

fc1 = Linear(784, 128)
inp = Tensor(np.random.randn(16, 784).astype(np.float32))
out = fc1.forward(inp)
print(out.shape)
```

File tree (orientation)
-----------------------
```
kimi_k2/
|-- python/
|   `-- greyml/
|       |-- __init__.py
|       |-- tensor.py            # Tensor class, autograd helpers, ops wrappers
|       |-- _bindings.py         # ctypes signatures for greyarea.dll
|       |-- _lib_loader.py       # DLL loading logic
|       |-- autograd.py
|       |-- nn/                  # layers, activations, containers, init, rnn, attention, pooling, dropout
|       |-- optim/               # SGD, Adam, optimizer base, schedulers
|       |-- ml/                  # tree, forest, svm, cluster, neighbors
|       |-- data/                # dataset + dataloader helpers
|       `-- utils/               # io, metrics
|-- src/
|   |-- core/                    # ga_common, ga_mem, ga_tensor, views/print, random
|   |-- ops/                     # binary, unary, reduce, matmul, conv, transform, activation, optional BLAS
|   |-- autograd/                # tape, graph, backward kernels (in progress)
|   |-- nn/                      # linear, conv, pooling, norm, dropout, embed, rnn, attention, containers, init
|   |-- optim/                   # sgd, adam, scheduler
|   |-- loss/                    # mse, ce, other losses
|   |-- ml/                      # tree, forest, svm, cluster (kmeans, dbscan), neighbors
|   |-- io/                      # tensor/model save/load stubs, csv loader
|   `-- simd/                    # CPU feature detect, AVX implementations
|-- examples/
|   |-- c/                       # xor.c, mnist_mlp.c
|   `-- python/                  # tensor/autograd/nn/ml demos
|-- tests/
|   |-- c/                       # unit tests for core/ops/nn/ml
|   `-- python/                  # pytest suites for tensor/autograd/nn/optim/ml/io
|-- docs/                        # architecture, api_reference, tutorial, windows_setup, contributing
|-- scripts/                     # build.bat, run_tests.bat, dataset helpers
|-- CMakeLists.txt               # C build configuration
|-- CMakePresets.json            # Windows presets
|-- pyproject.toml, setup.py     # Python packaging
|-- quick_test.py                # Python smoke test
|-- demo_greyml.py               # Feature showcase (LSTM/GRU/attention/MLP)
|-- dtype_test.py                # dtype checks
|-- LICENSE                      # MIT license
`-- README.md                    # This document
```

Module breakdown (summary)
--------------------------
- Core: `ga_common`, `ga_mem`, `ga_tensor` (+ views/print), `ga_random` - memory, tensor creation, layout, RNG.
- Ops: binary/unary math, reductions, matmul (naive + AVX2; optional BLAS), conv, transforms, activations.
- Autograd: tape/graph + backward kernels (in progress; partial coverage).
- NN: Linear, Conv, Pooling, Normalization, Dropout, Embedding, RNN/LSTM/GRU, Attention, containers, init.
- Optim: SGD, Adam, schedulers.
- Loss: MSE, Cross-entropy, L1, Huber, BCE.
- Classical ML: decision tree/forest, SVM (classification/regression), KMeans, DBSCAN, KNN.
- I/O & Utils: tensor/model save/load stubs; CSV loader WIP.
- Python bindings: `Tensor`, autograd helpers, nn/optim/ml wrappers, metrics, datasets/dataloaders.

Build and install
-----------------
- Toolchain: MSVC 2022 or Clang-CL, Windows SDK, CMake 3.20+, Python 3.10+. Optional: Ninja, OpenBLAS.
- Build script: `scripts\build.bat` sets up MSVC env, configures CMake (presets in `CMakePresets.json`), builds `greyarea.dll`, and copies it into `python/greyml/`.
- Editable install: run `pip install -e .` from `kimi_k2/` to expose the Python package.

Testing and demos
-----------------
- Python smoke: `python quick_test.py` (basic tensor ops).
- Feature demo: `python demo_greyml.py` (tensors, LSTM/GRU, attention, simple MLP, classic ML helpers).
- Full suite: `scripts\run_tests.bat` (C and Python; some skips if symbols are missing).

Current limitations (work in flight)
------------------------------------
- C-side autograd not complete; some gradients are Python-only.
- Model save/load and CSV loaders are mid-flight.
- CPU-only; no GPU/offload yet.
- Some NN/ML kernels still rely on Python fallbacks when C implementations are missing.

Roadmap highlights
------------------
- Finish C autograd coverage and backprop paths for all major ops.
- Harden C-backed NN/ML kernels; reduce Python fallbacks.
- Complete model save/load and CSV I/O.
- Performance tuning (SIMD/BLAS) and benchmarking.
- Explore GPU/offload after CPU parity is stable.

Project paper
-------------
Placeholder: The GreyML paper (GreyArea Labs) will be published via https://greyarea.icu — link will be added here when live.

Credit
------
GreyML is made and developed by GreyArea Labs — https://greyarea.icu

License
-------
MIT (see `LICENSE`).

Complete file list (with roles)
-------------------------------
- `.clang-format` — formatting rules for C sources.
- `CMakeLists.txt` — top-level CMake build config for the core library.
- `CMakePresets.json` — Windows build presets (debug/release).
- `demo_greyml.py` — Python feature showcase (LSTM/GRU/attention/MLP/ops).
- `dtype_test.py` — dtype sanity checks.
- `LICENSE` — MIT license.
- `pyproject.toml` — project metadata and dependencies for Python build.
- `quick_test.py` — Python smoke test for tensors/ops.
- `README.md` — this document.
- `setup.py` — setuptools entry for editable/installable Python package.

.github/
- `.github/workflows/ci.yml` — CI workflow for lint/build/test on Windows/Python.

.pytest_cache/ (generated pytest cache)
- `.pytest_cache/.gitignore` — ignore cache files.
- `.pytest_cache/CACHEDIR.TAG` — cache marker.
- `.pytest_cache/README.md` — pytest cache info.
- `.pytest_cache/v/cache/lastfailed` — last failed tests cache.
- `.pytest_cache/v/cache/nodeids` — cached test node IDs.

benchmarks/
- `benchmarks/bench_conv.c` — C benchmark for convolution.
- `benchmarks/bench_matmul.c` — C benchmark for matrix multiplication.
- `benchmarks/bench_vs_numpy.py` — Python benchmark comparing against NumPy.

build/release/ (generated build artifacts)
- `build/release/build.ninja`, `.ninja_deps`, `.ninja_log` — Ninja build files.
- `build/release/CMakeCache.txt`, `cmake_install.cmake` — CMake cache/install scripts.
- `build/release/greyarea.dll` — built DLL for the core library.
- `build/release/greyarea.exp`, `greyarea.lib` — import/lib files for the DLL.
- `build/release/CMakeFiles/...` — CMake internal metadata, object files for every C source, compiler ID probes, and include tracing (all generated).
- `build/release/CMakeFiles/ShowIncludes/*` — sample include output from MSVC (generated).

docs/
- `docs/api_reference.md` — API reference notes.
- `docs/architecture.md` — architecture overview.
- `docs/contributing.md` — contribution guidelines.
- `docs/tutorial.md` — step-by-step tutorial.
- `docs/windows_setup.md` — Windows environment setup.

examples/c/
- `examples/c/xor.c` — XOR training example (C).
- `examples/c/mnist_mlp.c` — MNIST MLP example (C).

examples/python/
- `examples/python/00_quickstart.py` — minimal quickstart.
- `examples/python/01_tensor_basics.py` — tensor basics.
- `examples/python/02_autograd_intro.py` — autograd intro.
- `examples/python/03_linear_regression.py` — linear regression example.
- `examples/python/04_mnist_mlp.py` — MNIST MLP example (Python).
- `examples/python/05_mnist_cnn.py` — MNIST CNN example.
- `examples/python/06_decision_tree.py` — decision tree demo.
- `examples/python/07_random_forest.py` — random forest demo.
- `examples/python/08_svm_classification.py` — SVM classification demo.
- `examples/python/09_kmeans_clustering.py` — k-means clustering demo.

include/greyarea/ (public headers)
- `ga_autograd.h` — autograd API.
- `ga_cluster.h` — clustering API (k-means/DBSCAN).
- `ga_common.h` — shared types/macros/errors.
- `ga_io.h` — tensor/model IO API.
- `ga_loss.h` — loss functions API.
- `ga_mem.h` — memory allocator API.
- `ga_neighbors.h` — KNN API.
- `ga_nn.h` — neural network API.
- `ga_ops.h` — operations API.
- `ga_optim.h` — optimizer API.
- `ga_random.h` — RNG API.
- `ga_simd.h` — SIMD detection helpers.
- `ga_svm.h` — SVM API.
- `ga_tensor.h` — tensor API.
- `ga_tree.h` — decision tree/forest API.
- `greyarea.h` — master include aggregating all public headers.

python/ (Python packaging metadata)
- `python/pyproject.toml` — Python build backend metadata (package-local).
- `python/setup.py` — setuptools helper for the Python-only layout.

python/greyml/ (Python package)
- `__init__.py` — package init/version exports.
- `tensor.py` — Tensor class, ctypes bindings, Python-side autograd helpers, ops wrappers.
- `_bindings.py` — ctypes signatures for `greyarea.dll`.
- `_lib_loader.py` — DLL loading logic.
- `autograd.py` — Python autograd helpers/context managers.
- `greyarea.dll` — shipped DLL copy for Python package.

python/greyml/data/
- `dataset.py` — dataset base class.
- `dataloader.py` — DataLoader with batching/shuffling.
- `__init__.py` — data package init.

python/greyml/ml/
- `cluster.py` — KMeans/DBSCAN Python wrappers.
- `forest.py` — random forest wrappers.
- `neighbors.py` — KNN wrappers.
- `svm.py` — SVM/SVR wrappers.
- `tree.py` — decision tree wrappers.
- `__init__.py` — ML package init.
- `__pycache__/*.pyc` — generated Python bytecode caches.

python/greyml/nn/
- `activation.py` — activation helpers.
- `attention.py` — multi-head attention layer.
- `container.py` — Sequential/ModuleList containers.
- `dropout.py` — dropout layer.
- `functional.py` — functional NN API.
- `init.py` — weight initialization helpers.
- `layers.py` — core layers (Linear, Conv2D, etc.).
- `module.py` — base Module class.
- `normalization.py` — BatchNorm/LayerNorm layers.
- `pooling.py` — pooling layers.
- `rnn.py` — RNN/LSTM/GRU layers.
- `__init__.py` — nn package init.
- `__pycache__/*.pyc` — generated Python bytecode caches.

python/greyml/optim/
- `adam.py` — Adam/AdamW optimizer.
- `optimizer.py` — optimizer base class.
- `scheduler.py` — learning rate schedulers.
- `sgd.py` — SGD optimizer.
- `__init__.py` — optim package init.
- `__pycache__/*.pyc` — generated Python bytecode caches.

python/greyml/utils/
- `io.py` — save/load helpers.
- `metrics.py` — metrics utilities.
- `__init__.py` — utils package init.
- `__pycache__/*.pyc` — generated Python bytecode caches.

python/greyml/__pycache__/ (generated) — bytecode caches for package modules.

scripts/
- `build.bat` — build the C core and copy DLL into the Python package.
- `download_mnist.py` — helper to fetch MNIST dataset.
- `run_tests.bat` — run C and Python test suites.

src/autograd/
- `ga_autograd.c` — autograd tape/engine core (WIP).
- `ga_autograd_ops.c` — backward implementations (WIP).
- `ga_graph.c` — computation graph utilities.

src/core/
- `ga_common.c` — shared utilities/error handling.
- `ga_mem.c` — memory allocator/arena/pools.
- `ga_random.c` — RNG implementation.
- `ga_tensor.c` — tensor creation/manipulation core.
- `ga_tensor_print.c` — tensor pretty-printing.
- `ga_tensor_view.c` — views/slicing/reshape helpers.

src/io/
- `ga_io_csv.c` — CSV loading (WIP).
- `ga_io_model.c` — model save/load (WIP).
- `ga_io_tensor.c` — tensor save/load.

src/loss/
- `ga_loss_ce.c` — cross-entropy losses.
- `ga_loss_mse.c` — mean squared error.
- `ga_loss_other.c` — L1/Huber and others.

src/ml/
- `ga_cluster_dbscan.c` — DBSCAN clustering.
- `ga_cluster_kmeans.c` — KMeans clustering.
- `ga_forest.c` — random forest training/inference.
- `ga_neighbors.c` — KNN algorithms.
- `ga_svm.c` — SVM/SVR training/inference.
- `ga_svm_kernel.c` — SVM kernel functions.
- `ga_tree.c` — decision tree core.
- `ga_tree_split.c` — tree split finding.

src/nn/
- `ga_nn_attention.c` — attention layers.
- `ga_nn_container.c` — container utilities.
- `ga_nn_conv.c` — convolution layers.
- `ga_nn_dropout.c` — dropout layer.
- `ga_nn_embed.c` — embedding layer.
- `ga_nn_init.c` — weight initialization.
- `ga_nn_linear.c` — linear layers.
- `ga_nn_norm.c` — normalization layers.
- `ga_nn_pool.c` — pooling layers.
- `ga_nn_rnn.c` — RNN/LSTM/GRU layers.

src/ops/
- `ga_ops_activation.c` — activation ops.
- `ga_ops_binary.c` — binary elementwise ops.
- `ga_ops_blas.c` — optional BLAS-backed ops.
- `ga_ops_conv.c` — convolution ops.
- `ga_ops_matmul.c` — matrix multiplication.
- `ga_ops_reduce.c` — reduction ops.
- `ga_ops_transform.c` — reshape/transpose/concat/split ops.
- `ga_ops_unary.c` — unary elementwise ops.

src/optim/
- `ga_optim_adam.c` — Adam optimizer.
- `ga_optim_scheduler.c` — learning rate schedulers.
- `ga_optim_sgd.c` — SGD optimizer.

src/simd/
- `ga_simd_avx.c` — AVX/AVX2 SIMD implementations.
- `ga_simd_detect.c` — CPU feature detection.

tests/c/
- `test_autograd.c` — C autograd tests.
- `test_cluster.c` — clustering tests.
- `test_io.c` — IO tests.
- `test_main.c` — C test runner.
- `test_mem.c` — memory allocator tests.
- `test_neighbors.c` — KNN tests.
- `test_nn.c` — neural network tests.
- `test_ops.c` — ops tests.
- `test_svm.c` — SVM tests.
- `test_tensor.c` — tensor tests.
- `test_tree.c` — decision tree tests.

tests/python/
- `test_autograd.py` — Python autograd tests.
- `test_bindings.py` — ctypes binding tests.
- `test_integration.py` — end-to-end integration tests.
- `test_io.py` — IO tests.
- `test_io_rng.py` — RNG/IO tests.
- `test_ml.py` — ML wrappers tests.
- `test_ml_algorithms.py` — ML algorithm behavior tests.
- `test_nn.py` — NN wrapper tests.
- `test_nn_layers.py` — NN layer-specific tests.
- `test_ops_extended.py` — extended ops tests.
- `test_optim.py` — optimizer tests.
- `test_plan.py` — planning/test scaffolding.
- `test_tensor.py` — tensor tests.
- `__pycache__/*.pyc` — generated Python bytecode caches.

__pycache__/ (root)
- `__pycache__/basic_smoke_test.cpython-312-pytest-9.0.2.pyc` — cached test.
- `__pycache__/dtype_test.cpython-312-pytest-9.0.2.pyc` — cached dtype test.
- `__pycache__/quick_test.cpython-312-pytest-9.0.2.pyc` — cached smoke test.
