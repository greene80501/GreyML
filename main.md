# GreyML Deep Dive

This document explains how the GreyML stack is put together, what each directory and module does, and how the pieces work end-to-end.

## Purpose & Scope
- Deliver a Windows-first ML runtime with a C17 core (`greyarea.dll`) and a lightweight Python API (`greyml`).
- Keep dependencies minimal (NumPy only on Python), remain CPU-only for now, and stay readable so new contributors can follow the code path from Python calls down into C kernels.

**Snapshot note**: This GreyML_V0.2 copy is trimmed to essentials (src/include/python/examples/docs/benchmarks/scripts plus smoke demos). Full C/Python test suites, CI config, and prior build/dist artifacts are not included; pull them from the original repo if needed.

## Architecture Overview
### C Backend (DLL)
- **Build config**: `CMakeLists.txt` sets C17, forces Release flags on MSVC, and defines options (`GA_USE_OPENMP`, `GA_USE_BLAS`, `GA_BUILD_TESTS`, `GA_BUILD_EXAMPLES`, `GA_BUILD_PYTHON`). Presets in `CMakePresets.json` target MSVC + Ninja (`build/release` and `build/debug`). `scripts/build.bat` boots the VS toolchain, configures CMake, builds, then copies the DLL into `python/greyml/`.
- **Core (`src/core/`)**:  
  - `ga_common.c` for error/status reporting; `ga_mem.c` for aligned alloc, arenas, and pools; `ga_random.c` for PCG RNG and tensor random fills.  
  - `ga_tensor.c` owns tensor allocation, shape/stride computation, contiguous/views, reshape/unsqueeze/squeeze/transposes, rand/randn, retain/release, and zero-copy wrapping via `ga_tensor_from_data`. `ga_tensor_print.c` and `ga_tensor_view.c` provide display stubs and view helpers.
- **Ops (`src/ops/`)**: Elementwise binary/unary math, reductions, matmul (`ga_ops_matmul.c` with optional BLAS), transforms, activations, softmax/log-softmax, and convolution/pooling stubs. Ops allocate output tensors, run naive CPU loops, and attach autograd nodes when grad is enabled.
- **Autograd (`src/autograd/`)**:  
  - `ga_autograd.c` holds the grad-enabled flag, node creation, grad accumulation, and the backward driver.  
  - `ga_graph.c` builds topological order for backward traversal.  
  - `ga_autograd_ops.c` implements backward kernels for the covered ops (add/sub/mul/div/matmul/relu/sigmoid/log/softmax/sum/mean/pool/conv/etc.), plus unbroadcast helpers. Coverage is partial; unsupported ops will not backprop.
- **NN (`src/nn/`)**: Modules for Linear, Conv2D, pooling, BatchNorm, Dropout, Embedding, RNN/LSTM/GRU, Multi-head Attention, and Sequential containers. Layers own parameters (created as tensors with grad enabled) and call into ops (e.g., Linear uses transpose + matmul). Initializers live in `ga_nn_init.c`. Forward wrappers in each file wire modules into the generic `GAModule` layout defined in `include/greyarea/ga_nn.h`.
- **Optim & Loss (`src/optim/`, `src/loss/`)**: SGD and Adam implement parameter updates and zero-grad; StepLR scheduler is available. Losses cover MSE, L1, cross-entropy/NLL, BCE, and Huber.
- **Classical ML (`src/ml/`)**: Decision trees/forests, SVM/SVR (with kernel helpers), KMeans/DBSCAN clustering, and KNN search. APIs expose create/fit/predict primitives that the Python wrappers call.
- **IO & Misc**: `src/io/ga_io_tensor.c` (tensor save/load), `ga_io_model.c` (model save/load stub), `ga_io_csv.c` (CSV loader); SIMD feature detection and AVX kernels live in `src/simd/`.
- **Public headers (`include/greyarea/`)**: `greyarea.h` aggregates all GA_* headers for consumers; headers define ABI-visible structs (e.g., `GATensor`, `GAModule`, `GASVM`, `GATree`) and function contracts. `ga_simd.h` exposes CPU feature probes.
- **Build artifacts**: `build/release/` stores the compiled DLL, import libs, and Ninja/CMake metadata; treat as generated output.

### Python Layer (`python/greyml/`)
- **Library loading**: `_lib_loader.py` searches `build/Release`, `build/release`, packaged `python/greyml/`, and PATH entries for `greyarea.dll` and sets up a minimal signature set. `_bindings.py` contains an exhaustive ctypes signature map covering tensors, ops, autograd, NN/optim/ML, RNG, and init routines for future expansion.
- **Tensor & autograd (`tensor.py`, `autograd.py`)**:  
  - Loads the DLL (searching packaged + Python installation paths), maps core functions, and tracks grad state via `_set_grad_enabled` / `_is_grad_enabled`.  
  - `Tensor` wraps a `GATensor*`, handles creation from Python data, exposes metadata (`shape`, `dtype`, `ndim`), numpy interop (`numpy()`, `item()`), in-place fills/clone/contiguous, and serialization (`save`/`load` when C symbols exist).  
  - Math ops (`+`, `-`, `*`, `/`, matmul, reshape/unsqueeze/squeeze/transpose, sum/mean/relu/sigmoid/softmax/exp/log/sqrt/abs`) call C kernels when present, otherwise fall back to NumPy; each op wires Python-side autograd closures when C grad is missing.  
  - Loss helpers (MSE, L1, BCE, Huber, cross-entropy), random creators (`zeros`, `ones`, `randn`), CSV loader, and grad context managers (`no_grad`, `enable_grad`).  
  - Autograd is hybrid: prefers C `ga_backward` when available; otherwise uses recorded `_backward` closures.
- **Ops facade (`ops.py`)**: Functional wrappers that delegate to `Tensor` methods for relu/sigmoid/softmax/add/mul/matmul/sum`.
- **NN (`nn/`)**:  
  - `module.py` defines a minimal `Module` base with parameter tracking, train/eval switches, and zero_grad.  
  - `layers.py` implements Linear, Conv2d, pooling, BatchNorm, Dropout, Embedding, SimpleRNN, LSTM, GRU, and MultiheadAttention using NumPy math when C bindings are absent.  
  - Convenience modules: activations (`activation.py`), attention wrapper, containers (`Sequential`, `ModuleList`), init helpers (`init.py`), normalization (`BatchNorm2d`, `LayerNorm`), pooling, and RNN exports.  
  - Some ops fall back to identity/no-op if DLL symbols are missing (e.g., Conv2d/Pooling) to keep demos runnable.
- **Optim (`optim/`)**: Python SGD and Adam mirror their C counterparts, operating on `Tensor` params with NumPy updates under a `no_grad` guard. StepLR scheduler multiplies LR every N steps.
- **Classical ML (`ml/`)**: Wrappers for tree/forest/SVM/KMeans/DBSCAN/KNN that bind to C functions when present and provide small NumPy fallbacks (e.g., `_PyFallbackTree`, KNN distance voting) to keep tests running without the DLL.
- **Data & Utils**: `data/dataset.py` and `data/dataloader.py` implement a simple iterable DataLoader with batching/shuffling and tensor stacking; `utils/io.py` provides tensor/model save/load via C APIs or NumPy `.npz` fallback, and `utils/metrics.py` offers accuracy/MSE/MAE helpers.
- **Package glue**: `__init__.py` exports the public API; version is `0.1.0`. `ops.py` and `autograd.py` re-export functional helpers.

### Packaging & Tooling
- **Python packaging**: Root `setup.py` and `pyproject.toml` define the `greyml` package (version `0.1.0`, NumPy dependency). `MANIFEST.in` ensures the DLL ships with the package. `dist/` will be produced when you build/publish; it is not tracked in this trimmed copy. `python/setup.py`/`python/pyproject.toml` mirror minimal metadata inside the package directory.
- **Scripts**: `scripts/build.bat` (build + copy DLL), `scripts/run_tests.bat` (warns because tests are not copied here), `scripts/download_mnist.py` (placeholder).
- **CI**: Not included in this trimmed snapshot; pull `.github/workflows/ci.yml` from the original repo if you need it.

## Examples, Demos, Benchmarks
- **Python demos (`examples/python/`)**:  
  - `00_quickstart.py` tensor/Linear/KMeans walkthrough.  
  - `01_tensor_basics.py` basic ops; `02_autograd_intro.py` square + backward; `03_linear_regression.py` toy regression.  
  - `04_mnist_mlp.py` minimal MLP loop with dummy data; `05_mnist_cnn.py` placeholder.  
  - `06_decision_tree.py`, `07_random_forest.py`, `08_svm_classification.py`, `09_kmeans_clustering.py` cover classical ML wrappers.
- **C examples (`examples/c/`)**: XOR and MNIST MLP placeholders that currently just print guidance.
- **Feature demo (`demo_greyml.py`)**: Runs tensor math, LSTM/GRU, attention, feedforward with dropout, seq2seq, and a toy sentiment classifier; emphasizes shapes and available layers.
- **Benchmarks (`benchmarks/`)**: Matmul benchmark in C, conv placeholder, and a Python NumPy vs GreyML matmul comparison.

## Tests
- Full C and Python suites live in `tests/` in the original repo; this trimmed copy omits them. Copy that directory if you need pytest/ctest coverage.
- Quick scripts: `quick_test.py` (tensor ops), `dtype_test.py` (dtype handling), and `demo_greyml.py` serve as manual sanity checks.

## Docs & Generated Artifacts
- **Docs**: `docs/windows_setup.md` outlines prerequisites and build steps; other docs (`architecture.md`, `tutorial.md`, `api_reference.md`, `contributing.md`) are placeholders awaiting real content.
- **Generated**: `build/` (CMake/Ninja + DLL outputs), `dist/` (wheels and cached deps), `__pycache__/`, `.pytest_cache/`. Avoid editing these directly.

## Known Gaps & Follow-ups
- C-side autograd coverage is incomplete; some ops rely on Python autograd closures or have no gradients.
- IO/model serialization and CSV loading are partial stubs; expand `ga_io_*` and Python `utils/io.py` once formats are finalized.
- Several NN/ML kernels fall back to NumPy or identity behavior when DLL symbols are missing (Conv2d, pooling, some ML models); fill in bindings and C implementations for correctness and speed.
- Performance work (SIMD/BLAS, better blocking for matmul/conv) remains to be done; current kernels are naive.
- Placeholders in examples (`examples/c/*`, `examples/python/05_mnist_cnn.py`, `scripts/download_mnist.py`) should be fleshed out with real data paths and training loops.
