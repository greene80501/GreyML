"""Support Vector Machines.
Bindings for SVM training/inference using GreyML math kernels.
"""

import ctypes
import numpy as np
from ..tensor import Tensor, _lib, no_grad

try:
    _ga_svc_create = _lib.ga_svc_create
    _ga_svc_create.restype = ctypes.c_void_p
    _ga_svc_create.argtypes = [ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int]
    _ga_svr_create = _lib.ga_svr_create
    _ga_svr_create.restype = ctypes.c_void_p
    _ga_svr_create.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int]
    _ga_svm_fit = _lib.ga_svm_fit
    _ga_svm_fit.restype = None
    _ga_svm_fit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _ga_svm_predict = _lib.ga_svm_predict
    _ga_svm_predict.restype = ctypes.c_void_p
    _ga_svm_predict.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _ga_svm_decision = _lib.ga_svm_decision_function
    _ga_svm_decision.restype = ctypes.c_void_p
    _ga_svm_decision.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
except AttributeError:  # pragma: no cover - optional symbols
    _ga_svc_create = _ga_svr_create = _ga_svm_fit = _ga_svm_predict = _ga_svm_decision = None


_KERNEL_MAP = {"linear": 0, "rbf": 1, "poly": 2, "sigmoid": 3}


class _BaseSVM:
    def __init__(self, ptr):
        self._ptr = ptr

    def fit(self, X: Tensor, y: Tensor):
        if self._ptr and _ga_svm_fit:
            with no_grad():
                _ga_svm_fit(self._ptr, X._c_ptr, y._c_ptr)
        else:
            self._fallback_fit(X, y)
        return self

    def decision_function(self, X: Tensor) -> Tensor:
        if self._ptr and _ga_svm_decision:
            ptr = _ga_svm_decision(self._ptr, X._c_ptr)
            return Tensor(_c_ptr=ptr, _shape=(X.shape[0],), _dtype=float)
        return self._fallback_decision(X)

    def predict(self, X: Tensor) -> Tensor:
        if self._ptr and _ga_svm_predict:
            ptr = _ga_svm_predict(self._ptr, X._c_ptr)
            dtype = float if isinstance(self, SVR) else int
            return Tensor(_c_ptr=ptr, _shape=(X.shape[0],), _dtype=dtype)
        return self._fallback_predict(X)

    # Fallbacks use very small numpy helpers for CI environments without DLLs.
    def _fallback_fit(self, X: Tensor, y: Tensor):
        Xn = X.numpy()
        yn = y.numpy()
        self._w = np.linalg.pinv(Xn) @ yn

    def _fallback_decision(self, X: Tensor) -> Tensor:
        scores = X.numpy() @ getattr(self, "_w", np.zeros(X.shape[1], dtype=np.float32))
        return Tensor(scores.astype(np.float32), dtype=np.float32)

    def _fallback_predict(self, X: Tensor) -> Tensor:
        scores = self._fallback_decision(X).numpy()
        if isinstance(self, SVR):
            return Tensor(scores.astype(np.float32), dtype=np.float32)
        return Tensor((scores >= 0).astype(np.int64), dtype=np.int64)


class SVC(_BaseSVM):
    def __init__(self, C: float = 1.0, kernel: str = "linear", gamma: float = 0.0, degree: int = 3, coef0: float = 0.0, tol: float = 1e-3, max_iter: int = 1000):
        ptr = _ga_svc_create(C, _KERNEL_MAP.get(kernel, 0), gamma, degree, coef0, tol, max_iter) if _ga_svc_create else None
        super().__init__(ptr)


class SVR(_BaseSVM):
    def __init__(self, C: float = 1.0, epsilon: float = 0.1, kernel: str = "rbf", gamma: float = 0.0, degree: int = 3, coef0: float = 0.0, tol: float = 1e-3, max_iter: int = 1000):
        ptr = _ga_svr_create(C, epsilon, _KERNEL_MAP.get(kernel, 1), gamma, degree, coef0, tol, max_iter) if _ga_svr_create else None
        super().__init__(ptr)
