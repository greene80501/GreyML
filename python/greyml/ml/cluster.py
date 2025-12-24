"""Clustering algorithms.
Python drivers for DBSCAN/KMeans built atop the C backend.
"""

import ctypes
from ..tensor import Tensor, _lib, no_grad

# Bindings
try:
    _ga_kmeans_create = _lib.ga_kmeans_create
    _ga_kmeans_create.restype = ctypes.c_void_p
    _ga_kmeans_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
    _ga_kmeans_fit = _lib.ga_kmeans_fit
    _ga_kmeans_fit.restype = None
    _ga_kmeans_fit.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _ga_kmeans_predict = _lib.ga_kmeans_predict
    _ga_kmeans_predict.restype = ctypes.c_void_p
    _ga_kmeans_predict.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _ga_kmeans_free = _lib.ga_kmeans_free
    _ga_kmeans_free.restype = None
    _ga_kmeans_free.argtypes = [ctypes.c_void_p]
except AttributeError:  # pragma: no cover
    _ga_kmeans_create = _ga_kmeans_fit = _ga_kmeans_predict = _ga_kmeans_free = None

try:
    _ga_dbscan_create = _lib.ga_dbscan_create
    _ga_dbscan_create.restype = ctypes.c_void_p
    _ga_dbscan_create.argtypes = [ctypes.c_float, ctypes.c_int]
    _ga_dbscan_fit_predict = _lib.ga_dbscan_fit_predict
    _ga_dbscan_fit_predict.restype = ctypes.c_void_p
    _ga_dbscan_fit_predict.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _ga_dbscan_free = _lib.ga_dbscan_free
    _ga_dbscan_free.restype = None
    _ga_dbscan_free.argtypes = [ctypes.c_void_p]
except AttributeError:  # pragma: no cover
    _ga_dbscan_create = _ga_dbscan_fit_predict = _ga_dbscan_free = None


class _CKMeans(ctypes.Structure):
    _fields_ = [
        ("n_clusters", ctypes.c_int),
        ("max_iter", ctypes.c_int),
        ("tol", ctypes.c_float),
        ("n_init", ctypes.c_int),
        ("centroids", ctypes.c_void_p),
        ("inertia_", ctypes.c_float),
        ("n_iter_", ctypes.c_int),
    ]


class KMeans:
    def __init__(self, n_clusters: int = 8, max_iter: int = 100, tol: float = 1e-4, n_init: int = 1):
        self.n_clusters = n_clusters
        self._ptr = _ga_kmeans_create(n_clusters, max_iter, float(tol), n_init) if _ga_kmeans_create else None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None

    def __del__(self):
        if getattr(self, "_ptr", None) and _ga_kmeans_free:
            try:
                _ga_kmeans_free(self._ptr)
            except Exception:
                pass

    def fit(self, X: Tensor):
        if self._ptr and _ga_kmeans_fit:
            with no_grad():
                _ga_kmeans_fit(self._ptr, X._c_ptr)
            # pull inertia and iterations from struct
            ptr_val = int(self._ptr) if self._ptr else 0
            view = _CKMeans.from_address(ptr_val)
            self.inertia_ = float(view.inertia_)
            self.n_iter_ = int(view.n_iter_)
        return self

    def predict(self, X: Tensor) -> Tensor:
        if self._ptr and _ga_kmeans_predict:
            ptr = _ga_kmeans_predict(self._ptr, X._c_ptr)
            return Tensor(_c_ptr=ptr, _shape=(X.shape[0],), _dtype=int)
        return Tensor([0] * X.shape[0], dtype=int)

    def fit_predict(self, X: Tensor) -> Tensor:
        self.fit(X)
        return self.predict(X)


class DBSCAN:
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self._ptr = _ga_dbscan_create(float(eps), min_samples) if _ga_dbscan_create else None

    def __del__(self):
        if getattr(self, "_ptr", None) and _ga_dbscan_free:
            try:
                _ga_dbscan_free(self._ptr)
            except Exception:
                pass

    def fit_predict(self, X: Tensor) -> Tensor:
        if self._ptr and _ga_dbscan_fit_predict:
            with no_grad():
                ptr = _ga_dbscan_fit_predict(self._ptr, X._c_ptr)
            return Tensor(_c_ptr=ptr, _shape=(X.shape[0],), _dtype=int)
        return Tensor([-1] * X.shape[0], dtype=int)
