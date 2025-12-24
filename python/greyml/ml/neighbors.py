"""Nearest neighbor search.
KNN/neighbor routines backed by GreyML tensors.
"""

import ctypes
import numpy as np
from ..tensor import Tensor, _lib, no_grad

try:
    _ga_knn_classifier_create = _lib.ga_knn_classifier_create
    _ga_knn_classifier_create.restype = ctypes.c_void_p
    _ga_knn_classifier_create.argtypes = [ctypes.c_int, ctypes.c_int]
    _ga_knn_regressor_create = _lib.ga_knn_regressor_create
    _ga_knn_regressor_create.restype = ctypes.c_void_p
    _ga_knn_regressor_create.argtypes = [ctypes.c_int, ctypes.c_int]
    _ga_knn_fit = _lib.ga_knn_fit
    _ga_knn_fit.restype = None
    _ga_knn_fit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _ga_knn_predict = _lib.ga_knn_predict
    _ga_knn_predict.restype = ctypes.c_void_p
    _ga_knn_predict.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _ga_knn_kneighbors = _lib.ga_knn_kneighbors
    _ga_knn_kneighbors.restype = None
    _ga_knn_kneighbors.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
except AttributeError:  # pragma: no cover
    _ga_knn_classifier_create = _ga_knn_regressor_create = _ga_knn_fit = _ga_knn_predict = _ga_knn_kneighbors = None


class _BaseKNN:
    def __init__(self, n_neighbors: int, weights: str, reg: bool):
        weight_enum = 1 if weights == "distance" else 0
        if reg and _ga_knn_regressor_create:
            self._ptr = _ga_knn_regressor_create(n_neighbors, weight_enum)
        elif _ga_knn_classifier_create:
            self._ptr = _ga_knn_classifier_create(n_neighbors, weight_enum)
        else:
            self._ptr = None
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._y = None
        self._X = None
        self._reg = reg

    def fit(self, X: Tensor, y: Tensor):
        # Always keep numpy copies for graceful fallbacks
        self._X = X.numpy()
        self._y = y.numpy()
        if self._ptr and _ga_knn_fit:
            with no_grad():
                _ga_knn_fit(self._ptr, X._c_ptr, y._c_ptr)
        return self

    def predict(self, X: Tensor) -> Tensor:
        if self._ptr and _ga_knn_predict:
            ptr = _ga_knn_predict(self._ptr, X._c_ptr)
            dtype = float if self._reg else int
            return Tensor(_c_ptr=ptr, _shape=(X.shape[0],), _dtype=dtype)
        return self._fallback_predict(X)

    def kneighbors(self, X: Tensor, k: int | None = None):
        k = k or self.n_neighbors
        if self._ptr and _ga_knn_kneighbors:
            dist_ptr = ctypes.c_void_p()
            idx_ptr = ctypes.c_void_p()
            _ga_knn_kneighbors(self._ptr, X._c_ptr, k, ctypes.byref(dist_ptr), ctypes.byref(idx_ptr))
            dists = Tensor(_c_ptr=dist_ptr.value, _shape=(X.shape[0], k), _dtype=float)
            inds = Tensor(_c_ptr=idx_ptr.value, _shape=(X.shape[0], k), _dtype=int)
            return dists, inds
        return self._fallback_kneighbors(X, k)

    def _fallback_predict(self, X: Tensor) -> Tensor:
        dists_t, inds_t = self._fallback_kneighbors(X, self.n_neighbors)
        dists = dists_t.numpy()
        inds = inds_t.numpy()
        if self._reg:
            vals = self._y[inds]
            weights = 1.0 / (dists + 1e-8) if self.weights == "distance" else np.ones_like(dists)
            pred = (vals * weights).sum(axis=1) / weights.sum(axis=1)
            return Tensor(pred.astype(np.float32), dtype=np.float32)
        vals = self._y[inds]
        weights = 1.0 / (dists + 1e-8) if self.weights == "distance" else np.ones_like(dists)
        preds = []
        for i in range(vals.shape[0]):
            vote = {}
            for t in range(vals.shape[1]):
                cls = int(vals[i, t])
                vote[cls] = vote.get(cls, 0.0) + weights[i, t]
            best = max(vote.items(), key=lambda kv: kv[1])[0]
            preds.append(best)
        return Tensor(np.array(preds, dtype=np.int64), dtype=np.int64)

    def _fallback_kneighbors(self, X: Tensor, k: int):
        if self._X is None:
            raise RuntimeError("KNN fallback requires prior fit with numpy data.")
        x = X.numpy()
        dists = np.linalg.norm(self._X[None, :, :] - x[:, None, :], axis=2)
        idx = np.argsort(dists, axis=1)[:, :k]
        dist_sorted = np.take_along_axis(dists, idx, axis=1)
        return Tensor(dist_sorted.astype(np.float32), dtype=np.float32), Tensor(idx.astype(np.int64), dtype=np.int64)


class KNeighborsClassifier(_BaseKNN):
    def __init__(self, n_neighbors: int = 5, weights: str = "uniform"):
        super().__init__(n_neighbors, weights, reg=False)


class KNeighborsRegressor(_BaseKNN):
    def __init__(self, n_neighbors: int = 5, weights: str = "uniform"):
        super().__init__(n_neighbors, weights, reg=True)
