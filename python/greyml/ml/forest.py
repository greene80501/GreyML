"""Random forest implementation.
Exposes tree ensemble training/inference using GreyML primitives.
"""

import ctypes
from ..tensor import Tensor, _lib, no_grad

try:
    _ga_forest_create = _lib.ga_forest_create
    _ga_forest_create.restype = ctypes.c_void_p
    _ga_forest_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _ga_forest_regressor_create = _lib.ga_forest_regressor_create
    _ga_forest_regressor_create.restype = ctypes.c_void_p
    _ga_forest_regressor_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _ga_forest_fit = _lib.ga_forest_fit
    _ga_forest_fit.restype = None
    _ga_forest_fit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _ga_forest_predict = _lib.ga_forest_predict
    _ga_forest_predict.restype = ctypes.c_void_p
    _ga_forest_predict.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
except AttributeError:  # pragma: no cover
    _ga_forest_create = None
    _ga_forest_regressor_create = None
    _ga_forest_fit = None
    _ga_forest_predict = None


class RandomForestClassifier:
    def __init__(self, n_estimators: int = 10, max_depth: int | None = None, n_classes: int = 2, max_features: int | None = None):
        depth = max_depth if max_depth is not None else 10
        mf = max_features if max_features is not None else 0
        self._ptr = _ga_forest_create(n_estimators, depth, n_classes) if _ga_forest_create else None
        self.n_classes = n_classes
        self.max_features = mf

    def fit(self, X: Tensor, y: Tensor):
        if self._ptr and _ga_forest_fit:
            with no_grad():
                _ga_forest_fit(self._ptr, X._c_ptr, y._c_ptr)
        return self

    def predict(self, X: Tensor) -> Tensor:
        if self._ptr and _ga_forest_predict:
            ptr = _ga_forest_predict(self._ptr, X._c_ptr)
            return Tensor(_c_ptr=ptr, _shape=(X.shape[0],), _dtype=int)
        return Tensor([0] * X.shape[0], dtype=int)


class RandomForestRegressor:
    def __init__(self, n_estimators: int = 10, max_depth: int | None = None, max_features: int | None = None):
        depth = max_depth if max_depth is not None else 10
        mf = max_features if max_features is not None else 0
        self._ptr = _ga_forest_regressor_create(n_estimators, depth, mf) if _ga_forest_regressor_create else None
        self.max_features = mf

    def fit(self, X: Tensor, y: Tensor):
        if self._ptr and _ga_forest_fit:
            with no_grad():
                _ga_forest_fit(self._ptr, X._c_ptr, y._c_ptr)
        return self

    def predict(self, X: Tensor) -> Tensor:
        if self._ptr and _ga_forest_predict:
            ptr = _ga_forest_predict(self._ptr, X._c_ptr)
            return Tensor(_c_ptr=ptr, _shape=(X.shape[0],), _dtype=float)
        return Tensor([0.0] * X.shape[0], dtype=float)
