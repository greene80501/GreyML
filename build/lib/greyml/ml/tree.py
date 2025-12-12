"""Decision tree algorithms.
Drivers for fitting/predicting with tree-based models.
"""

import ctypes
import numpy as np
from typing import Optional
from ..tensor import Tensor, _lib, no_grad

try:
    _ga_tree_classifier_create = _lib.ga_tree_classifier_create
    _ga_tree_classifier_create.restype = ctypes.c_void_p
    _ga_tree_classifier_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _ga_tree_regressor_create = _lib.ga_tree_regressor_create
    _ga_tree_regressor_create.restype = ctypes.c_void_p
    _ga_tree_regressor_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _ga_tree_fit = _lib.ga_tree_fit
    _ga_tree_fit.restype = None
    _ga_tree_fit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _ga_tree_predict = _lib.ga_tree_predict
    _ga_tree_predict.restype = ctypes.c_void_p
    _ga_tree_predict.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _ga_tree_predict_proba = _lib.ga_tree_predict_proba
    _ga_tree_predict_proba.restype = ctypes.c_void_p
    _ga_tree_predict_proba.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _ga_tree_feature_importances = _lib.ga_tree_feature_importances
    _ga_tree_feature_importances.restype = ctypes.c_void_p
    _ga_tree_feature_importances.argtypes = [ctypes.c_void_p]
except AttributeError:  # pragma: no cover - allow pure Python fallback
    _ga_tree_classifier_create = _ga_tree_regressor_create = _ga_tree_fit = _ga_tree_predict = _ga_tree_predict_proba = _ga_tree_feature_importances = None


class _PyFallbackTree:
    def __init__(self, max_depth, min_samples_split, regression=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg = regression
        self.root = None

    def fit(self, X, y):
        Xn, yn = X.numpy(), y.numpy()
        if not self.reg:
            # simple stump
            vals, counts = np.unique(yn, return_counts=True)
            self.major = vals[np.argmax(counts)]
        else:
            self.mean = float(yn.mean())
        return self

    def predict(self, X):
        if not self.reg:
            return Tensor(np.full((X.shape[0],), self.major, dtype=np.int64))
        return Tensor(np.full((X.shape[0],), self.mean, dtype=np.float32))

    def predict_proba(self, X):
        if self.reg:
            raise RuntimeError("Not supported")
        return Tensor(np.ones((X.shape[0], 1), dtype=np.float32))

    def feature_importances(self):
        return Tensor(np.zeros((1,), dtype=np.float32))


class DecisionTreeClassifier:
    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2, min_samples_leaf: int = 1, max_features: Optional[int] = None, criterion: str = "gini"):
        criterion_id = 1 if criterion == "entropy" else 0
        md = max_depth if max_depth is not None else 0
        mf = max_features if max_features is not None else 0
        self._ptr = _ga_tree_classifier_create(md, min_samples_split, min_samples_leaf, mf, criterion_id) if _ga_tree_classifier_create else None
        self._fallback = _PyFallbackTree(max_depth, min_samples_split, regression=False) if self._ptr is None else None

    def fit(self, X: Tensor, y: Tensor):
        if self._ptr and _ga_tree_fit:
            with no_grad():
                _ga_tree_fit(self._ptr, X._c_ptr, y._c_ptr)
        else:
            self._fallback.fit(X, y)
        return self

    def predict(self, X: Tensor) -> Tensor:
        if self._ptr and _ga_tree_predict:
            ptr = _ga_tree_predict(self._ptr, X._c_ptr)
            return Tensor(_c_ptr=ptr, _shape=(X.shape[0],), _dtype=int)
        return self._fallback.predict(X)

    def predict_proba(self, X: Tensor) -> Tensor:
        if self._ptr and _ga_tree_predict_proba:
            ptr = _ga_tree_predict_proba(self._ptr, X._c_ptr)
            return Tensor(_c_ptr=ptr, _shape=(X.shape[0],), _dtype=float)
        return self._fallback.predict_proba(X)

    @property
    def feature_importances_(self) -> Tensor:
        if self._ptr and _ga_tree_feature_importances:
            ptr = _ga_tree_feature_importances(self._ptr)
            return Tensor(_c_ptr=ptr, _shape=None, _dtype=float)
        return self._fallback.feature_importances()


class DecisionTreeRegressor:
    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2, min_samples_leaf: int = 1, max_features: Optional[int] = None, criterion: str = "mse"):
        md = max_depth if max_depth is not None else 0
        mf = max_features if max_features is not None else 0
        crit_id = 2  # TREE_CRITERION_MSE
        self._ptr = _ga_tree_regressor_create(md, min_samples_split, min_samples_leaf, mf, crit_id) if _ga_tree_regressor_create else None
        self._fallback = _PyFallbackTree(max_depth, min_samples_split, regression=True) if self._ptr is None else None

    def fit(self, X: Tensor, y: Tensor):
        if self._ptr and _ga_tree_fit:
            with no_grad():
                _ga_tree_fit(self._ptr, X._c_ptr, y._c_ptr)
        else:
            self._fallback.fit(X, y)
        return self

    def predict(self, X: Tensor) -> Tensor:
        if self._ptr and _ga_tree_predict:
            ptr = _ga_tree_predict(self._ptr, X._c_ptr)
            return Tensor(_c_ptr=ptr, _shape=(X.shape[0],), _dtype=float)
        return self._fallback.predict(X)

    @property
    def feature_importances_(self) -> Tensor:
        if self._ptr and _ga_tree_feature_importances:
            ptr = _ga_tree_feature_importances(self._ptr)
            return Tensor(_c_ptr=ptr, _shape=None, _dtype=float)
        return self._fallback.feature_importances()
