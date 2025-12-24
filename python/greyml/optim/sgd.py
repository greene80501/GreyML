"""SGD optimizer.
Implements plain/with-momentum SGD updates for GreyML models.
"""

from typing import List, Tuple
import ctypes
import numpy as np
from .optimizer import Optimizer
from ..tensor import Tensor, no_grad, _lib

_c_void_p = ctypes.c_void_p
_c_size_t = ctypes.c_size_t
_c_float = ctypes.c_float

_ga_sgd_create = getattr(_lib, "ga_sgd_create", None)
_ga_sgd_step = getattr(_lib, "ga_sgd_step", None)
_ga_sgd_free = getattr(_lib, "ga_sgd_free", None)

if _ga_sgd_create is not None:
    _ga_sgd_create.restype = _c_void_p
    _ga_sgd_create.argtypes = [ctypes.POINTER(_c_void_p), _c_size_t, _c_float, _c_float, _c_float]
if _ga_sgd_step is not None:
    _ga_sgd_step.restype = None
    _ga_sgd_step.argtypes = [_c_void_p]
if _ga_sgd_free is not None:
    _ga_sgd_free.restype = None
    _ga_sgd_free.argtypes = [_c_void_p]


def _as_void_p(ptr: object) -> _c_void_p:
    if isinstance(ptr, _c_void_p):
        return ptr
    return _c_void_p(int(ptr))


class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr: float, momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros(p.shape, dtype=p.dtype) for p in params]
        self._c_opt = None
        self._c_param_array = None
        self._c_signature: Tuple[Tuple[int, ...], Tuple[float, float, float]] | None = None
        self._c_opt_failed = False

    def step(self):
        if self._can_use_c():
            self._maybe_init_c()
            if self._c_opt is not None:
                _ga_sgd_step(self._c_opt)
                return
        with no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue

                grad_arr = param.grad.numpy()
                param_arr = param.numpy()

                if self.weight_decay != 0:
                    grad_arr = grad_arr + self.weight_decay * param_arr

                if self.momentum != 0:
                    self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grad_arr
                    update = self.velocity[i]
                else:
                    update = -self.lr * grad_arr

                param.data = param_arr + update

    def _can_use_c(self) -> bool:
        if _ga_sgd_create is None or _ga_sgd_step is None:
            return False
        if self._c_opt_failed:
            return False
        for param in self.params:
            if param.dtype != np.float32:
                return False
            if getattr(param, "_grad", None) is not None:
                return False
        return True

    def _maybe_init_c(self) -> None:
        param_ids = tuple(id(p) for p in self.params)
        hparams = (float(self.lr), float(self.momentum), float(self.weight_decay))
        signature = (param_ids, hparams)
        if self._c_opt is not None and self._c_signature != signature:
            if _ga_sgd_free is not None:
                _ga_sgd_free(self._c_opt)
            self._c_opt = None
            self._c_param_array = None
        if self._c_opt is None:
            if not self.params:
                return
            ptrs = (_c_void_p * len(self.params))(
                *(_as_void_p(p._c_ptr) for p in self.params)
            )
            self._c_param_array = ptrs
            try:
                self._c_opt = _ga_sgd_create(ptrs, len(self.params), *hparams)
                self._c_signature = signature
            except OSError:
                self._c_opt_failed = True
                self._c_opt = None
                self._c_param_array = None

    def __del__(self):
        if getattr(self, "_c_opt", None) is not None and _ga_sgd_free is not None:
            try:
                _ga_sgd_free(self._c_opt)
            except Exception:
                pass
