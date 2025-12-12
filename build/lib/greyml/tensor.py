"""GreyML Tensor implementation.
Wraps the C backend tensor with Python helpers, autograd hooks, and numpy interop.
"""

import ctypes
import os
import sys
from pathlib import Path
from typing import Tuple, Union, List, Optional, Any, Iterable
import numpy as np


# Load the C library
def _load_library():
    dll_name = "greyarea.dll"
    search_paths = [
        Path(__file__).parent / dll_name,
        Path(sys.prefix) / "Library" / "bin" / dll_name,
        Path(sys.prefix) / "DLLs" / dll_name,
    ]

    # Also search PATH
    for path in os.environ.get("PATH", "").split(os.pathsep):
        if path:
            search_paths.append(Path(path) / dll_name)

    for path in search_paths:
        if path.exists():
            try:
                return ctypes.CDLL(str(path))
            except OSError as e:
                print(f"Warning: Failed to load {path}: {e}")
                continue

    raise ImportError(
        f"Could not find {dll_name}. Please build the library first with scripts\\build.bat\n"
        f"Searched paths: {search_paths}"
    )


_lib = _load_library()

# -----------------------------------------------------------------------------
# Autograd state helpers (Python-side)
# -----------------------------------------------------------------------------
_grad_enabled: bool = True


def _set_grad_enabled(enabled: bool) -> None:
    """Toggle autograd tracking for Python wrappers (best-effort sync to C)."""
    global _grad_enabled
    _grad_enabled = bool(enabled)
    try:
        _lib.ga_set_grad_enabled(bool(enabled))
    except Exception:
        # Older DLLs may not expose the toggle; ignore silently.
        pass


def _is_grad_enabled() -> bool:
    return _grad_enabled

# Define C types for better readability
c_tensor_ptr = ctypes.c_void_p
c_dtype = ctypes.c_int
c_bool = ctypes.c_bool
c_int = ctypes.c_int
c_int64 = ctypes.c_int64
c_float = ctypes.c_float
c_void_p = ctypes.c_void_p

# ---------------------------------------------------------------------
# Configure C function signatures
# ---------------------------------------------------------------------

# Core Tensor API
_lib.ga_tensor_empty.restype = c_tensor_ptr
_lib.ga_tensor_empty.argtypes = [c_int, ctypes.POINTER(c_int64), c_dtype]

_lib.ga_tensor_zeros.restype = c_tensor_ptr
_lib.ga_tensor_zeros.argtypes = [c_int, ctypes.POINTER(c_int64), c_dtype]

_lib.ga_tensor_ones.restype = c_tensor_ptr
_lib.ga_tensor_ones.argtypes = [c_int, ctypes.POINTER(c_int64), c_dtype]

_lib.ga_tensor_from_data.restype = c_tensor_ptr
_lib.ga_tensor_from_data.argtypes = [c_int, ctypes.POINTER(c_int64), c_dtype, c_void_p]

_lib.ga_tensor_release.restype = None
_lib.ga_tensor_release.argtypes = [c_tensor_ptr]

_lib.ga_tensor_retain.restype = None
_lib.ga_tensor_retain.argtypes = [c_tensor_ptr]

_lib.ga_tensor_clone.restype = c_tensor_ptr
_lib.ga_tensor_clone.argtypes = [c_tensor_ptr]

_lib.ga_tensor_contiguous.restype = c_tensor_ptr
_lib.ga_tensor_contiguous.argtypes = [c_tensor_ptr]

_lib.ga_tensor_detach.restype = c_tensor_ptr
_lib.ga_tensor_detach.argtypes = [c_tensor_ptr]

_lib.ga_tensor_reshape.restype = c_tensor_ptr
_lib.ga_tensor_reshape.argtypes = [c_tensor_ptr, c_int, ctypes.POINTER(c_int64)]

_lib.ga_tensor_unsqueeze.restype = c_tensor_ptr
_lib.ga_tensor_unsqueeze.argtypes = [c_tensor_ptr, c_int]

_lib.ga_tensor_squeeze.restype = c_tensor_ptr
_lib.ga_tensor_squeeze.argtypes = [c_tensor_ptr, c_int]

_lib.ga_transpose.restype = c_tensor_ptr
_lib.ga_transpose.argtypes = [c_tensor_ptr]

_lib.ga_tensor_is_contiguous.restype = c_bool
_lib.ga_tensor_is_contiguous.argtypes = [c_tensor_ptr]

# ga_tensor_ndim may not exist in all builds – guard it
try:
    _ga_tensor_ndim = _lib.ga_tensor_ndim
    _ga_tensor_ndim.restype = c_int
    _ga_tensor_ndim.argtypes = [c_tensor_ptr]
except AttributeError:
    _ga_tensor_ndim = None

_lib.ga_tensor_get_shape.restype = None
_lib.ga_tensor_get_shape.argtypes = [c_tensor_ptr, ctypes.POINTER(c_int64)]

_lib.ga_tensor_copy_from.restype = None
_lib.ga_tensor_copy_from.argtypes = [c_tensor_ptr, c_void_p]

_lib.ga_tensor_copy_to.restype = None
_lib.ga_tensor_copy_to.argtypes = [c_tensor_ptr, c_void_p]

_lib.ga_tensor_set_requires_grad.restype = None
_lib.ga_tensor_set_requires_grad.argtypes = [c_tensor_ptr, c_bool]

_lib.ga_tensor_get_grad.restype = c_tensor_ptr
_lib.ga_tensor_get_grad.argtypes = [c_tensor_ptr]

# Grad helpers
try:
    _ga_tensor_zero_grad = _lib.ga_tensor_zero_grad
    _ga_tensor_zero_grad.restype = None
    _ga_tensor_zero_grad.argtypes = [c_tensor_ptr]
except AttributeError:
    _ga_tensor_zero_grad = None

# Optional functions – wrap in try/except so import doesn't explode
try:
    _ga_tensor_fill = _lib.ga_tensor_fill
    _ga_tensor_fill.restype = None
    _ga_tensor_fill.argtypes = [c_tensor_ptr, c_void_p]
except AttributeError:
    _ga_tensor_fill = None

try:
    _ga_tensor_save = _lib.ga_tensor_save
    _ga_tensor_save.restype = c_int
    _ga_tensor_save.argtypes = [c_tensor_ptr, ctypes.c_char_p]
except AttributeError:
    _ga_tensor_save = None

try:
    _ga_tensor_load = _lib.ga_tensor_load
    _ga_tensor_load.restype = c_tensor_ptr
    _ga_tensor_load.argtypes = [ctypes.c_char_p]
except AttributeError:
    _ga_tensor_load = None

# CSV loader
try:
    _ga_csv_load = _lib.ga_csv_load
    _ga_csv_load.restype = c_tensor_ptr
    _ga_csv_load.argtypes = [ctypes.c_char_p, c_bool, ctypes.POINTER(c_int), ctypes.POINTER(c_int)]
except AttributeError:
    _ga_csv_load = None

try:
    _ga_tensor_randn_ = _lib.ga_tensor_randn_
    _ga_tensor_randn_.restype = None
    _ga_tensor_randn_.argtypes = [c_tensor_ptr]
except AttributeError:
    _ga_tensor_randn_ = None

# Operations
_lib.ga_add.restype = c_tensor_ptr
_lib.ga_add.argtypes = [c_tensor_ptr, c_tensor_ptr]

_lib.ga_sub.restype = c_tensor_ptr
_lib.ga_sub.argtypes = [c_tensor_ptr, c_tensor_ptr]

_lib.ga_mul.restype = c_tensor_ptr
_lib.ga_mul.argtypes = [c_tensor_ptr, c_tensor_ptr]

_lib.ga_div.restype = c_tensor_ptr
_lib.ga_div.argtypes = [c_tensor_ptr, c_tensor_ptr]

_lib.ga_add_scalar.restype = c_tensor_ptr
_lib.ga_add_scalar.argtypes = [c_tensor_ptr, c_float]

_lib.ga_mul_scalar.restype = c_tensor_ptr
_lib.ga_mul_scalar.argtypes = [c_tensor_ptr, c_float]

_lib.ga_relu.restype = c_tensor_ptr
_lib.ga_relu.argtypes = [c_tensor_ptr]

_lib.ga_sigmoid.restype = c_tensor_ptr
_lib.ga_sigmoid.argtypes = [c_tensor_ptr]

_lib.ga_softmax.restype = c_tensor_ptr
_lib.ga_softmax.argtypes = [c_tensor_ptr, c_int]

_lib.ga_sum.restype = c_tensor_ptr
_lib.ga_sum.argtypes = [c_tensor_ptr, c_int, c_bool]

_lib.ga_mean.restype = c_tensor_ptr
_lib.ga_mean.argtypes = [c_tensor_ptr, c_int, c_bool]

_lib.ga_matmul.restype = c_tensor_ptr
_lib.ga_matmul.argtypes = [c_tensor_ptr, c_tensor_ptr]

# Losses
_lib.ga_mse_loss.restype = c_tensor_ptr
_lib.ga_mse_loss.argtypes = [c_tensor_ptr, c_tensor_ptr, c_int]
for _name, _restype, _argtypes in [
    ("ga_l1_loss", c_tensor_ptr, [c_tensor_ptr, c_tensor_ptr, c_int]),
    ("ga_cross_entropy_loss", c_tensor_ptr, [c_tensor_ptr, c_tensor_ptr, c_int]),
    ("ga_binary_cross_entropy", c_tensor_ptr, [c_tensor_ptr, c_tensor_ptr, c_int]),
    ("ga_huber_loss", c_tensor_ptr, [c_tensor_ptr, c_tensor_ptr, c_float, c_int]),
]:
    if hasattr(_lib, _name):
        func = getattr(_lib, _name)
        func.restype = _restype
        func.argtypes = _argtypes
    else:
        setattr(_lib, _name, None)

# Convolution and pooling
try:
    _lib.ga_conv2d.restype = c_tensor_ptr
    _lib.ga_conv2d.argtypes = [c_tensor_ptr, c_tensor_ptr, c_tensor_ptr, c_int, c_int, c_int, c_int]
    _lib.ga_max_pool2d.restype = c_tensor_ptr
    _lib.ga_max_pool2d.argtypes = [c_tensor_ptr, c_int, c_int, c_int, c_int]
    _lib.ga_avg_pool2d.restype = c_tensor_ptr
    _lib.ga_avg_pool2d.argtypes = [c_tensor_ptr, c_int, c_int, c_int]
    _lib.ga_adaptive_avg_pool2d.restype = c_tensor_ptr
    _lib.ga_adaptive_avg_pool2d.argtypes = [c_tensor_ptr, c_int, c_int]
except AttributeError:
    _lib.ga_conv2d = None
    _lib.ga_max_pool2d = None
    _lib.ga_avg_pool2d = None
    _lib.ga_adaptive_avg_pool2d = None

# Autograd
_lib.ga_backward.restype = None
_lib.ga_backward.argtypes = [c_tensor_ptr, c_tensor_ptr]

_lib.ga_detach.restype = None
_lib.ga_detach.argtypes = [c_tensor_ptr]

_lib.ga_set_grad_enabled.restype = None
_lib.ga_set_grad_enabled.argtypes = [c_bool]

_lib.ga_is_grad_enabled.restype = c_bool
_lib.ga_is_grad_enabled.argtypes = []

# Dtype mapping
_DTYPE_MAP = {
    np.float32: 0, np.dtype("float32"): 0,
    np.float64: 1, np.dtype("float64"): 1,
    np.int32: 2,   np.dtype("int32"): 2,
    np.int64: 3,   np.dtype("int64"): 3,
    np.uint8: 4,   np.dtype("uint8"): 4,
    bool: 5,       np.dtype("bool"): 5,
}

_DTYPE_REV_MAP = {
    0: np.float32,
    1: np.float64,
    2: np.int32,
    3: np.int64,
    4: np.uint8,
    5: bool,
}


def _as_tensor(value: Union["Tensor", np.ndarray, float, int], dtype: np.dtype) -> "Tensor":
    """Convert scalars/arrays to Tensor (grad disabled)."""
    if isinstance(value, Tensor):
        return value
    arr = np.array(value, dtype=dtype)
    # Avoid tracking when creating helper tensors
    prev = _is_grad_enabled()
    _set_grad_enabled(False)
    try:
        return Tensor(arr, dtype=dtype)
    finally:
        _set_grad_enabled(prev)


def _unbroadcast(grad: "Tensor", target_shape: Tuple[int, ...]) -> "Tensor":
    """
    Sum gradient over broadcasted dimensions to match target shape.
    Uses numpy for simplicity, keeping grad disabled during the process.
    """
    if grad.shape == target_shape:
        return grad
    arr = grad.numpy()
    # Remove leading dims
    while arr.ndim > len(target_shape):
        arr = arr.sum(axis=0)
    for axis, (gdim, tdim) in enumerate(zip(arr.shape, target_shape)):
        if tdim == 1 and gdim != 1:
            arr = arr.sum(axis=axis, keepdims=True)
    prev = _is_grad_enabled()
    _set_grad_enabled(False)
    try:
        return Tensor(arr.astype(grad.dtype))
    finally:
        _set_grad_enabled(prev)


def _mul_scalar_no_grad(tensor: "Tensor", scalar: float) -> "Tensor":
    """Multiply tensor by scalar without tracking gradients."""
    prev = _is_grad_enabled()
    _set_grad_enabled(False)
    try:
        ptr = _lib.ga_mul_scalar(tensor._c_ptr, float(scalar))
        return Tensor(_c_ptr=ptr, _shape=tensor.shape, _dtype=tensor.dtype)
    finally:
        _set_grad_enabled(prev)


class Tensor:
    """
    Core tensor class that wraps the C GATensor* pointer.
    Provides numpy-like interface with autograd support.
    """

    def __init__(
        self,
        data: Optional[Union[List, np.ndarray]] = None,
        dtype: np.dtype = np.float32,
        requires_grad: bool = False,
        _c_ptr: Optional[int] = None,
        _shape: Optional[Tuple[int, ...]] = None,
        _dtype: Optional[np.dtype] = None,
        _requires_grad: Optional[bool] = None,
    ):
        """
        Create a tensor.

        Args:
            data: Initial data (list, tuple, or numpy array)
            dtype: Data type (np.float32, np.float64, etc.)
            requires_grad: Enable gradient tracking for autograd
            _c_ptr: Internal use only - wrap existing C pointer
            _shape: Internal use only - known shape
            _dtype: Internal use only - known dtype
        """
        # Autograd bookkeeping
        self._grad: Optional["Tensor"] = None
        self._backward = lambda: None
        self._prev: Tuple["Tensor", ...] = ()
        self._op: Optional[str] = None

        if _c_ptr is not None:
            # Wrap existing C tensor pointer
            self._c_ptr = ctypes.c_void_p(_c_ptr)
            self._shape = _shape
            self._dtype = _dtype if _dtype is not None else dtype
            if _requires_grad is None:
                _requires_grad = requires_grad
            self._requires_grad = bool(_requires_grad)
            return

        # Convert data to numpy array
        if data is not None:
            if isinstance(data, (list, tuple)):
                arr = np.array(data, dtype=dtype)
            elif isinstance(data, np.ndarray):
                arr = data.astype(dtype)
            else:
                raise ValueError("Data must be list, tuple, or numpy array")
        else:
            raise ValueError("Data must be provided")

        self._shape = tuple(arr.shape)
        self._dtype = arr.dtype

        # Create via C API
        ndim = len(arr.shape)
        shape_arr = (c_int64 * ndim)(*arr.shape)
        dtype_code = _DTYPE_MAP.get(arr.dtype, 0)

        self._c_ptr = _lib.ga_tensor_empty(ndim, shape_arr, dtype_code)
        if not self._c_ptr:
            raise MemoryError(f"Failed to allocate tensor with shape {arr.shape}")

        # Copy data
        _lib.ga_tensor_copy_from(self._c_ptr, arr.ctypes.data)

        requires_flag = _requires_grad if _requires_grad is not None else requires_grad
        self._requires_grad = bool(requires_flag)
        if requires_flag:
            _lib.ga_tensor_set_requires_grad(self._c_ptr, True)

    def __del__(self):
        """Release the C tensor when Python object is garbage collected."""
        if hasattr(self, "_c_ptr") and self._c_ptr:
            try:
                _lib.ga_tensor_release(self._c_ptr)
            except Exception:
                # Avoid noisy errors on interpreter shutdown
                pass

    # ------------------------------------------------------------------
    # Shape / dtype / metadata
    # ------------------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        if self._shape is None:
            if _ga_tensor_ndim is not None:
                ndim = _ga_tensor_ndim(self._c_ptr)
                shape_arr = (c_int64 * ndim)()
                _lib.ga_tensor_get_shape(self._c_ptr, shape_arr)
                self._shape = tuple(shape_arr[i] for i in range(ndim))
            else:
                # Fallback: ask C for shape via some other API or assume cached.
                # For v0.1, we assume shape is known at construction.
                raise RuntimeError("Shape unknown and ga_tensor_ndim is not available")
        return self._shape

    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        """Get data type."""
        return self._dtype

    @property
    def requires_grad(self) -> bool:
        """Check if gradient tracking is enabled."""
        return bool(getattr(self, "_requires_grad", False))

    @property
    def grad(self) -> Optional["Tensor"]:
        """Return accumulated gradient (Python-side autograd)."""
        try:
            ptr = _lib.ga_tensor_get_grad(self._c_ptr)
            if ptr:
                _lib.ga_tensor_retain(ptr)
                return Tensor(
                    _c_ptr=ptr,
                    _shape=self.shape,
                    _dtype=self.dtype,
                    _requires_grad=False,
                )
        except Exception:
            # Fall back to Python-tracked gradient if C API not available
            pass
        return self._grad

    def zero_grad(self) -> None:
        """Clear accumulated gradient."""
        self._grad = None
        if _ga_tensor_zero_grad is not None:
            try:
                _ga_tensor_zero_grad(self._c_ptr)
            except Exception:
                pass

    def _accumulate_grad(self, grad: Optional["Tensor"]) -> None:
        """Internal util to sum gradients without tracking."""
        if grad is None:
            return
        if self._grad is None:
            self._grad = grad
        else:
            prev = _is_grad_enabled()
            _set_grad_enabled(False)
            try:
                self._grad = self._grad + grad
            finally:
                _set_grad_enabled(prev)

    # ------------------------------------------------------------------
    # Autograd
    # ------------------------------------------------------------------
    def backward(self, grad: Optional["Tensor"] = None):
        """
        Compute gradients via autograd.

        Args:
            grad: Optional gradient of the output. If None, uses ones.
        """
        if not self.requires_grad:
            return

        if grad is not None and not isinstance(grad, Tensor):
            grad = Tensor(np.array(grad, dtype=self.dtype), dtype=self.dtype)

        # Prefer C-side autograd when available
        try:
            grad_ptr = grad._c_ptr if isinstance(grad, Tensor) else None
            _lib.ga_backward(self._c_ptr, grad_ptr)
            g_ptr = _lib.ga_tensor_get_grad(self._c_ptr)
            if g_ptr:
                _lib.ga_tensor_retain(g_ptr)
                self._grad = Tensor(
                    _c_ptr=g_ptr,
                    _shape=self.shape,
                    _dtype=self.dtype,
                    _requires_grad=False,
                )
                return
        except Exception:
            # Fall back to Python-side autograd if C path is unavailable
            pass

        if grad is None:
            grad = Tensor(np.ones(self.shape, dtype=self.dtype), dtype=self.dtype)
        self._grad = grad

        topo: List["Tensor"] = []
        visited = set()

        def build(t: "Tensor"):
            if id(t) in visited:
                return
            visited.add(id(t))
            for child in getattr(t, "_prev", ()):
                build(child)
            topo.append(t)

        build(self)
        for t in reversed(topo):
            if hasattr(t, "_backward") and t._backward:
                t._backward()

    def detach(self) -> "Tensor":
        """Return a detached tensor that's no longer part of the computation graph."""
        new_ptr = _lib.ga_tensor_detach(self._c_ptr)
        out = Tensor(_c_ptr=new_ptr, _shape=self.shape, _dtype=self._dtype, _requires_grad=False)
        out._backward = lambda: None
        out._prev = ()
        out._grad = None
        return out

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------
    def numpy(self) -> np.ndarray:
        """Convert to numpy array (copies data)."""
        arr = np.empty(self.shape, dtype=self.dtype)
        _lib.ga_tensor_copy_to(self._c_ptr, arr.ctypes.data)
        return arr

    @property
    def data(self) -> np.ndarray:
        """Alias for raw tensor data (detached as numpy array)."""
        return self.numpy()

    @data.setter
    def data(self, value: Union[np.ndarray, List, float, int]) -> None:
        arr = np.array(value, dtype=self.dtype)
        if arr.shape != self.shape:
            raise ValueError(f"New data shape {arr.shape} does not match tensor shape {self.shape}")
        _lib.ga_tensor_copy_from(self._c_ptr, arr.ctypes.data)

    def item(self) -> Union[float, int]:
        """Get scalar value from 0-dim tensor."""
        if self.size != 1:
            raise ValueError("Item can only be called on 0-dimensional tensors")
        return self.numpy().item()

    @property
    def size(self) -> int:
        """Get total number of elements."""
        return int(np.prod(self.shape))

    # ------------------------------------------------------------------
    # Structural ops
    # ------------------------------------------------------------------
    def clone(self) -> "Tensor":
        """Create a copy of the tensor."""
        new_ptr = _lib.ga_tensor_clone(self._c_ptr)
        requires_grad = self.requires_grad and _is_grad_enabled()
        out = Tensor(_c_ptr=new_ptr, _shape=self.shape, _dtype=self.dtype, _requires_grad=requires_grad)
        if requires_grad:
            out._prev = (self,)

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    self._accumulate_grad(out.grad)

            out._backward = _backward
        return out

    def contiguous(self) -> "Tensor":
        """Return a contiguous copy if not already contiguous."""
        new_ptr = _lib.ga_tensor_contiguous(self._c_ptr)
        out = Tensor(_c_ptr=new_ptr, _shape=self.shape, _dtype=self.dtype, _requires_grad=self.requires_grad and _is_grad_enabled())
        if out.requires_grad:
            out._prev = (self,)
            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    self._accumulate_grad(out.grad)
            out._backward = _backward
        return out

    def reshape(self, *shape: Union[int, Tuple[int, ...]]) -> "Tensor":
        """Reshape tensor."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)

        ndim = len(shape)
        shape_arr = (c_int64 * ndim)(*shape)
        new_ptr = _lib.ga_tensor_reshape(self._c_ptr, ndim, shape_arr)
        requires_grad = self.requires_grad and _is_grad_enabled()
        out = Tensor(_c_ptr=new_ptr, _shape=shape, _dtype=self.dtype, _requires_grad=requires_grad)
        if requires_grad:
            parent_shape = self.shape
            out._prev = (self,)

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    self._accumulate_grad(out.grad.reshape(parent_shape))

            out._backward = _backward
        return out

    def unsqueeze(self, dim: int) -> "Tensor":
        """Add a dimension of size 1 at the specified position."""
        new_ptr = _lib.ga_tensor_unsqueeze(self._c_ptr, dim)
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)
        requires_grad = self.requires_grad and _is_grad_enabled()
        out = Tensor(_c_ptr=new_ptr, _shape=tuple(new_shape), _dtype=self._dtype, _requires_grad=requires_grad)
        if requires_grad:
            out._prev = (self,)

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    self._accumulate_grad(out.grad.squeeze(dim))

            out._backward = _backward
        return out

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        """Remove dimensions of size 1."""
        if dim is not None:
            new_ptr = _lib.ga_tensor_squeeze(self._c_ptr, dim)
            new_shape = list(self.shape)
            if new_shape[dim] == 1:
                new_shape.pop(dim)
            requires_grad = self.requires_grad and _is_grad_enabled()
            out = Tensor(_c_ptr=new_ptr, _shape=tuple(new_shape), _dtype=self._dtype, _requires_grad=requires_grad)
            if requires_grad:
                out._prev = (self,)

                def _backward():
                    if out.grad is None:
                        return
                    with no_grad():
                        self._accumulate_grad(out.grad.unsqueeze(dim))

                out._backward = _backward
            return out
        # Squeeze all dims of size 1 in Python
        new_shape = tuple(s for s in self.shape if s != 1)
        return self.reshape(new_shape)

    def transpose(self) -> "Tensor":
        """Transpose tensor (2D only for v0.1)."""
        if self.ndim != 2:
            raise ValueError("Transpose only supports 2D tensors in v0.1")
        new_ptr = _lib.ga_transpose(self._c_ptr)
        new_shape = (self.shape[1], self.shape[0])
        requires_grad = self.requires_grad and _is_grad_enabled()
        out = Tensor(_c_ptr=new_ptr, _shape=new_shape, _dtype=self._dtype, _requires_grad=requires_grad)
        if requires_grad:
            out._prev = (self,)

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    self._accumulate_grad(out.grad.transpose())

            out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Arithmetic operations
    # ------------------------------------------------------------------
    def __neg__(self) -> "Tensor":
        try:
            ptr = _lib.ga_neg(self._c_ptr)
        except AttributeError:
            ptr = None
        if ptr:
            out = Tensor(_c_ptr=ptr, _dtype=self.dtype, _shape=self._shape, _requires_grad=_is_grad_enabled() and self.requires_grad)
        else:
            out = Tensor((-self.numpy()).astype(self.dtype), dtype=self.dtype, _requires_grad=_is_grad_enabled() and self.requires_grad)
        if out.requires_grad:
            out._prev = (self,)

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    self._accumulate_grad(_mul_scalar_no_grad(out.grad, -1.0))

            out._backward = _backward
        return out

    def __add__(self, other: Union["Tensor", float, int]) -> "Tensor":
        """Element-wise addition."""
        if isinstance(other, Tensor):
            result_ptr = _lib.ga_add(self._c_ptr, other._c_ptr)
            if not result_ptr:
                raise RuntimeError("Addition failed")
            out = Tensor(_c_ptr=result_ptr, _dtype=self.dtype)
            requires_grad = _is_grad_enabled() and (self.requires_grad or other.requires_grad)
            out._requires_grad = requires_grad
            if requires_grad:
                out._prev = (self, other)

                def _backward():
                    if out.grad is None:
                        return
                    with no_grad():
                        if self.requires_grad:
                            self._accumulate_grad(_unbroadcast(out.grad, self.shape))
                        if other.requires_grad:
                            other._accumulate_grad(_unbroadcast(out.grad, other.shape))

                out._backward = _backward
            return out
        # Scalar add (no grad for scalar)
        result_ptr = _lib.ga_add_scalar(self._c_ptr, float(other))
        out = Tensor(_c_ptr=result_ptr, _shape=self.shape, _dtype=self.dtype)
        if _is_grad_enabled() and self.requires_grad:
            out._requires_grad = True
            out._prev = (self,)

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    self._accumulate_grad(out.grad)

            out._backward = _backward
        return out

    def __radd__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        """Element-wise subtraction."""
        if isinstance(other, Tensor):
            result_ptr = _lib.ga_sub(self._c_ptr, other._c_ptr)
            if not result_ptr:
                raise RuntimeError("Subtraction failed")
            out = Tensor(_c_ptr=result_ptr, _dtype=self.dtype)
            requires_grad = _is_grad_enabled() and (self.requires_grad or other.requires_grad)
            out._requires_grad = requires_grad
            if requires_grad:
                out._prev = (self, other)

                def _backward():
                    if out.grad is None:
                        return
                    with no_grad():
                        if self.requires_grad:
                            self._accumulate_grad(_unbroadcast(out.grad, self.shape))
                        if other.requires_grad:
                            other._accumulate_grad(_unbroadcast(_mul_scalar_no_grad(out.grad, -1.0), other.shape))

                out._backward = _backward
            return out
        # scalar
        result_ptr = _lib.ga_add_scalar(self._c_ptr, -float(other))
        out = Tensor(_c_ptr=result_ptr, _shape=self.shape, _dtype=self.dtype)
        if _is_grad_enabled() and self.requires_grad:
            out._requires_grad = True
            out._prev = (self,)

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    self._accumulate_grad(out.grad)

            out._backward = _backward
        return out

    def __rsub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return (-self).__add__(other)

    def __mul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        """Element-wise multiplication."""
        if isinstance(other, Tensor):
            result_ptr = _lib.ga_mul(self._c_ptr, other._c_ptr)
            if not result_ptr:
                raise RuntimeError("Multiplication failed")
            out = Tensor(_c_ptr=result_ptr, _dtype=self.dtype)
            requires_grad = _is_grad_enabled() and (self.requires_grad or other.requires_grad)
            out._requires_grad = requires_grad
            if requires_grad:
                out._prev = (self, other)

                def _backward():
                    if out.grad is None:
                        return
                    with no_grad():
                        if self.requires_grad:
                            self._accumulate_grad(_unbroadcast(out.grad * other, self.shape))
                        if other.requires_grad:
                            other._accumulate_grad(_unbroadcast(out.grad * self, other.shape))

                out._backward = _backward
            return out
        # scalar
        result_ptr = _lib.ga_mul_scalar(self._c_ptr, float(other))
        out = Tensor(_c_ptr=result_ptr, _shape=self.shape, _dtype=self.dtype)
        if _is_grad_enabled() and self.requires_grad:
            out._requires_grad = True
            out._prev = (self,)

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    self._accumulate_grad(_mul_scalar_no_grad(out.grad, float(other)))

            out._backward = _backward
        return out

    def __rmul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: Union["Tensor", float, int]) -> "Tensor":
        """Element-wise division."""
        if isinstance(other, Tensor):
            result_ptr = _lib.ga_div(self._c_ptr, other._c_ptr)
            if not result_ptr:
                raise RuntimeError("Division failed")
            out = Tensor(_c_ptr=result_ptr, _dtype=self.dtype)
            requires_grad = _is_grad_enabled() and (self.requires_grad or other.requires_grad)
            out._requires_grad = requires_grad
            if requires_grad:
                out._prev = (self, other)

                def _backward():
                    if out.grad is None:
                        return
                    with no_grad():
                        if self.requires_grad:
                            self._accumulate_grad(_unbroadcast(out.grad / other, self.shape))
                        if other.requires_grad:
                            num = out.grad * self
                            denom = other * other
                            other._accumulate_grad(_unbroadcast(num / denom * -1.0, other.shape))

                out._backward = _backward
            return out
        # scalar division
        scale = 1.0 / float(other)
        result_ptr = _lib.ga_mul_scalar(self._c_ptr, scale)
        out = Tensor(_c_ptr=result_ptr, _shape=self.shape, _dtype=self.dtype)
        if _is_grad_enabled() and self.requires_grad:
            out._requires_grad = True
            out._prev = (self,)

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    self._accumulate_grad(_mul_scalar_no_grad(out.grad, scale))

            out._backward = _backward
        return out

    def __rtruediv__(self, other: Union["Tensor", float, int]) -> "Tensor":
        if isinstance(other, Tensor):
            return other.__truediv__(self)
        # scalar / tensor
        other_tensor = _as_tensor(other, self.dtype)
        return other_tensor.__truediv__(self)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication."""
        assert isinstance(other, Tensor), "Can only matmul Tensor by Tensor"
        result_ptr = _lib.ga_matmul(self._c_ptr, other._c_ptr)
        if not result_ptr:
            raise RuntimeError("Matrix multiplication failed")
        out = Tensor(_c_ptr=result_ptr, _dtype=self.dtype)
        requires_grad = _is_grad_enabled() and (self.requires_grad or other.requires_grad)
        out._requires_grad = requires_grad
        if requires_grad:
            out._prev = (self, other)

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    if self.requires_grad:
                        self._accumulate_grad(out.grad @ other.transpose())
                    if other.requires_grad:
                        other._accumulate_grad(self.transpose() @ out.grad)

            out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # In-place / scalar ops
    # ------------------------------------------------------------------
    def fill_(self, value: Union[float, int]) -> "Tensor":
        """Fill tensor with scalar value."""
        if _ga_tensor_fill is None:
            raise RuntimeError("ga_tensor_fill not available in this build")
        val = np.array(value, dtype=self.dtype)
        _ga_tensor_fill(self._c_ptr, val.ctypes.data)
        return self

    # ------------------------------------------------------------------
    # Reduction operations
    # ------------------------------------------------------------------
    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        """Sum elements."""
        dim_val = dim if dim is not None else -1
        result_ptr = _lib.ga_sum(self._c_ptr, dim_val, keepdim)
        if not result_ptr:
            raise RuntimeError("Sum failed")
        # Infer output shape
        dim_resolved = self.ndim - 1 if dim is None else (dim if dim >= 0 else self.ndim + dim)
        if keepdim:
            out_shape = list(self.shape)
            out_shape[dim_resolved] = 1
        else:
            out_shape = [s for i, s in enumerate(self.shape) if i != dim_resolved]
        out = Tensor(_c_ptr=result_ptr, _shape=tuple(out_shape), _dtype=self.dtype)
        requires_grad = _is_grad_enabled() and self.requires_grad
        out._requires_grad = requires_grad
        if requires_grad:
            out._prev = (self,)

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    grad_arr = out.grad.numpy()
                    if not keepdim:
                        grad_arr = np.expand_dims(grad_arr, axis=dim_resolved)
                    grad_arr = np.broadcast_to(grad_arr, self.shape)
                    self._accumulate_grad(Tensor(grad_arr.astype(self.dtype)))

            out._backward = _backward
        return out

    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        """Mean of elements."""
        dim_val = dim if dim is not None else -1
        result_ptr = _lib.ga_mean(self._c_ptr, dim_val, keepdim)
        if not result_ptr:
            raise RuntimeError("Mean failed")
        dim_resolved = self.ndim - 1 if dim is None else (dim if dim >= 0 else self.ndim + dim)
        if keepdim:
            out_shape = list(self.shape)
            out_shape[dim_resolved] = 1
        else:
            out_shape = [s for i, s in enumerate(self.shape) if i != dim_resolved]
        out = Tensor(_c_ptr=result_ptr, _shape=tuple(out_shape), _dtype=self.dtype)
        requires_grad = _is_grad_enabled() and self.requires_grad
        out._requires_grad = requires_grad
        if requires_grad:
            out._prev = (self,)
            reduce_count = self.shape[dim_resolved] if self.ndim > 0 else 1

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    grad_arr = out.grad.numpy()
                    if not keepdim:
                        grad_arr = np.expand_dims(grad_arr, axis=dim_resolved)
                    grad_arr = np.broadcast_to(grad_arr, self.shape) / float(reduce_count)
                    self._accumulate_grad(Tensor(grad_arr.astype(self.dtype)))

            out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------
    def relu(self) -> "Tensor":
        """Apply ReLU activation."""
        result_ptr = _lib.ga_relu(self._c_ptr)
        if not result_ptr:
            raise RuntimeError("ReLU failed")
        out = Tensor(_c_ptr=result_ptr, _dtype=self.dtype)
        requires_grad = _is_grad_enabled() and self.requires_grad
        out._requires_grad = requires_grad
        if requires_grad:
            out._prev = (self,)

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    mask = (self.numpy() > 0).astype(self.dtype)
                    grad_arr = out.grad.numpy() * mask
                    self._accumulate_grad(Tensor(grad_arr.astype(self.dtype)))

            out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        """Apply sigmoid activation."""
        result_ptr = _lib.ga_sigmoid(self._c_ptr)
        if not result_ptr:
            raise RuntimeError("Sigmoid failed")
        out = Tensor(_c_ptr=result_ptr, _dtype=self.dtype)
        requires_grad = _is_grad_enabled() and self.requires_grad
        out._requires_grad = requires_grad
        if requires_grad:
            out._prev = (self,)

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    out_arr = out.numpy()
                    grad_arr = out.grad.numpy() * out_arr * (1.0 - out_arr)
                    self._accumulate_grad(Tensor(grad_arr.astype(self.dtype)))

            out._backward = _backward
        return out

    def softmax(self, dim: int = -1) -> "Tensor":
        """Apply softmax activation."""
        arr = self.numpy()
        dim_resolved = dim if dim >= 0 else arr.ndim + dim
        exp = np.exp(arr - np.max(arr, axis=dim_resolved, keepdims=True))
        arr_out = exp / np.sum(exp, axis=dim_resolved, keepdims=True)
        out = Tensor(arr_out.astype(self.dtype), dtype=self.dtype)
        requires_grad = _is_grad_enabled() and self.requires_grad
        out._requires_grad = requires_grad
        if requires_grad:
            out._prev = (self,)

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    grad_arr = out.grad.numpy()
                    dot = np.sum(grad_arr * arr_out, axis=dim_resolved, keepdims=True)
                    grad_input = arr_out * (grad_arr - dot)
                    self._accumulate_grad(Tensor(grad_input.astype(self.dtype)))

            out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        result_ptr = _lib.ga_exp(self._c_ptr)
        if not result_ptr:
            raise RuntimeError("exp failed")
        out = Tensor(_c_ptr=result_ptr, _dtype=self.dtype)
        requires_grad = _is_grad_enabled() and self.requires_grad
        out._requires_grad = requires_grad
        if requires_grad:
            out._prev = (self,)

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    self._accumulate_grad(out.grad * out)

            out._backward = _backward
        return out

    def log(self) -> "Tensor":
        try:
            result_ptr = _lib.ga_log(self._c_ptr)
        except AttributeError:
            result_ptr = None
        if result_ptr:
            out = Tensor(_c_ptr=result_ptr, _dtype=self.dtype)
        else:
            out = Tensor(np.log(self.numpy()).astype(self.dtype), dtype=self.dtype, _requires_grad=_is_grad_enabled() and self.requires_grad)
        requires_grad = _is_grad_enabled() and self.requires_grad
        out._requires_grad = requires_grad
        if requires_grad:
            out._prev = (self,)

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    self._accumulate_grad(out.grad / self)

            out._backward = _backward
        return out

    def sqrt(self) -> "Tensor":
        result_ptr = _lib.ga_sqrt(self._c_ptr)
        if not result_ptr:
            raise RuntimeError("sqrt failed")
        out = Tensor(_c_ptr=result_ptr, _dtype=self.dtype)
        requires_grad = _is_grad_enabled() and self.requires_grad
        out._requires_grad = requires_grad
        if requires_grad:
            out._prev = (self,)

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    grad_input = out.grad / (out * 2.0)
                    self._accumulate_grad(grad_input)

            out._backward = _backward
        return out

    def abs(self) -> "Tensor":
        try:
            result_ptr = _lib.ga_abs(self._c_ptr)
        except AttributeError:
            result_ptr = None
        if result_ptr:
            out = Tensor(_c_ptr=result_ptr, _dtype=self.dtype)
        else:
            out = Tensor(np.abs(self.numpy()).astype(self.dtype), dtype=self.dtype)
        requires_grad = _is_grad_enabled() and self.requires_grad
        out._requires_grad = requires_grad
        if requires_grad:
            out._prev = (self,)

            def _backward():
                if out.grad is None:
                    return
                with no_grad():
                    sign = np.sign(self.numpy()).astype(self.dtype)
                    grad_arr = out.grad.numpy() * sign
                    self._accumulate_grad(Tensor(grad_arr.astype(self.dtype)))

            out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save tensor to file."""
        if _ga_tensor_save is None:
            raise RuntimeError("ga_tensor_save not available in this build")
        result = _ga_tensor_save(self._c_ptr, path.encode())
        if result != 0:
            raise RuntimeError(f"Failed to save tensor to {path}")

    @staticmethod
    def load(path: str) -> "Tensor":
        """Load tensor from file."""
        if _ga_tensor_load is None:
            raise RuntimeError("ga_tensor_load not available in this build")
        ptr = _ga_tensor_load(path.encode())
        if not ptr:
            raise RuntimeError(f"Failed to load tensor from {path}")
        return Tensor(_c_ptr=ptr, _shape=None, _dtype=np.float32)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        return f"Tensor{self.shape}, dtype={self.dtype}"

    def __repr__(self) -> str:
        return f"Tensor(\n{self.numpy()}\nshape={self.shape}, dtype={self.dtype})"


# ----------------------------------------------------------------------
# Convenience functions
# ----------------------------------------------------------------------
def zeros(*shape, dtype=np.float32, requires_grad: bool = False) -> Tensor:
    """Create tensor of zeros."""
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    shape = tuple(shape)
    ndim = len(shape)
    shape_arr = (c_int64 * ndim)(*shape)
    dtype_code = _DTYPE_MAP.get(dtype, 0)
    ptr = _lib.ga_tensor_zeros(ndim, shape_arr, dtype_code)
    t = Tensor(_c_ptr=ptr, _shape=shape, _dtype=dtype, _requires_grad=requires_grad)
    if requires_grad:
        _lib.ga_tensor_set_requires_grad(t._c_ptr, True)
    return t


def ones(*shape, dtype=np.float32, requires_grad: bool = False) -> Tensor:
    """Create tensor of ones."""
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    shape = tuple(shape)
    ndim = len(shape)
    shape_arr = (c_int64 * ndim)(*shape)
    dtype_code = _DTYPE_MAP.get(dtype, 0)
    ptr = _lib.ga_tensor_ones(ndim, shape_arr, dtype_code)
    t = Tensor(_c_ptr=ptr, _shape=shape, _dtype=dtype, _requires_grad=requires_grad)
    if requires_grad:
        _lib.ga_tensor_set_requires_grad(t._c_ptr, True)
    return t


def randn(*shape, dtype=np.float32, requires_grad: bool = False) -> Tensor:
    """Create tensor with random normal values."""
    t = zeros(*shape, dtype=dtype, requires_grad=requires_grad)
    if _ga_tensor_randn_ is None:
        raise RuntimeError("ga_tensor_randn_ not available in this build")
    _ga_tensor_randn_(t._c_ptr)
    return t


def load_csv(path: str, has_header: bool = True) -> Tensor:
    """Load a CSV file into a Tensor (float32)."""
    if _ga_csv_load is None:
        raise RuntimeError("ga_csv_load not available in this build")
    rows = c_int(0)
    cols = c_int(0)
    ptr = _ga_csv_load(path.encode(), bool(has_header), ctypes.byref(rows), ctypes.byref(cols))
    if not ptr:
        raise RuntimeError(f"Failed to load CSV from {path}")
    shape = (rows.value, cols.value)
    return Tensor(_c_ptr=ptr, _shape=shape, _dtype=np.float32)


_REDUCTION = {"none": 0, "mean": 1, "sum": 2}


def _to_reduction_code(reduction: str) -> int:
    if reduction not in _REDUCTION:
        raise ValueError(f"Invalid reduction: {reduction}")
    return _REDUCTION[reduction]


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    """Apply reduction over all elements (mean, sum, or none)."""
    if reduction == "none":
        return loss
    flat = loss.reshape(loss.size)
    if reduction == "sum":
        return flat.sum()
    if reduction == "mean":
        return flat.sum() / float(loss.size)
    raise ValueError(f"Invalid reduction: {reduction}")


def mse_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    loss = (pred - target) * (pred - target)
    return _reduce(loss, reduction)


def l1_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    if getattr(_lib, "ga_l1_loss", None) is None:
        raise RuntimeError("l1_loss not available in this build")
    loss = (pred - target).abs()
    return _reduce(loss, reduction)


def binary_cross_entropy(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    eps = 1e-7
    one = _as_tensor(1.0, pred.dtype)
    loss = -(target * (pred + eps).log() + (one - target) * (one - pred + eps).log())
    return _reduce(loss, reduction)


def huber_loss(pred: Tensor, target: Tensor, delta: float = 1.0, reduction: str = "mean") -> Tensor:
    diff = pred - target
    abs_diff = diff.abs()
    delta_t = _as_tensor(delta, pred.dtype)
    mask = Tensor((abs_diff.numpy() <= delta).astype(pred.dtype), dtype=pred.dtype)
    quadratic = 0.5 * (diff * diff)
    linear = delta_t * (abs_diff - 0.5 * delta_t)
    loss = quadratic * mask + linear * (1.0 - mask)
    return _reduce(loss, reduction)


def cross_entropy(logits: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    # Simple cross-entropy using softmax + NLL for one-hot/label targets
    target_arr = np.array(target.numpy(), copy=True)
    probs = logits.softmax(dim=-1)
    if target_arr.ndim == 1:
        # target contains class indices
        one_hot = np.zeros(probs.shape, dtype=probs.dtype)
        idx = target_arr.astype(int)
        one_hot[np.arange(idx.shape[0]), idx] = 1.0
        target_t = Tensor(one_hot.astype(probs.dtype), dtype=probs.dtype)
    else:
        target_t = Tensor(np.array(target_arr, dtype=probs.dtype), dtype=probs.dtype)
    log_probs = probs.log()
    loss = -(target_t * log_probs)
    return _reduce(loss, reduction)


def from_numpy(arr: np.ndarray) -> Tensor:
    """Create tensor from numpy array."""
    return Tensor(arr)


# ----------------------------------------------------------------------
# Gradient context managers
# ----------------------------------------------------------------------
class no_grad:
    """Context manager to disable gradient computation."""

    def __enter__(self):
        self.prev = _is_grad_enabled()
        _set_grad_enabled(False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _set_grad_enabled(self.prev)


class enable_grad:
    """Context manager to enable gradient computation."""

    def __enter__(self):
        self.prev = _is_grad_enabled()
        _set_grad_enabled(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _set_grad_enabled(self.prev)
