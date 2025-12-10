"""
Complete ctypes bindings for the greyarea C library.

This module defines all ctypes function signatures for the C API.
Import `lib` to access the raw C library.
"""

import ctypes
from pathlib import Path
import sys
import os

# =============================================================================
# Library Loading
# =============================================================================

def load_library():
    """Load greyarea.dll from repo build outputs first, then fall back."""
    dll_name = "greyarea.dll"

    repo_root = Path(__file__).resolve().parents[2]
    search_paths = [
        repo_root / "build" / "Release" / dll_name,  # CMake multi-config
        repo_root / "build" / "release" / dll_name,  # lowercase variant
        Path(__file__).parent / dll_name,             # packaged copy (fallback)
        Path(sys.prefix) / "Library" / "bin" / dll_name,
        Path(sys.prefix) / "DLLs" / dll_name,
    ]

    # Add PATH environment variable
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
        f"Could not find {dll_name}. Please build the library first.\n"
        f"Run: scripts\\build.bat\n"
        f"Searched paths: {[str(p) for p in search_paths]}"
    )


lib = load_library()

# =============================================================================
# Type Definitions
# =============================================================================

# Common C types for readability
c_tensor_ptr = ctypes.c_void_p
c_node_ptr = ctypes.c_void_p
c_module_ptr = ctypes.c_void_p
c_optim_ptr = ctypes.c_void_p

c_dtype = ctypes.c_int
c_bool = ctypes.c_bool
c_int = ctypes.c_int
c_int64 = ctypes.c_int64
c_size_t = ctypes.c_size_t
c_float = ctypes.c_float
c_double = ctypes.c_double
c_void_p = ctypes.c_void_p
c_char_p = ctypes.c_char_p

c_int64_p = ctypes.POINTER(c_int64)
c_float_p = ctypes.POINTER(c_float)
c_tensor_pp = ctypes.POINTER(c_tensor_ptr)

# =============================================================================
# Core Tensor API (ga_tensor.h)
# =============================================================================

# Creation functions
lib.ga_tensor_empty.restype = c_tensor_ptr
lib.ga_tensor_empty.argtypes = [c_int, c_int64_p, c_dtype]

lib.ga_tensor_zeros.restype = c_tensor_ptr
lib.ga_tensor_zeros.argtypes = [c_int, c_int64_p, c_dtype]

lib.ga_tensor_ones.restype = c_tensor_ptr
lib.ga_tensor_ones.argtypes = [c_int, c_int64_p, c_dtype]

lib.ga_tensor_full.restype = c_tensor_ptr
lib.ga_tensor_full.argtypes = [c_int, c_int64_p, c_dtype, c_void_p]

lib.ga_tensor_from_data.restype = c_tensor_ptr
lib.ga_tensor_from_data.argtypes = [c_int, c_int64_p, c_dtype, c_void_p]

lib.ga_tensor_arange.restype = c_tensor_ptr
lib.ga_tensor_arange.argtypes = [c_int64, c_int64, c_int64, c_dtype]

lib.ga_tensor_linspace.restype = c_tensor_ptr
lib.ga_tensor_linspace.argtypes = [c_float, c_float, c_int64]

lib.ga_tensor_eye.restype = c_tensor_ptr
lib.ga_tensor_eye.argtypes = [c_int64, c_int64, c_dtype]

lib.ga_tensor_rand.restype = c_tensor_ptr
lib.ga_tensor_rand.argtypes = [c_int, c_int64_p]

lib.ga_tensor_randn.restype = c_tensor_ptr
lib.ga_tensor_randn.argtypes = [c_int, c_int64_p]

# Memory management
lib.ga_tensor_retain.restype = None
lib.ga_tensor_retain.argtypes = [c_tensor_ptr]

lib.ga_tensor_release.restype = None
lib.ga_tensor_release.argtypes = [c_tensor_ptr]

lib.ga_tensor_clone.restype = c_tensor_ptr
lib.ga_tensor_clone.argtypes = [c_tensor_ptr]

lib.ga_tensor_contiguous.restype = c_tensor_ptr
lib.ga_tensor_contiguous.argtypes = [c_tensor_ptr]

lib.ga_tensor_detach.restype = c_tensor_ptr
lib.ga_tensor_detach.argtypes = [c_tensor_ptr]

# Shape manipulation
lib.ga_tensor_reshape.restype = c_tensor_ptr
lib.ga_tensor_reshape.argtypes = [c_tensor_ptr, c_int, c_int64_p]

lib.ga_tensor_flatten.restype = c_tensor_ptr
lib.ga_tensor_flatten.argtypes = [c_tensor_ptr, c_int, c_int]

lib.ga_tensor_unsqueeze.restype = c_tensor_ptr
lib.ga_tensor_unsqueeze.argtypes = [c_tensor_ptr, c_int]

lib.ga_tensor_squeeze.restype = c_tensor_ptr
lib.ga_tensor_squeeze.argtypes = [c_tensor_ptr, c_int]

lib.ga_tensor_transpose.restype = c_tensor_ptr
lib.ga_tensor_transpose.argtypes = [c_tensor_ptr, c_int, c_int]

lib.ga_tensor_permute.restype = c_tensor_ptr
lib.ga_tensor_permute.argtypes = [c_tensor_ptr, ctypes.POINTER(c_int)]

lib.ga_tensor_expand.restype = c_tensor_ptr
lib.ga_tensor_expand.argtypes = [c_tensor_ptr, c_int, c_int64_p]

# Indexing
lib.ga_tensor_get.restype = c_tensor_ptr
lib.ga_tensor_get.argtypes = [c_tensor_ptr, c_int64_p]

lib.ga_tensor_set.restype = None
lib.ga_tensor_set.argtypes = [c_tensor_ptr, c_int64_p, c_void_p]

lib.ga_tensor_slice.restype = c_tensor_ptr
lib.ga_tensor_slice.argtypes = [c_tensor_ptr, c_int64_p, c_int64_p, c_int64_p]

lib.ga_tensor_select.restype = c_tensor_ptr
lib.ga_tensor_select.argtypes = [c_tensor_ptr, c_int, c_int64]

lib.ga_tensor_index.restype = c_tensor_ptr
lib.ga_tensor_index.argtypes = [c_tensor_ptr, c_tensor_ptr]

# Properties
lib.ga_tensor_is_contiguous.restype = c_bool
lib.ga_tensor_is_contiguous.argtypes = [c_tensor_ptr]

lib.ga_tensor_is_view.restype = c_bool
lib.ga_tensor_is_view.argtypes = [c_tensor_ptr]

lib.ga_tensor_same_shape.restype = c_bool
lib.ga_tensor_same_shape.argtypes = [c_tensor_ptr, c_tensor_ptr]

lib.ga_tensor_broadcastable.restype = c_bool
lib.ga_tensor_broadcastable.argtypes = [c_tensor_ptr, c_tensor_ptr]

# Data access
lib.ga_tensor_data.restype = c_void_p
lib.ga_tensor_data.argtypes = [c_tensor_ptr]

lib.ga_tensor_data_f32.restype = c_float_p
lib.ga_tensor_data_f32.argtypes = [c_tensor_ptr]

lib.ga_tensor_data_f64.restype = ctypes.POINTER(c_double)
lib.ga_tensor_data_f64.argtypes = [c_tensor_ptr]

lib.ga_tensor_copy_to.restype = None
lib.ga_tensor_copy_to.argtypes = [c_tensor_ptr, c_void_p]

lib.ga_tensor_copy_from.restype = None
lib.ga_tensor_copy_from.argtypes = [c_tensor_ptr, c_void_p]

lib.ga_tensor_fill.restype = None
lib.ga_tensor_fill.argtypes = [c_tensor_ptr, c_void_p]

# Accessors
lib.ga_tensor_ndim.restype = c_int
lib.ga_tensor_ndim.argtypes = [c_tensor_ptr]

lib.ga_tensor_get_shape.restype = None
lib.ga_tensor_get_shape.argtypes = [c_tensor_ptr, c_int64_p]

lib.ga_tensor_set_requires_grad.restype = None
lib.ga_tensor_set_requires_grad.argtypes = [c_tensor_ptr, c_bool]

lib.ga_tensor_get_grad.restype = c_tensor_ptr
lib.ga_tensor_get_grad.argtypes = [c_tensor_ptr]

lib.ga_tensor_zero_grad.restype = None
lib.ga_tensor_zero_grad.argtypes = [c_tensor_ptr]

# =============================================================================
# Operations API (ga_ops.h)
# =============================================================================

# Binary operations
lib.ga_add.restype = c_tensor_ptr
lib.ga_add.argtypes = [c_tensor_ptr, c_tensor_ptr]

lib.ga_sub.restype = c_tensor_ptr
lib.ga_sub.argtypes = [c_tensor_ptr, c_tensor_ptr]

lib.ga_mul.restype = c_tensor_ptr
lib.ga_mul.argtypes = [c_tensor_ptr, c_tensor_ptr]

lib.ga_div.restype = c_tensor_ptr
lib.ga_div.argtypes = [c_tensor_ptr, c_tensor_ptr]

lib.ga_pow.restype = c_tensor_ptr
lib.ga_pow.argtypes = [c_tensor_ptr, c_tensor_ptr]

lib.ga_add_scalar.restype = c_tensor_ptr
lib.ga_add_scalar.argtypes = [c_tensor_ptr, c_float]

lib.ga_mul_scalar.restype = c_tensor_ptr
lib.ga_mul_scalar.argtypes = [c_tensor_ptr, c_float]

# Unary operations
lib.ga_neg.restype = c_tensor_ptr
lib.ga_neg.argtypes = [c_tensor_ptr]

lib.ga_exp.restype = c_tensor_ptr
lib.ga_exp.argtypes = [c_tensor_ptr]

lib.ga_log.restype = c_tensor_ptr
lib.ga_log.argtypes = [c_tensor_ptr]

lib.ga_sqrt.restype = c_tensor_ptr
lib.ga_sqrt.argtypes = [c_tensor_ptr]

lib.ga_abs.restype = c_tensor_ptr
lib.ga_abs.argtypes = [c_tensor_ptr]

lib.ga_square.restype = c_tensor_ptr
lib.ga_square.argtypes = [c_tensor_ptr]

lib.ga_sin.restype = c_tensor_ptr
lib.ga_sin.argtypes = [c_tensor_ptr]

lib.ga_cos.restype = c_tensor_ptr
lib.ga_cos.argtypes = [c_tensor_ptr]

lib.ga_tanh.restype = c_tensor_ptr
lib.ga_tanh.argtypes = [c_tensor_ptr]

lib.ga_sigmoid.restype = c_tensor_ptr
lib.ga_sigmoid.argtypes = [c_tensor_ptr]

lib.ga_relu.restype = c_tensor_ptr
lib.ga_relu.argtypes = [c_tensor_ptr]

lib.ga_leaky_relu.restype = c_tensor_ptr
lib.ga_leaky_relu.argtypes = [c_tensor_ptr, c_float]

lib.ga_gelu.restype = c_tensor_ptr
lib.ga_gelu.argtypes = [c_tensor_ptr]

lib.ga_silu.restype = c_tensor_ptr
lib.ga_silu.argtypes = [c_tensor_ptr]

lib.ga_softmax.restype = c_tensor_ptr
lib.ga_softmax.argtypes = [c_tensor_ptr, c_int]

lib.ga_log_softmax.restype = c_tensor_ptr
lib.ga_log_softmax.argtypes = [c_tensor_ptr, c_int]

# Reduction operations
lib.ga_sum.restype = c_tensor_ptr
lib.ga_sum.argtypes = [c_tensor_ptr, c_int, c_bool]

lib.ga_mean.restype = c_tensor_ptr
lib.ga_mean.argtypes = [c_tensor_ptr, c_int, c_bool]

lib.ga_var.restype = c_tensor_ptr
lib.ga_var.argtypes = [c_tensor_ptr, c_int, c_bool, c_bool]

lib.ga_max.restype = c_tensor_ptr
lib.ga_max.argtypes = [c_tensor_ptr, c_int, c_bool]

lib.ga_min.restype = c_tensor_ptr
lib.ga_min.argtypes = [c_tensor_ptr, c_int, c_bool]

lib.ga_argmax.restype = c_tensor_ptr
lib.ga_argmax.argtypes = [c_tensor_ptr, c_int]

# Matrix operations
lib.ga_matmul.restype = c_tensor_ptr
lib.ga_matmul.argtypes = [c_tensor_ptr, c_tensor_ptr]

lib.ga_bmm.restype = c_tensor_ptr
lib.ga_bmm.argtypes = [c_tensor_ptr, c_tensor_ptr]

lib.ga_dot.restype = c_tensor_ptr
lib.ga_dot.argtypes = [c_tensor_ptr, c_tensor_ptr]

lib.ga_outer.restype = c_tensor_ptr
lib.ga_outer.argtypes = [c_tensor_ptr, c_tensor_ptr]

# Convolution operations
lib.ga_conv1d.restype = c_tensor_ptr
lib.ga_conv1d.argtypes = [c_tensor_ptr, c_tensor_ptr, c_tensor_ptr, c_int, c_int, c_int, c_int]

lib.ga_conv2d.restype = c_tensor_ptr
lib.ga_conv2d.argtypes = [c_tensor_ptr, c_tensor_ptr, c_tensor_ptr, c_int, c_int, c_int, c_int]

lib.ga_conv_transpose2d.restype = c_tensor_ptr
lib.ga_conv_transpose2d.argtypes = [c_tensor_ptr, c_tensor_ptr, c_tensor_ptr, c_int, c_int, c_int, c_int, c_int]

# Pooling operations
lib.ga_max_pool2d.restype = c_tensor_ptr
lib.ga_max_pool2d.argtypes = [c_tensor_ptr, c_int, c_int, c_int, c_int]

lib.ga_avg_pool2d.restype = c_tensor_ptr
lib.ga_avg_pool2d.argtypes = [c_tensor_ptr, c_int, c_int, c_int]

lib.ga_adaptive_avg_pool2d.restype = c_tensor_ptr
lib.ga_adaptive_avg_pool2d.argtypes = [c_tensor_ptr, c_int, c_int]

# Transform operations
lib.ga_cat.restype = c_tensor_ptr
lib.ga_cat.argtypes = [c_tensor_pp, c_int, c_int]

lib.ga_stack.restype = c_tensor_ptr
lib.ga_stack.argtypes = [c_tensor_pp, c_int, c_int]

lib.ga_split.restype = None
lib.ga_split.argtypes = [c_tensor_ptr, c_int, c_int, c_tensor_pp]

lib.ga_gather.restype = c_tensor_ptr
lib.ga_gather.argtypes = [c_tensor_ptr, c_int, c_tensor_ptr]

lib.ga_scatter.restype = c_tensor_ptr
lib.ga_scatter.argtypes = [c_tensor_ptr, c_int, c_tensor_ptr, c_tensor_ptr]

lib.ga_where.restype = c_tensor_ptr
lib.ga_where.argtypes = [c_tensor_ptr, c_tensor_ptr, c_tensor_ptr]

# Additional transform ops
lib.ga_transpose.restype = c_tensor_ptr
lib.ga_transpose.argtypes = [c_tensor_ptr]

lib.ga_reshape.restype = c_tensor_ptr
lib.ga_reshape.argtypes = [c_tensor_ptr, c_int, c_int64_p]

lib.ga_flatten.restype = c_tensor_ptr
lib.ga_flatten.argtypes = [c_tensor_ptr]

# =============================================================================
# Autograd API (ga_autograd.h)
# =============================================================================

lib.ga_set_grad_enabled.restype = None
lib.ga_set_grad_enabled.argtypes = [c_bool]

lib.ga_is_grad_enabled.restype = c_bool
lib.ga_is_grad_enabled.argtypes = []

lib.ga_backward.restype = None
lib.ga_backward.argtypes = [c_tensor_ptr, c_tensor_ptr]

lib.ga_node_create.restype = c_node_ptr
lib.ga_node_create.argtypes = [c_char_p, c_int, c_int, c_int, c_int]

lib.ga_node_save_tensor.restype = None
lib.ga_node_save_tensor.argtypes = [c_node_ptr, c_int, c_tensor_ptr]

lib.ga_node_save_scalar.restype = None
lib.ga_node_save_scalar.argtypes = [c_node_ptr, c_int, c_float]

lib.ga_node_save_int.restype = None
lib.ga_node_save_int.argtypes = [c_node_ptr, c_int, c_int]

lib.ga_accumulate_grad.restype = None
lib.ga_accumulate_grad.argtypes = [c_tensor_ptr, c_tensor_ptr]

lib.ga_detach.restype = None
lib.ga_detach.argtypes = [c_tensor_ptr]

lib.ga_no_grad.restype = None
lib.ga_no_grad.argtypes = []

lib.ga_enable_grad.restype = None
lib.ga_enable_grad.argtypes = []

# =============================================================================
# Neural Network API (ga_nn.h)
# =============================================================================

# Linear layer
lib.ga_linear_create.restype = c_module_ptr
lib.ga_linear_create.argtypes = [c_int, c_int, c_bool]

lib.ga_linear_free.restype = None
lib.ga_linear_free.argtypes = [c_module_ptr]

lib.ga_linear_forward.restype = c_tensor_ptr
lib.ga_linear_forward.argtypes = [c_module_ptr, c_tensor_ptr]

# Conv2D layer
lib.ga_conv2d_create.restype = c_module_ptr
lib.ga_conv2d_create.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_bool]

lib.ga_conv2d_free.restype = None
lib.ga_conv2d_free.argtypes = [c_module_ptr]

lib.ga_conv2d_forward.restype = c_tensor_ptr
lib.ga_conv2d_forward.argtypes = [c_module_ptr, c_tensor_ptr]

# BatchNorm2D layer
lib.ga_batchnorm2d_create.restype = c_module_ptr
lib.ga_batchnorm2d_create.argtypes = [c_int, c_float, c_float]

lib.ga_batchnorm2d_free.restype = None
lib.ga_batchnorm2d_free.argtypes = [c_module_ptr]

lib.ga_batchnorm2d_forward.restype = c_tensor_ptr
lib.ga_batchnorm2d_forward.argtypes = [c_module_ptr, c_tensor_ptr]

# Dropout layer
lib.ga_dropout_create.restype = c_module_ptr
lib.ga_dropout_create.argtypes = [c_float]

lib.ga_dropout_free.restype = None
lib.ga_dropout_free.argtypes = [c_module_ptr]

lib.ga_dropout_forward.restype = c_tensor_ptr
lib.ga_dropout_forward.argtypes = [c_module_ptr, c_tensor_ptr]

# Pooling layers
lib.ga_maxpool2d_create.restype = c_module_ptr
lib.ga_maxpool2d_create.argtypes = [c_int, c_int, c_int, c_int]

lib.ga_avgpool2d_create.restype = c_module_ptr
lib.ga_avgpool2d_create.argtypes = [c_int, c_int, c_int]

lib.ga_pool2d_free.restype = None
lib.ga_pool2d_free.argtypes = [c_module_ptr]

lib.ga_pool2d_forward.restype = c_tensor_ptr
lib.ga_pool2d_forward.argtypes = [c_module_ptr, c_tensor_ptr]

# Embedding layer
lib.ga_embedding_create.restype = c_module_ptr
lib.ga_embedding_create.argtypes = [c_int, c_int]

lib.ga_embedding_free.restype = None
lib.ga_embedding_free.argtypes = [c_module_ptr]

lib.ga_embedding_forward.restype = c_tensor_ptr
lib.ga_embedding_forward.argtypes = [c_module_ptr, c_tensor_ptr]

# RNN layer
lib.ga_rnn_create.restype = c_module_ptr
lib.ga_rnn_create.argtypes = [c_int, c_int]

lib.ga_rnn_free.restype = None
lib.ga_rnn_free.argtypes = [c_module_ptr]

lib.ga_rnn_forward.restype = c_tensor_ptr
lib.ga_rnn_forward.argtypes = [c_module_ptr, c_tensor_ptr, c_tensor_pp]

# Multi-head attention
lib.ga_mha_create.restype = c_module_ptr
lib.ga_mha_create.argtypes = [c_int, c_int]

lib.ga_mha_free.restype = None
lib.ga_mha_free.argtypes = [c_module_ptr]

lib.ga_mha_forward.restype = c_tensor_ptr
lib.ga_mha_forward.argtypes = [c_module_ptr, c_tensor_ptr, c_tensor_ptr, c_tensor_ptr, c_tensor_pp]

# Sequential container
lib.ga_sequential_create.restype = c_module_ptr
lib.ga_sequential_create.argtypes = [ctypes.POINTER(c_module_ptr), c_size_t]

lib.ga_sequential_free.restype = None
lib.ga_sequential_free.argtypes = [c_module_ptr]

lib.ga_sequential_forward.restype = c_tensor_ptr
lib.ga_sequential_forward.argtypes = [c_module_ptr, c_tensor_ptr]

# Weight initialization
lib.ga_init_uniform.restype = None
lib.ga_init_uniform.argtypes = [c_tensor_ptr, c_float, c_float]

lib.ga_init_normal.restype = None
lib.ga_init_normal.argtypes = [c_tensor_ptr, c_float, c_float]

lib.ga_init_xavier_uniform.restype = None
lib.ga_init_xavier_uniform.argtypes = [c_tensor_ptr]

lib.ga_init_kaiming_uniform.restype = None
lib.ga_init_kaiming_uniform.argtypes = [c_tensor_ptr]

# =============================================================================
# Random Number Generation API (ga_random.h)
# =============================================================================

lib.ga_random_seed.restype = None
lib.ga_random_seed.argtypes = [ctypes.c_uint64]

lib.ga_random_uint32.restype = ctypes.c_uint32
lib.ga_random_uint32.argtypes = []

lib.ga_random_float.restype = c_float
lib.ga_random_float.argtypes = []

lib.ga_random_normal.restype = c_float
lib.ga_random_normal.argtypes = []

lib.ga_random_shuffle.restype = None
lib.ga_random_shuffle.argtypes = [ctypes.POINTER(c_int), c_int]

lib.ga_tensor_rand_.restype = None
lib.ga_tensor_rand_.argtypes = [c_tensor_ptr]

lib.ga_tensor_randn_.restype = None
lib.ga_tensor_randn_.argtypes = [c_tensor_ptr]

# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'lib',
    'c_tensor_ptr',
    'c_node_ptr',
    'c_module_ptr',
    'c_optim_ptr',
    'c_dtype',
    'c_bool',
    'c_int',
    'c_int64',
    'c_float',
    'c_void_p',
]
