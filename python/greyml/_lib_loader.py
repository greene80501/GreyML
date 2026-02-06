"""Shared library loader.
Locates and loads the GreyML C backend across platforms.
"""

import ctypes
import os
import sys
from pathlib import Path

def load_library():
    dll_name = "greyarea.dll"

    # Prefer freshly built artifacts first (CI/build tree), then fall back.
    repo_root = Path(__file__).resolve().parents[2]
    search_paths = [
        repo_root / "build" / "Release" / dll_name,
        repo_root / "build" / "release" / dll_name,
        Path(__file__).parent / dll_name,
        Path(sys.prefix) / "Library" / "bin" / dll_name,
        Path(sys.prefix) / "DLLs" / dll_name,
    ]

    # Add PATH entries
    for path in os.environ.get("PATH", "").split(os.pathsep):
        if path:
            search_paths.append(Path(path) / dll_name)

    for path in search_paths:
        if path.exists():
            try:
                return ctypes.CDLL(str(path))
            except OSError as e:
                print(f"Failed to load {path}: {e}")
                continue
    
    raise ImportError(f"Could not find {dll_name} in any search path")

# Add ctypes function signatures
def _setup_signatures(lib):
    lib.ga_tensor_empty.restype = ctypes.c_void_p
    lib.ga_tensor_empty.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int]
    
    lib.ga_tensor_release.restype = None
    lib.ga_tensor_release.argtypes = [ctypes.c_void_p]
    
    lib.ga_add.restype = ctypes.c_void_p
    lib.ga_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    
    lib.ga_mul.restype = ctypes.c_void_p
    lib.ga_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    
    lib.ga_matmul.restype = ctypes.c_void_p
    lib.ga_matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    
    lib.ga_sum.restype = ctypes.c_void_p
    lib.ga_sum.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_bool]
    
    lib.ga_backward.restype = None
    lib.ga_backward.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    
    lib.ga_detach.restype = None
    lib.ga_detach.argtypes = [ctypes.c_void_p]
    
    lib.ga_tensor_copy_from.restype = None
    lib.ga_tensor_copy_from.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    
    lib.ga_tensor_copy_to.restype = None
    lib.ga_tensor_copy_to.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    
    lib.ga_tensor_ndim.restype = ctypes.c_int
    lib.ga_tensor_ndim.argtypes = [ctypes.c_void_p]
    
    lib.ga_tensor_get_shape.restype = None
    lib.ga_tensor_get_shape.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)]
    
    lib.ga_tensor_get_grad.restype = ctypes.c_void_p
    lib.ga_tensor_get_grad.argtypes = [ctypes.c_void_p]
    
    lib.ga_tensor_set_requires_grad.restype = None
    lib.ga_tensor_set_requires_grad.argtypes = [ctypes.c_void_p, ctypes.c_bool]
