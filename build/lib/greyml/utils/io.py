"""IO utilities.
Helpers for reading/writing tensors or model snapshots on the Python side.
"""

import os
import numpy as np
from ..tensor import Tensor, _lib
from ..nn.module import Module


def save(obj, path: str):
    if isinstance(obj, Tensor):
        if not hasattr(_lib, "ga_tensor_save"):
            raise RuntimeError("ga_tensor_save not available in this build")
        status = _lib.ga_tensor_save(obj._c_ptr, path.encode())
        if status != 0:
            raise RuntimeError(f"Failed to save tensor to {path}")
        return
    if isinstance(obj, Module):
        state = _state_dict(obj)
        np.savez_compressed(path, **state)
        return
    raise NotImplementedError("Unsupported object type for save")


def load(path: str, module: Module | None = None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path, allow_pickle=False)
        if module is None:
            return {k: v for k, v in data.items()}
        _load_state_dict(module, data)
        return module
    if not hasattr(_lib, "ga_tensor_load"):
        raise RuntimeError("ga_tensor_load not available in this build")
    ptr = _lib.ga_tensor_load(path.encode())
    if not ptr:
        raise RuntimeError(f"Failed to load tensor from {path}")
    return Tensor(_c_ptr=ptr, _shape=None, _dtype=None)


def _state_dict(module: Module, prefix: str = "") -> dict[str, np.ndarray]:
    state = {}
    for name, param in module._parameters.items():
        key = f"{prefix}{name}"
        state[key] = param.numpy()
    for child_name, child in module._modules.items():
        child_prefix = f"{prefix}{child_name}."
        state.update(_state_dict(child, child_prefix))
    return state


def _load_state_dict(module: Module, data: dict):
    for name, param in module._parameters.items():
        if name not in data:
            continue
        arr = np.array(data[name], dtype=param.dtype, copy=False)
        if arr.shape != param.shape:
            raise ValueError(f"Shape mismatch for param {name}: expected {param.shape}, got {arr.shape}")
        _lib.ga_tensor_copy_from(param._c_ptr, arr.ctypes.data)
    for child_name, child in module._modules.items():
        child_prefix = f"{child_name}."
        sub_data = {k[len(child_prefix):]: v for k, v in data.items() if k.startswith(child_prefix)}
        if sub_data:
            _load_state_dict(child, sub_data)
