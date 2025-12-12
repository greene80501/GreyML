"""Multi-head attention glue code.
Provides Python-side convenience around the attention kernels.
"""

from .layers import MultiheadAttention

__all__ = ["MultiheadAttention"]
