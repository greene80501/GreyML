"""Pooling layers.
Max/average pooling wrappers over the C kernels.
"""

from .layers import MaxPool2d, AvgPool2d, AdaptiveAvgPool2d

__all__ = ["MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d"]
