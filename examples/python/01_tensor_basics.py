"""GreyML example: 01 tensor basics.
Shows a minimal usage pattern you can copy into your own experiments.
"""

import sys
sys.path.insert(0, "python")

from greyml import Tensor
import numpy as np

# Create tensors
x = Tensor([1, 2, 3, 4], dtype=np.float32)
y = Tensor([[1, 2], [3, 4]], dtype=np.float32)

print(f"Tensor x shape: {x.shape}")
print(f"Tensor y shape: {y.shape}")

# Operations
z = x + x
print(f"x + x = {z.numpy()}")