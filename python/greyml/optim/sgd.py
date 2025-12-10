"""SGD optimizer.
Implements plain/with-momentum SGD updates for GreyML models.
"""

from typing import List
import numpy as np
from .optimizer import Optimizer
from ..tensor import Tensor, no_grad


class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr: float, momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros(p.shape, dtype=p.dtype) for p in params]

    def step(self):
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
