"""Adam optimizer.
Python interface for Adam parameter updates on GreyML tensors.
"""

from typing import List
import numpy as np
from .optimizer import Optimizer
from ..tensor import Tensor, no_grad


class Adam(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.001, betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros(p.shape, dtype=p.dtype) for p in params]
        self.v = [np.zeros(p.shape, dtype=p.dtype) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        with no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue

                grad_arr = param.grad.numpy()
                param_arr = param.numpy()

                if self.weight_decay != 0:
                    grad_arr = grad_arr + self.weight_decay * param_arr

                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad_arr
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad_arr * grad_arr)

                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                param.data = param_arr - update
