"""Base Module class for GreyML.
Defines a lightweight parameter container and training/eval switching.
"""

from typing import List, Dict, Any
from ..tensor import Tensor

class Module:
    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def parameters(self) -> List[Tensor]:
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params
    
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()
    
    def train(self, mode: bool = True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
    
    def eval(self):
        self.train(False)
    
    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor) and name in ['weight', 'bias']:
            self._parameters[name] = value
        super().__setattr__(name, value)
