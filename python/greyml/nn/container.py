"""Sequential/container utilities.
Simple module containers to compose layers in Python.
"""

from typing import List
from .module import Module

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self._module_list = list(modules)
        for i, module in enumerate(modules):
            self._modules[str(i)] = module
    
    def forward(self, x):
        for module in self._module_list:
            x = module(x)
        return x

class ModuleList(Module):
    def __init__(self, modules: List[Module] = None):
        super().__init__()
        self._module_list = modules or []
    
    def __getitem__(self, idx):
        return self._module_list[idx]
    
    def __setitem__(self, idx, module):
        self._module_list[idx] = module