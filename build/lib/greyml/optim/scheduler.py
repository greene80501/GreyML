"""Learning rate scheduler.
Manages LR schedules on top of optimizer instances.
"""

class StepLR:
    """Minimal StepLR scheduler compatible with Optimizer step() usage."""

    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma


__all__ = ["StepLR"]
