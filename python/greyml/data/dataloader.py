"""Mini-batch data loader.
Simple iterable loader that batches dataset samples for training loops.
"""

import math
import random
from typing import Iterator, List, Sequence

from ..tensor import Tensor, stack
from .dataset import Dataset


class DataLoader:
    """
    Simple Python-side DataLoader.

    Accepts any Dataset returning Tensors or numpy arrays; batches are stacked
    along the first dimension when possible.
    """

    def __init__(self, dataset: Dataset | Sequence, batch_size: int = 1, shuffle: bool = False, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        batch: List = []
        for idx in indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield _collate(batch)
                batch = []
        if batch and not self.drop_last:
            yield _collate(batch)

    def __len__(self) -> int:
        total = len(self.dataset)
        if self.drop_last:
            return total // self.batch_size
        return math.ceil(total / self.batch_size)


def _collate(items: List):
    # If items are Tensors, stack on dim=0
    if all(isinstance(x, Tensor) for x in items):
        return stack(items, dim=0)
    # Tuple/list batches -> collate elementwise
    if all(isinstance(x, (list, tuple)) for x in items):
        transposed = list(zip(*items))
        return tuple(_collate(list(part)) for part in transposed)
    return items
