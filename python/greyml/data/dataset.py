"""Dataset base class.
Defines the interface datasets must follow inside GreyML.
"""

from typing import Any, Iterator


class Dataset:
    """Minimal dataset interface."""

    def __len__(self) -> int:  # pragma: no cover - simple abstract
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Any:  # pragma: no cover - simple abstract
        raise NotImplementedError

    def __iter__(self) -> Iterator[Any]:
        for i in range(len(self)):
            yield self[i]
