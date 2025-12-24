"""Pytest configuration for GreyML.

Ensures the local python package is importable and seeds randomness for stable assertions.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


@pytest.fixture(autouse=True)
def _reset_seed():
    """Reset NumPy RNG before each test for reproducibility."""
    np.random.seed(0)
