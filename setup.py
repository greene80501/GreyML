"""Package metadata entry point.
Defines how the Python package is built and installed for GreyML.
"""

from setuptools import setup, find_packages
from pathlib import Path

root = Path(__file__).parent

setup(
    name="greyml",
    version="0.1.0",
    description="GreyArea Labs AI Library (Python bindings)",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=["numpy>=1.24"],
)
