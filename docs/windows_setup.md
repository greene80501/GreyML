<!-- GreyML doc: windows setup. Added context for collaborators. -->

# Windows Setup Guide

## Prerequisites
- Visual Studio 2022 (MSVC v143) or Clang-CL
- Python 3.10-3.13
- CMake 3.20+
- Ninja build system (recommended)

## Build Steps
1. Open "Developer Command Prompt for VS 2022"
2. Navigate to project root
3. Run: `scripts\build.bat`
4. Install Python package: `pip install -e python`

## Verify Installation
```python
from greyml import Tensor
t = Tensor([1, 2, 3])
print(t + t)