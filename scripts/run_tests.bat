@echo off
setlocal
echo Running Python tests...
python -m pytest -q tests/python
echo (C tests require CMake build; run ctest in your build directory.)
endlocal
