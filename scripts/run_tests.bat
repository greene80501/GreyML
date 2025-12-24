@echo off
setlocal

echo Running Python tests...
python -m pytest -q tests/python
if errorlevel 1 exit /b 1

if exist build\release (
    echo (Optional) Run C tests with: ctest --test-dir build\release
) else (
    echo (C tests require a CMake build directory; skipping.)
)

endlocal
