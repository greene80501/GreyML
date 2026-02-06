@echo off
REM Build script for Windows (Visual Studio BuildTools 2022)

setlocal enabledelayedexpansion

REM Find vcvars64.bat
set VCVARS_PATH="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if not exist !VCVARS_PATH! (
    echo ERROR: Build Tools not found at !VCVARS_PATH!
    echo Please install Visual Studio Build Tools 2022 with C++ workload
    exit /b 1
)

call !VCVARS_PATH!
if errorlevel 1 exit /b 1

REM Clean then build
if exist build\release rd /s /q build\release

echo Configuring CMake...
cmake --preset windows-release -DCMAKE_BUILD_TYPE=Release

echo Building...
cmake --build build/release --config Release --parallel

REM Verify and copy DLL
if exist build\release\greyarea.dll (
    copy build\release\greyarea.dll python\greyml\
    echo SUCCESS: Build complete!
) else (
    echo ERROR: DLL not found
    exit /b 1
)
