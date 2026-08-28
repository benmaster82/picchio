@echo off
REM build_cuda.bat - build the OPTIONAL CUDA backend for Picchio on Windows.
REM
REM Produces picchio_cuda.dll, loaded at runtime by picchio.exe when GPU=1.
REM The engine itself never links against CUDA: without this DLL (or without a
REM CUDA device) picchio runs the pure-C CPU path unchanged. So this build is
REM entirely optional.
REM
REM Requirements:
REM   - NVIDIA CUDA Toolkit (nvcc) on PATH.
REM   - MSVC (Visual Studio Build Tools) as nvcc's host compiler.
REM   - An NVIDIA GPU. Set the arch to match yours (default sm_75 = Turing,
REM     e.g. GTX 1650 / RTX 20xx). RTX 30xx = sm_86, RTX 40xx = sm_89.
REM
REM Usage:  build_cuda.bat            (builds the DLL)
REM         build_cuda.bat test       (also builds+runs the numeric self-test)

setlocal
if "%CUDA_ARCH%"=="" set CUDA_ARCH=sm_75

REM Bring MSVC (cl.exe) into the environment for nvcc's host compiler.
set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if exist "%VCVARS%" call "%VCVARS%" >nul

where nvcc >nul 2>&1
if errorlevel 1 (
    echo nvcc not found. Install the CUDA Toolkit and ensure nvcc is on PATH.
    exit /b 1
)

echo Building picchio_cuda.dll (arch %CUDA_ARCH%, static cudart)...
nvcc -O3 -arch=%CUDA_ARCH% --cudart static -DPGPU_BUILD_DLL -shared -o picchio_cuda.dll picchio_cuda.cu
if errorlevel 1 ( echo Build error. & exit /b 1 )
echo === picchio_cuda.dll built ===

if /i "%1"=="test" (
    echo Building numeric self-test...
    nvcc -O3 -arch=%CUDA_ARCH% -DPGPU_TEST -o pgpu_test.exe picchio_cuda.cu
    if errorlevel 1 ( echo Test build error. & exit /b 1 )
    echo === running pgpu_test.exe ===
    pgpu_test.exe
)

echo.
echo Done. Run the engine with GPU=1 to use it, e.g.:
echo   set GPU=1 ^& picchio.exe C:\models\gptoss20b_i8h
endlocal
