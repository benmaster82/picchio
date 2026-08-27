@echo off
REM build.bat - build Picchio on Windows with MSYS2/MinGW

set GCC=C:\msys64\mingw64\bin\gcc.exe
set PATH=C:\msys64\mingw64\bin;C:\msys64\usr\bin;%PATH%

if not exist "%GCC%" (
    echo GCC not found at %GCC%
    echo Install: pacman -S mingw-w64-x86_64-gcc
    exit /b 1
)

echo Building picchio.exe...
REM -fopenmp is required: without it the #pragma omp directives of the matmul
REM kernels in quant.h are ignored and all computation stays on a single core.
REM -mavx2 -mfma enable the SIMD paths of quant.h, guarded by #ifdef __AVX2__:
REM without these flags the kernels use only scalar code. Requires a CPU with AVX2.
REM -static produces a self-contained executable: it does not need libgomp-1.dll
REM or libwinpthread-1.dll at runtime, so picchio.exe runs even without MinGW on PATH.
REM -lws2_32: Winsock, for the distributed pipeline mode (TCP stages).
"%GCC%" -O2 -Wall -fopenmp -mavx2 -mfma -Wno-misleading-indentation -Wno-unused-function -static -Wl,--stack,8388608 -o picchio.exe picchio.c -lm -lpsapi -lws2_32
if %ERRORLEVEL% NEQ 0 (
    echo Build error.
    exit /b 1
)

echo.
echo === picchio.exe built ===
echo.
echo Test:   picchio.exe --self-test
echo Model:  picchio.exe test_model
echo Real:   set MODEL=C:\models\gptoss_i4 ^& picchio.exe
