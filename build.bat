@echo off
REM build.bat — Compila Picchio su Windows con MSYS2/MinGW

set GCC=C:\msys64\mingw64\bin\gcc.exe
set PATH=C:\msys64\mingw64\bin;C:\msys64\usr\bin;%PATH%

if not exist "%GCC%" (
    echo GCC non trovato in %GCC%
    echo Installa: pacman -S mingw-w64-x86_64-gcc
    exit /b 1
)

echo Compilo picchio.exe...
REM -fopenmp e' necessario: senza di esso i #pragma omp dei kernel matmul in quant.h
REM vengono ignorati e tutto il calcolo resta su un solo core.
REM -mavx2 -mfma attivano i percorsi SIMD di quant.h, protetti da #ifdef __AVX2__:
REM senza questi flag i kernel usano solo lo scalare. Richiede una CPU con AVX2.
"%GCC%" -O2 -Wall -fopenmp -mavx2 -mfma -Wno-misleading-indentation -Wno-unused-function -Wl,--stack,8388608 -o picchio.exe picchio.c -lm -lpsapi
if %ERRORLEVEL% NEQ 0 (
    echo Errore di compilazione.
    exit /b 1
)

echo.
echo === picchio.exe compilato ===
echo.
echo Test:     picchio.exe --self-test
echo Modello:  picchio.exe test_model
echo Reale:    set MODEL=D:\gptoss_i4 ^& picchio.exe
