@echo off
set GCC=C:\msys64\mingw64\bin\gcc.exe
set PATH=C:\msys64\mingw64\bin;C:\msys64\usr\bin;%PATH%
echo Building minimax_forward_check.exe...
"%GCC%" -O2 -Wall -fopenmp -mavx2 -mfma -Wno-misleading-indentation -Wno-unused-function -static -o minimax_forward_check.exe minimax_forward_check.c -lm
if %ERRORLEVEL% NEQ 0 (
    echo Build error.
    exit /b 1
)
echo === minimax_forward_check.exe built ===
