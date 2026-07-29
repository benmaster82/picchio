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
"%GCC%" -O2 -Wall -Wno-unknown-pragmas -Wno-misleading-indentation -Wno-unused-function -Wl,--stack,8388608 -o picchio.exe picchio.c -lm
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
