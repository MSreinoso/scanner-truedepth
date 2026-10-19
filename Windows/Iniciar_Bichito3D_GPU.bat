@echo off
cd /d "%~dp0"
if not exist "%~dp0.gpu-runtime\Scripts\pythonw.exe" (
    echo Falta el runtime CUDA. Ejecuta Instalar_GPU.bat primero.
    pause
    exit /b 1
)
start "" "%~dp0.gpu-runtime\Scripts\pythonw.exe" "%~dp0Bichito3D_ReconstructionEngine.py"
