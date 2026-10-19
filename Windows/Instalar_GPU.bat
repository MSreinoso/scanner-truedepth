@echo off
cd /d "%~dp0"
set "BASE_PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not exist "%BASE_PY%" (
    echo Se necesita Python 3.11 de 64 bits para crear el runtime aislado.
    pause
    exit /b 1
)
if not exist "%~dp0.gpu-runtime\Scripts\python.exe" "%BASE_PY%" -m venv "%~dp0.gpu-runtime"
set "GPU_PY=%~dp0.gpu-runtime\Scripts\python.exe"
"%GPU_PY%" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto fail
"%GPU_PY%" -m pip install -r "%~dp0requirements-gpu.txt"
if errorlevel 1 goto fail
"%GPU_PY%" "%~dp0ai_alignment.py"
if errorlevel 1 goto fail
echo Runtime RTX y pesos LoFTR preparados. Usa Iniciar_Bichito3D_GPU.bat.
pause
exit /b 0
:fail
echo No se completo la instalacion. Conserva los originales y revisa el error anterior.
pause
exit /b 1
