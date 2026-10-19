@echo off
cd /d "%~dp0"
set "TRAIN_PY=%~dp0.gpu-runtime\Scripts\python.exe"
if not exist "%TRAIN_PY%" (
    echo Ejecuta Instalar_GPU.bat primero.
    pause
    exit /b 1
)
"%TRAIN_PY%" "%~dp0download_nyu.py"
if errorlevel 1 goto fail
"%TRAIN_PY%" "%~dp0train_depth.py" --steps 2400
if errorlevel 1 goto fail
echo Entrenamiento y evaluacion terminados. El informe esta en models\depth_denoiser_v1.json.
pause
exit /b 0
:fail
echo El proceso fallo. Revisa el error; no se ha validado un nuevo modelo.
pause
exit /b 1
