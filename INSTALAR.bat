@echo off
rem =====================================================================================================
rem  INSTALAR.bat - crea el entorno de Python (.venv) e instala todo lo necesario. Doble clic y esperar.
rem  - Requiere Python 3.14 (https://www.python.org/downloads/, marcar "Add python.exe to PATH").
rem  - Si la PC tiene placa NVIDIA instala PyTorch con CUDA (GPU); si no, la version para CPU.
rem  - Se puede volver a correr: si .venv ya existe, solo completa o actualiza lo que falte.
rem  - Descarga ~3 GB (con GPU) y ocupa ~5 GB. Tarda de 5 a 20 minutos segun la conexion.
rem =====================================================================================================
setlocal
cd /d "%~dp0"
echo.
echo ===== Instalacion del identificador de particulas (Skipper-CCD) =====
echo.

rem ---- 1. Python 3.14 -----------------------------------------------------------------------------------
set "PY="
py -3.14 -c "import sys" >nul 2>&1 && set "PY=py -3.14"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 14) else 1)" >nul 2>&1 && set "PY=python"
if not defined PY (
    echo [ERROR] No se encontro Python 3.14.
    echo         Instalalo desde https://www.python.org/downloads/ ^(marcar "Add python.exe to PATH"^)
    echo         y volve a correr este archivo.
    start "" https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [1/5] Python 3.14 encontrado ^(%PY%^).

rem ---- 2. Entorno virtual .venv -------------------------------------------------------------------------
if exist ".venv\Scripts\python.exe" (
    echo [2/5] El entorno .venv ya existe: se completa lo que falte.
) else (
    echo [2/5] Creando el entorno .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear .venv.
        pause
        exit /b 1
    )
)
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip --quiet

rem ---- 3. PyTorch: con GPU NVIDIA (CUDA 12.6) o solo CPU ----------------------------------------------------
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    set "IDX=cpu"
    echo [3/5] No hay placa NVIDIA: se instala PyTorch para CPU ^(funciona, pero entrenar es mucho mas lento^).
) else (
    set "IDX=cu126"
    echo [3/5] Placa NVIDIA detectada: se instala PyTorch con CUDA 12.6 ^(la descarga es grande^).
)
"%VPY%" -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/%IDX%
if errorlevel 1 goto fallo

rem ---- 4. Resto de las librerias (versiones fijas en requirements.txt) -------------------------------------------
echo [4/5] Instalando el resto de las librerias ...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto fallo

rem ---- 5. Verificacion ----------------------------------------------------------------------------------
echo [5/5] Verificando la instalacion ...
"%VPY%" -c "import torch, ultralytics, gradio, uproot, astropy; import particulas.pipeline; g = torch.cuda.is_available(); print('      Todo instalado. GPU:', torch.cuda.get_device_name(0) if g else 'no (se usa la CPU)')"
if errorlevel 1 goto fallo
if "%IDX%"=="cu126" "%VPY%" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" || echo       AVISO: hay placa NVIDIA pero PyTorch no la ve. Actualiza el driver de NVIDIA y volve a correr este archivo.

echo.
echo ===== Listo =====
echo   Applet con el modelo del proyecto:   listo_para_usar\iniciar_applet.bat
echo   Entrenar con tus datos:              para_entrenar\ENTRENAR_CON_MIS_DATOS.bat
echo.
pause
exit /b 0

:fallo
echo.
echo [ERROR] Fallo la instalacion ^(ver el mensaje de arriba^). Causas comunes: sin conexion a internet,
echo         poco espacio en disco ^(hacen falta ~5 GB^), o un antivirus bloqueando la descarga.
echo         Se puede volver a correr este archivo: retoma donde quedo.
pause
exit /b 1
