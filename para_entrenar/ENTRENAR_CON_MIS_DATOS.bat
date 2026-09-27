@echo off
rem Entrena el detector con las imagenes de para_entrenar\mis_datos\ y arma para_entrenar\mi_applet\.
rem Opciones (se pasan tal cual a entrenar_todo.py):
rem   ENTRENAR_CON_MIS_DATOS.bat --revisar          solo dibuja las etiquetas automaticas
rem   ENTRENAR_CON_MIS_DATOS.bat --epochs 30 --nombre applet_run7
cd /d "%~dp0"
if not exist "..\.venv\Scripts\python.exe" (
    echo No se encontro el entorno de Python ^(..\.venv^). Ver "Instalacion" en el README de la raiz.
    pause
    exit /b 1
)
"..\.venv\Scripts\python.exe" entrenar_todo.py %*
pause
