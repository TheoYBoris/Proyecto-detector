@echo off
rem Inicia el applet de identificacion de particulas (modelo ya entrenado).
rem   iniciar_applet.bat             -> solo esta PC (el navegador se abre solo cuando el applet esta listo)
rem   iniciar_applet.bat --red       -> accesible desde otras PCs de la red local
rem   iniciar_applet.bat --compartir -> link publico temporal de Gradio
rem   iniciar_applet.bat --modelo ..\para_entrenar\modelo_entrenado\detector_particulas.pt
rem Para cerrarlo, cerrar esta ventana.
cd /d "%~dp0"
if not exist "..\.venv\Scripts\python.exe" (
    echo No se encontro el entorno de Python ^(..\.venv^). Ver "Instalacion" en el README de la raiz.
    pause
    exit /b 1
)
echo Iniciando el applet... el navegador se abre solo en unos segundos.
"..\.venv\Scripts\python.exe" app.py %*
pause
