@echo off
rem Inicia el applet de identificacion de particulas (modelo ya entrenado).
rem   iniciar_applet.bat             -> solo esta PC (http://127.0.0.1:7860)
rem   iniciar_applet.bat --red       -> accesible desde otras PCs de la red local
rem   iniciar_applet.bat --compartir -> link publico temporal de Gradio
rem   iniciar_applet.bat --modelo ..\para_entrenar\modelo_entrenado\detector_particulas.pt
cd /d "%~dp0"
start "" http://127.0.0.1:7860
"..\.venv\Scripts\python.exe" app.py %*
pause
