# Kit 1 · Listo para usar

Identifica partículas en imágenes de Skipper-CCD con el **modelo ya entrenado**
(`modelo/detector_particulas.pt`). No hace falta entrenar nada.

## Applet web

Doble clic en `iniciar_applet.bat`, o desde esta carpeta:

```powershell
..\.venv\Scripts\python.exe app.py              # solo esta PC:        http://127.0.0.1:7860
..\.venv\Scripts\python.exe app.py --red        # otras PCs de la red:  http://<IP-de-esta-PC>:7860
..\.venv\Scripts\python.exe app.py --compartir  # link público temporal (*.gradio.live, dura 1 semana)
```

En el navegador se suben uno o varios archivos (FITS, PNG, JPG, TIFF o PDF). El applet devuelve:
- Cada imagen con las trazas encerradas e identificadas.
- Una tabla con el conteo por partícula y por archivo, más el total.
- Un gráfico de barras con los totales.
- Un ZIP con las imágenes, los CSV de detecciones y los FITS convertidos.

**Usar un modelo propio** (entrenado con el kit `para_entrenar/`):

```powershell
..\.venv\Scripts\python.exe app.py --modelo ..\para_entrenar\modelo_entrenado\detector_particulas.pt
```

## Línea de comandos

```powershell
# detectar (FITS o imágenes); guarda PNG + CSV en detecciones\
..\.venv\Scripts\python.exe detectar.py nueva.fits "otra_carpeta\*.png" --metodo auto

# convertir PNG/JPG/TIFF/PDF a FITS de pseudo-electrones (sin detectar)
..\.venv\Scripts\python.exe convertir_a_fits.py figura.pdf --out convertidos
```

Métodos: `auto` (YOLO en imágenes con binning, reglas físicas sin binning), `yolo` o `reglas`.
Detalles del algoritmo y sus limitaciones: ver el `README.md` de la raíz.
