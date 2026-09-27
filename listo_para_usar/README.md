# Kit 1 · Listo para usar

Identifica partículas en imágenes de Skipper-CCD con el **modelo ya entrenado**
(`modelo/detector_particulas.pt`). No hace falta entrenar nada.

## Applet web

Doble clic en `iniciar_applet.bat`. El navegador se abre solo cuando el applet está listo (unos 5 s); para
cerrarlo, cerrar la ventana negra. También desde esta carpeta:

```powershell
..\.venv\Scripts\python.exe app.py              # solo esta PC:        http://127.0.0.1:7860
..\.venv\Scripts\python.exe app.py --red        # otras PCs de la red:  http://<IP-de-esta-PC>:7860
..\.venv\Scripts\python.exe app.py --compartir  # link público temporal (*.gradio.live, dura 1 semana)
..\.venv\Scripts\python.exe app.py --criterios ..\para_entrenar\criterios.yaml   # otros criterios
```

Si el puerto 7860 está ocupado (por ejemplo, otro applet abierto), usa el siguiente libre (7861, 7862…) y lo
muestra en la ventana. Si el navegador dice "conexión rechazada", fijate en esa ventana la dirección correcta;
con navegadores con VPN integrada (p. ej. Opera), la VPN puede bloquear `127.0.0.1`.

En el navegador se suben uno o varios archivos (FITS, ROOT, PNG, JPG, TIFF o PDF). El applet devuelve:
- Cada imagen con las trazas encerradas e identificadas.
- Una tabla con el conteo por partícula y por archivo, más el total.
- Un gráfico de barras con los totales.
- Un ZIP con las imágenes, los CSV de detecciones y los FITS convertidos.

Los **ROOT** (formato `skipper2root`, TTree/RNTuple con x, y, pix, o histogramas TH2) se calibran igual que
un FITS y dan la misma clasificación y energía. El ZIP incluye además cada ROOT convertido a FITS.

Además, a partir de las especificaciones publicadas del instrumental (ver `docs/bitacora/`), el applet:
- **Reconoce el instrumento** por el encabezado FITS (Atucha-II, CONNIE o genérico). Avisa si un amplificador
  tiene más ruido que el publicado.
- **Divide los depósitos puntuales** según la convención de CONNIE: *blob* (> 600 eV) y *difusión* (< 600 eV).
  Los de difusión incluyen los candidatos a CEvNS.
- **Dibuja el espectro de energía** con las líneas de fluorescencia del cobre (8.05 / 8.91 keV), para verificar
  la calibración.
- **Calcula tasas** en eventos/(g·día) con la masa activa del sensor y el tiempo de exposición.
- **Deja bajar el umbral de energía** a 45 eV (Atucha-II) o 15 eV (CONNIE). El valor por defecto es 225 eV.

**Usar un modelo propio** (entrenado con el kit `para_entrenar/`):

```powershell
..\.venv\Scripts\python.exe app.py --modelo ..\para_entrenar\modelo_entrenado\detector_particulas.pt
```

## Línea de comandos

```powershell
# detectar (FITS, ROOT o imágenes); guarda PNG + CSV en detecciones\
..\.venv\Scripts\python.exe detectar.py nueva.fits datos.root "otra_carpeta\*.png" --metodo auto

# convertir a FITS sin detectar: ROOT -> mismos ADU y encabezados; imagen -> pseudo-electrones
..\.venv\Scripts\python.exe convertir_a_fits.py figura.pdf --out convertidos
..\.venv\Scripts\python.exe convertir_a_fits.py "..\..\datos\201211\*.root"
```

Métodos: `auto` (YOLO en imágenes con binning, reglas físicas sin binning), `yolo` o `reglas`. Con `yolo` las
trazas salen de la reconstrucción por píxeles y la red clasifica cada una (columna `origen` del CSV).
Detalles del algoritmo y sus limitaciones: ver el `README.md` de la raíz.
