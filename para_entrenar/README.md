# Kit 2 · Para entrenar con tus datos

Parte de un **modelo sin entrenar en partículas** (`modelo_base/yolo11n.pt`: pesos genéricos de
ultralytics, preentrenados en fotos comunes) y lo entrena con tus imágenes.

No hacen falta etiquetas hechas a mano. Las trazas se etiquetan solas con las reglas físicas de
`particulas/core.py` (forma, largo, curvatura y energía), y el detector aprende de esas etiquetas.

## Paso a paso

Todos los comandos se corren desde esta carpeta (`para_entrenar/`).

### 1. Revisar cómo quedan las etiquetas automáticas

```powershell
..\.venv\Scripts\python.exe revisar_etiquetas.py "C:\mis_datos\*.fits" --max 5
```

Mira las figuras en `revision_etiquetas/`. Si las reglas no se ajustan a tu detector (otro tamaño de píxel,
otro espesor, otro ruido), cambia los umbrales en `Params` (`particulas/core.py`) y repite.

### 2. Construir el dataset

```powershell
..\.venv\Scripts\python.exe construir_dataset.py --datos "C:\mis_datos\run1" "C:\mis_datos\run2\*.fits"
```

- Cada `--datos` es un **grupo**: una carpeta o un patrón con comodines. Acepta FITS de Skipper-CCD y
  también PNG/JPG/TIFF/PDF (sin calibrar).
- De cada grupo se separa un 15 % de los archivos para validación (`--val-min 2` fuerza al menos 2 archivos
  por grupo: con grupos chicos, un solo archivo no alcanza para detectar sobreajuste).
- Los grupos chicos (menos del 10 % del más grande) se repiten ×4 en train, y las imágenes con alfas ×8.
- Sin `--datos` se usan los datos del experimento original (`../datos/`).

Resultado: `dataset/` en formato YOLO. Los `.txt` de `dataset/labels/` se pueden corregir a mano en CVAT o
Label Studio antes de entrenar, y así el modelo aprende de etiquetas mejores que las automáticas.

### 3. Entrenar

```powershell
..\.venv\Scripts\python.exe entrenar.py --epochs 60
```

- Con una GPU de 6 GB se usa `--batch 8`, y tarda ~1.5 h con ~2000 imágenes. Con más memoria se puede
  subir `--batch`.
- Sin GPU funciona en CPU, pero mucho más lento.
- `--cache-ram` carga todas las imágenes en memoria y entrena más rápido, pero con ~2000 imágenes una PC de
  16 GB se quedó sin memoria. Por defecto las lee del disco.
- Las métricas y curvas quedan en `runs/particulas/`.

Resultado: `modelo_entrenado/detector_particulas.pt`.

### 4. Usar tu modelo

```powershell
..\.venv\Scripts\python.exe ..\listo_para_usar\app.py --modelo modelo_entrenado\detector_particulas.pt
```

o `..\listo_para_usar\detectar.py archivo.fits --modelo modelo_entrenado\detector_particulas.pt`.

Para reemplazar el modelo del kit listo para usar, copia tu archivo encima de
`..\listo_para_usar\modelo\detector_particulas.pt`.
