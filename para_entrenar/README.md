# Kit 2 · Entrenar con tus datos

Entrena el detector con **tus** imágenes de Skipper-CCD y te deja **tu propio applet** listo para usar.
No hace falta etiquetar nada a mano: las trazas se etiquetan solas con criterios físicos que podés ajustar.

## En 3 pasos

1. **Copiá tus imágenes** en la carpeta `mis_datos/` (FITS, ROOT, PNG, JPG, TIFF o PDF).
   Si tenés varios runs o detectores, hacé una subcarpeta para cada uno.
2. **(Opcional) Revisá los criterios** en `criterios.yaml`. Ver la sección [Criterios](#criterios).
3. **Doble clic en `ENTRENAR_CON_MIS_DATOS.bat`.**

Al terminar aparece la carpeta `mi_applet/`:

| archivo | qué es |
|---|---|
| `iniciar_applet.bat` | abre el applet con tu modelo y tus criterios (doble clic) |
| `RESULTADOS.txt` | cuántas trazas de cada tipo se usaron, cómo le fue al modelo y avisos (pocos datos, sobreajuste) |
| `revision_etiquetas/` | figuras de muestra con las etiquetas automáticas |
| `modelo/`, `criterios.yaml` | tu detector y los criterios con que se etiquetó (el applet usa los mismos) |
| `runs/particulas/` | curvas y métricas del entrenamiento |

Requisitos: el entorno de Python instalado (ver *Instalación* en el README de la raíz). Con una GPU NVIDIA de
6 GB tarda ~1.5 h con ~2000 imágenes; con pocas imágenes, minutos. Sin GPU funciona, pero mucho más lento.

## Cómo funciona

`ENTRENAR_CON_MIS_DATOS.bat` corre `entrenar_todo.py`, que hace cuatro pasos:

1. **Revisar etiquetas**: aplica los criterios a 3 archivos de cada grupo y dibuja las trazas con su tipo.
2. **Construir el dataset**: aplica los criterios a todas tus imágenes. De cada subcarpeta aparta archivos
   enteros para validación (15 %, mínimo 2 si la carpeta tiene 6 o más), que el modelo no ve al entrenar.
3. **Entrenar**: parte del detector ya entrenado del proyecto y lo ajusta a tus datos (aprende rápido y sufre
   menos sobreajuste con pocos datos). Con `--partir-de base` parte de un modelo genérico sin entrenar.
4. **Armar el applet** en `mi_applet/`.

El modelo aprende a reproducir los criterios. Sus métricas (mAP, precisión, recall) miden cuánto coincide con
ellos, **no** con la partícula real. Para eso hacen falta etiquetas hechas a mano o simulaciones.

## Criterios

Todos los criterios están en **`criterios.yaml`**, explicados línea por línea. Se edita con cualquier editor
de texto (Bloc de notas, VS Code): cambiás el número y guardás. Si borrás una línea, se usa el valor por
defecto. Si escribís mal un nombre o un valor, el programa avisa cuál es.

**Por qué importan**: las etiquetas de entrenamiento salen de estos criterios. Si los cambiás, tu modelo
aprende los tuyos. El applet los usa además para separar las trazas y para clasificar las que la red no ve.

Qué hay en el archivo:

| sección | para qué | cuándo cambiarla |
|---|---|---|
| Sensor | espesor y tamaño de píxel | si tu CCD no es de 675 µm / 15 µm |
| Reconstrucción | qué píxeles forman una traza (semilla, crecimiento, energía mínima) | otro ruido, otro umbral |
| Unión de tramos | unir pedazos de un mismo muón (columnas enmascaradas, huecos) | si ves muones partidos o uniones de más |
| Muón / muón corto | largo, rectitud y la banda de energía de una partícula que cruza el sensor | si muones cortos reales salen como electrones (la banda depende de tu calibración) |
| Alfa, puntual, artefacto | energía, tamaño, forma | según tu detector |

**Probar criterios antes de entrenar** (no entrena, sólo dibuja):

```
ENTRENAR_CON_MIS_DATOS.bat --revisar
```

Mirá las figuras en `mi_applet/revision_etiquetas/`, ajustá `criterios.yaml` y repetí hasta que las etiquetas
te convenzan. Recién ahí entrená.

**Usar tus criterios sin entrenar**: el applet del kit listo para usar también los acepta y clasifica con las
reglas (método *Reglas físicas*), sin red:

```
..\.venv\Scripts\python.exe ..\listo_para_usar\app.py --criterios criterios.yaml
```

## Opciones

Se agregan después del `.bat` (o de `entrenar_todo.py`):

| opción | qué hace |
|---|---|
| `--revisar` | sólo dibuja las etiquetas de muestra y termina |
| `--datos D:\mis_imagenes` | usar otra carpeta en vez de `mis_datos/` |
| `--criterios otros.yaml` | usar otro archivo de criterios |
| `--nombre applet_run7` | nombre de la carpeta de salida (para tener varios applets) |
| `--epochs 40` | cuántas pasadas por los datos (60 por defecto; con pocos datos, menos) |
| `--batch 16` | imágenes por paso (8 entra en una GPU de 6 GB; con más memoria se puede subir) |
| `--partir-de base` | entrenar desde el modelo genérico en vez del detector del proyecto |

## Problemas frecuentes

- **"No hay imágenes en mis_datos"**: los archivos tienen que ser FITS, ROOT, PNG, JPG, TIFF o PDF.
- **Avisos en `RESULTADOS.txt`**:
  - *pocas trazas de entrenamiento para X*: el modelo va a reconocer mal esa clase. Hacen falta más imágenes.
  - *pocas imágenes de validación*: las métricas son poco confiables.
  - *posible sobreajuste*: el modelo memoriza tus imágenes en vez de generalizar. Probá con `--epochs` menor
    o con más datos.
- **Se queda sin memoria**: bajá `--batch`, o cerrá otros programas.
- **Muones cortos reales salen como electrones**: revisá `espesor_um`, `pixel_um` y las bandas `mip_kev_um`
  en `criterios.yaml`. Dependen de tu sensor y de tu calibración.

## Paso a paso manual (avanzado)

Los cuatro pasos se pueden correr por separado desde esta carpeta:

```powershell
..\.venv\Scripts\python.exe revisar_etiquetas.py mis_datos --criterios criterios.yaml --max 5
..\.venv\Scripts\python.exe construir_dataset.py --datos mis_datos\run1 mis_datos\run2 --criterios criterios.yaml --val-min 2
..\.venv\Scripts\python.exe entrenar.py --epochs 60          # --cache-ram: más rápido si sobra memoria
..\.venv\Scripts\python.exe ..\listo_para_usar\app.py --modelo modelo_entrenado\detector_particulas.pt --criterios criterios.yaml
```

- Cada `--datos` de `construir_dataset.py` es un grupo (carpeta o patrón con comodines). Los grupos chicos se
  repiten ×4 en train y las imágenes con alfas ×8. El dataset guarda una copia de los criterios usados.
- Los `.txt` de `dataset/labels/` (formato YOLO) se pueden corregir a mano en CVAT o Label Studio antes de
  entrenar: así el modelo aprende de etiquetas mejores que las automáticas.
- Para reemplazar el modelo del kit listo para usar, copiá tu modelo encima de
  `..\listo_para_usar\modelo\detector_particulas.pt`.
