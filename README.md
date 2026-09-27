# Identificación de partículas en imágenes Skipper-CCD

Detector que recibe una imagen de un Skipper-CCD (FITS, o PNG/JPG/TIFF/PDF), encierra en una caja cada
traza y dice qué partícula la produjo (muón, electrón, alfa, depósito puntual o artefacto), con una
estimación de la energía depositada cuando la imagen es un FITS calibrable.

## ¿Qué carpeta uso?

| quiero… | carpeta |
|---|---|
| identificar partículas en mis imágenes ya mismo (applet web o línea de comandos) | [`listo_para_usar/`](listo_para_usar/README.md) · modelo **entrenado** incluido |
| entrenar el detector con mis propios datos | [`para_entrenar/`](para_entrenar/README.md) · modelo **sin entrenar** + paso a paso |
| entender qué se hizo, los resultados y el estado del proyecto | [`docs/bitacora/bitacora.pdf`](docs/bitacora/bitacora.pdf) · bitácora (LaTeX) |

## Estructura

```
├── listo_para_usar/        KIT 1: applet + detección con el modelo ya entrenado
│   ├── modelo/detector_particulas.pt      (entrenado con ~100 000 trazas)
│   ├── app.py, iniciar_applet.bat         applet web (Gradio)
│   ├── detectar.py                        detección por línea de comandos
│   └── convertir_a_fits.py                PNG/JPG/TIFF/PDF → FITS
├── para_entrenar/          KIT 2: entrenar con datos propios
│   ├── modelo_base/yolo11n.pt             modelo sin entrenar en partículas (punto de partida)
│   ├── revisar_etiquetas.py               paso 1: revisar las etiquetas automáticas
│   ├── construir_dataset.py               paso 2: armar el dataset YOLO
│   └── entrenar.py                        paso 3: entrenar → modelo_entrenado/
├── particulas/             librería común que usan los dos kits
│   ├── core.py                            calibración, reconstrucción de trazas, reglas de clasificación
│   ├── imagenes.py                        lectura de PNG/JPG/TIFF/PDF y conversión a pseudo-electrones
│   ├── pipeline.py                        detección completa (YOLO o reglas) sobre un archivo
│   ├── instrumentos.py                    perfiles Atucha-II / CONNIE (specs publicadas), masa, blob/difusión
│   └── dibujo.py                          figuras con las cajas
├── docs/bitacora/          bitácora del proyecto (documento vivo)
│   ├── bitacora.tex / bitacora.pdf        compilar con: pdflatex bitacora.tex
│   └── generar_figuras.py                 regenera las figuras y los números citados
├── herramientas/
│   └── ver_imagenes.py                    visor simple de FITS/ROOT
└── datos/                  datos crudos del experimento (no se suben al repositorio)
```

## Instalación (después de clonar)

Requiere Python 3.14 y, para usar la GPU, una placa NVIDIA con drivers recientes.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Sin GPU NVIDIA, instalar `torch` y `torchvision` sin `--index-url`: funciona en CPU, más lento.
Los datos crudos del experimento (~1.5 GB) no están en el repositorio. Para reentrenar con ellos hay que
copiarlos en `datos/201211/` y `datos/proc_corr_proc/`.

## Cómo funciona

No había etiquetas, así que el entrenamiento es en dos etapas:

1. **Calibración** (`particulas/core.py`), para cada amplificador:
   - Pedestal por fila a partir del overscan.
   - Ruido: MAD del overscan.
   - Ganancia: ajuste del histograma de píxeles con ruido gaussiano ⊗ Poisson (picos de 0, 1, 2… e⁻).
     Así la imagen queda en electrones. Valores medidos:
     darks 2020 ≈ 500/500/600/570 ADU/e⁻; run 43 ≈ 860/860/735/770 ADU/e⁻; ruido ≈ 0.15–0.28 e⁻.
   - Enmascarado de columnas malas (prescan y columnas calientes) y de los NaN.
2. **Reconstrucción**: umbral con histéresis (semilla ≥ 20 e⁻, crecimiento ≥ 4 e⁻) → un cluster por traza.
   Si un cluster contiene una recta dominante (un muón que cruza o toca otra traza), se separa con RANSAC.
3. **Auto-etiquetado por morfología**, con las variables medidas en píxeles físicos
   (el run 43 tiene binning ×10 en columnas y se corrige):

   | clase | criterio |
   |---|---|
   | artefacto | línea de 1 fila (registro serie) o de 1 columna |
   | alfa | ≥ 1 MeV en una mancha compacta (≤ 60 px) y redonda; el núcleo satura el ADC |
   | puntual | largo ≤ 7 px (depósito limitado por difusión: rayos X, baja energía) |
   | muón | largo ≥ 30 px, ancho/largo ≤ 0.10, sagita/largo ≤ 0.03 (recta) |
   | electrón | todo lo demás (trazas curvas, "gusanos" de Compton/beta) |

   Los umbrales están en `Params` (`particulas/core.py`).
4. **Detector YOLO11n** (`para_entrenar/entrenar.py`), entrenado con esas etiquetas sobre imágenes de 3
   canales: log(E), E lineal de baja energía y máscara ≥ 4 e⁻. Cada amplificador es una imagen.

En la detección se aplican además dos filtros físicos. Se descartan las cajas sin un depósito real (sin
ningún píxel ≥ 20 e⁻ o con menos de 60 e⁻ en total), y se usa NMS agnóstico a la clase para que una traza
no reciba dos cajas de clases distintas.

## Resultados del modelo entrenado (validación: 300 imágenes no vistas, 15 434 trazas)

| clase | mAP50 | precisión | recall |
|---|---|---|---|
| artefacto | 0.97 | 0.93 | 0.92 |
| muón | 0.93 | 0.83 | 0.87 |
| electrón | 0.89 | 0.87 | 0.78 |
| puntual | 0.88 | 0.82 | 0.92 |
| alfa | ~0 | – | 0 (solo 9 casos) |

Estas métricas miden el acuerdo con las reglas, no con la física real (ver limitaciones).
Funciona bien en imágenes como las del run 43, que son el 98 % del entrenamiento.
En los darks de 2020 (sin binning, solo 9 archivos) fragmenta las trazas largas en varias cajas.
Por eso el método `auto` usa las reglas físicas en imágenes sin binning.

## Imágenes PNG / JPG / TIFF / PDF

Una imagen exportada no conserva los ADU del sensor, el overscan ni los encabezados, así que **no se puede
calibrar**. `particulas/imagenes.py` la convierte en un mapa de *pseudo-electrones*:
1. Pasa a luminancia y la invierte si las trazas son oscuras sobre fondo claro.
2. Si es una figura con ejes (matplotlib, etc.), detecta cada panel y descarta ejes, textos y barras de color.
3. Resta un fondo suave, mide el ruido de la propia imagen y reescala la intensidad (logarítmica) al rango
   que esperan las reglas.

La forma de las trazas se conserva y la clasificación funciona; **la energía no se informa**.
Validación sobre un amplificador de referencia (80 trazas en el FITS calibrado):

| entrada | trazas | comentario |
|---|---|---|
| PNG gris / invertido / JPEG / PNG 16 bits | 78–79 | mismos muones; electrones y puntuales casi iguales |
| figura con ejes / PDF de matplotlib | 49–52 | la figura re-muestrea la imagen: se pierden trazas débiles |
| figura de 4 paneles con escala lineal saturada | ~2× puntuales | el ruido quedó tan brillante como una traza |

Recomendaciones: subir la imagen cruda en vez de una figura. Si la imagen está ampliada, indicar la
*escala* (píxeles de imagen por píxel del CCD). No subir imágenes con anotaciones encima de los datos.

## Limitaciones importantes

- **Las etiquetas son automáticas, no verdad de campo.** El detector aprende a reproducir las reglas
  morfológicas: su "precisión" mide cuánto coincide con ellas, no con la física real. Para ir más allá:
  corregir a mano parte de las etiquetas (los `.txt` de `para_entrenar/dataset/labels` se abren en CVAT o
  Label Studio, formato YOLO) o entrenar con simulaciones (Geant4) donde se conoce la partícula real.
- **Muón vs. electrón es ambiguo en trazas rectas cortas**: los electrones de alta energía también dejan
  trazas casi rectas. El corte de largo ≥ 30 px es convencional.
- **Alfas**: son muy pocas (45 en todo el dataset), así que el detector las aprende mal.
- Las trazas que se cruzan sin que ninguna sea recta quedan en una sola caja.
- La energía de trazas saturadas (alfas, muones muy horizontales en el run 43) está subestimada.
- La GTX 1660 no soporta bien FP16 (AMP), así que se entrenó en FP32 con YOLO11n y batch 8.

## Entorno

`.venv` con Python 3.14, PyTorch 2.14 + CUDA 12.6, ultralytics 8.4 y Gradio 6.28 (ver `requirements.txt`).
