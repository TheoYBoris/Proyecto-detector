# Identificación de partículas en imágenes Skipper-CCD

Detector que recibe una imagen de un Skipper-CCD (FITS, ROOT, o PNG/JPG/TIFF/PDF), encierra en una caja
cada traza y dice qué partícula la produjo (muón, electrón, alfa, depósito puntual o artefacto), con una
estimación de la energía depositada cuando la imagen es un FITS o ROOT calibrable.

## ¿Qué carpeta uso?

| quiero… | carpeta |
|---|---|
| identificar partículas en mis imágenes ya mismo (applet web o línea de comandos) | [`listo_para_usar/`](listo_para_usar/README.md) · modelo **entrenado** incluido |
| entrenar el detector con mis propios datos y tener mi propio applet | [`para_entrenar/`](para_entrenar/README.md) · dejar los datos en `mis_datos/` y doble clic en `ENTRENAR_CON_MIS_DATOS.bat`; criterios ajustables en `criterios.yaml` |
| entender qué se hizo, los resultados y el estado del proyecto | [`docs/bitacora/bitacora.pdf`](docs/bitacora/bitacora.pdf) · bitácora (LaTeX) |

## Estructura

```
├── listo_para_usar/        KIT 1: applet + detección con el modelo ya entrenado
│   ├── modelo/detector_particulas.pt      (entrenado con ~100 000 trazas)
│   ├── app.py, iniciar_applet.bat         applet web (Gradio)
│   ├── detectar.py                        detección por línea de comandos
│   └── convertir_a_fits.py                ROOT/PNG/JPG/TIFF/PDF → FITS
├── para_entrenar/          KIT 2: entrenar con datos propios
│   ├── ENTRENAR_CON_MIS_DATOS.bat         todo en uno: mis_datos/ -> modelo -> mi_applet/
│   ├── entrenar_todo.py                   lo que corre el .bat (revisar, dataset, entrenar, applet)
│   ├── criterios.yaml                     criterios de reconstrucción y clasificación, explicados
│   ├── mis_datos/                         donde cada usuario deja sus imágenes
│   ├── modelo_base/yolo11n.pt             modelo sin entrenar en partículas (punto de partida)
│   ├── revisar_etiquetas.py               paso 1: revisar las etiquetas automáticas
│   ├── construir_dataset.py               paso 2: armar el dataset YOLO
│   └── entrenar.py                        paso 3: entrenar → modelo_entrenado/
├── particulas/             librería común que usan los dos kits
│   ├── core.py                            calibración, reconstrucción de trazas, reglas de clasificación
│   ├── imagenes.py                        lectura de PNG/JPG/TIFF/PDF y conversión a pseudo-electrones
│   ├── root.py                            lectura de archivos ROOT (skipper2root, TTree/RNTuple, TH2)
│   ├── pipeline.py                        detección completa (YOLO o reglas) sobre un archivo
│   ├── instrumentos.py                    perfiles Atucha-II / CONNIE (specs publicadas), masa, blob/difusión
│   └── dibujo.py                          figuras con las cajas
├── docs/bitacora/          bitácora del proyecto (documento vivo)
│   ├── bitacora.tex / bitacora.pdf        compilar con: pdflatex bitacora.tex
│   └── generar_figuras.py                 regenera las figuras y los números citados
├── herramientas/
│   └── ver_imagenes.py                    visor simple de FITS/ROOT
├── datos/                  datos crudos del experimento (no se suben al repositorio)
└── LICENSE                 licencia de uso libre (texto estándar tipo MIT)
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
   | artefacto | línea de 1 fila (registro serie) o de 1 columna; o línea horizontal corta sin difusión vertical (σy < 0.3 px, corte de Atucha-II) y sin energía de muón |
   | alfa | ≥ 1 MeV en una mancha compacta (≤ 60 px) y redonda; el núcleo satura el ADC |
   | puntual | largo ≤ 7 px (depósito limitado por difusión: rayos X, baja energía) |
   | muón | largo ≥ 30 px, ancho/largo ≤ 0.10, sagita/largo ≤ 0.03 (recta) |
   | muón corto | recta de 7–30 px (15–30 px con binning) con la energía de una MIP que cruza los 675 µm |
   | electrón | todo lo demás (trazas curvas, "gusanos" de Compton/beta) |

   Los umbrales están en `Params` (`particulas/core.py`) y se pueden cambiar sin tocar código con un archivo
   de criterios ([`para_entrenar/criterios.yaml`](para_entrenar/criterios.yaml), opción `--criterios` del
   applet, `detectar.py` y los scripts de entrenamiento).

   **Muones cortos.** Un muón atraviesa todo el espesor, así que su traza mide 45·tan α píxeles (α: ángulo
   con la normal al CCD) y deposita la energía de una MIP a lo largo de √((15 L)² + 675²) µm. La banda de
   energía (p5 a 1.5×p98 de los muones largos del mismo tipo de imagen) es 0.126–0.57 keV/µm sin binning y
   0.236–0.79 con binning; la diferencia entre ambas escalas es un punto abierto. El criterio se validó
   contra un Monte Carlo con flujo ∝ cos²θ y la orientación de cada CCD. El CCD de los darks es horizontal y
   el de Atucha-II vertical ([JHEP24], Fig. 2; confirmado porque los muones del run 43 van a lo largo de x):

   | largo (px) | 7–15 | 15–30 | 30–45 | 45–60 | 60–90 | 90+ |
   |---|---|---|---|---|---|---|
   | darks: muones con la regla / predicción MC | 32 / 34 | 84 / 72 | 44 / 47 | 20 / 23 | 22 / 16 | 9 / 6 |
   | run 43 (41 imágenes): regla / predicción MC | 9 / 44 | 263 / 242 | 415 / 402 | 348 / 402 | 508 / 566 | 779 / 813 |

   Por debajo de 7 px (y de 15 px con binning) los muones casi perpendiculares no se distinguen de otros
   depósitos y se dejan como están.
4. **Detector YOLO11n** (`para_entrenar/entrenar.py`), entrenado con esas etiquetas sobre imágenes de 3
   canales: log(E), E lineal de baja energía y máscara ≥ 4 e⁻. Cada amplificador es una imagen.

**Detección = reconstrucción + red** (`pipeline.detectar_hibrido`, desde v0.6). Las trazas (píxeles, caja y
energía) salen de la reconstrucción por píxeles, y la red sólo las clasifica: cada traza toma la clase de la
caja de la red que mejor se le superpone (IoU ≥ 0.3). Si la red no tiene caja para esa traza (por ejemplo,
depósitos bajo los 60 e⁻ con que se entrenó), clasifican las reglas; la columna `origen` del CSV dice cuál.
Antes, las cajas de la red eran el resultado final y un muón largo cruzado por otra partícula podía salir
partido en varias cajas: la red dejaba fragmentos en el 19 % de esos muones. La reconstrucción no parte muones
y separa las partículas que los cruzan; además la energía de cada traza ya no incluye la de lo que la cruza.
En los 75 archivos de validación (15 554 trazas) coincide con las reglas más que la red sola en todas las
clases (electrón 79 → 86 %, muón 78 → 80 %, puntual 91 → 98 %, artefacto 85 → 89 %).

## Resultados del modelo entrenado (validación: 300 imágenes no vistas, 15 434 trazas)

Modelo v2 (27/09/2026), reentrenado con las reglas que incluyen muones cortos y eventos del registro serie:

| clase | mAP50 | precisión | recall | v1 contra las mismas etiquetas (mAP50) |
|---|---|---|---|---|
| artefacto | 0.94 | 0.88 | 0.89 | 0.79 |
| muón | 0.90 | 0.83 | 0.82 | 0.86 |
| electrón | 0.86 | 0.83 | 0.78 | 0.80 |
| puntual | 0.88 | 0.83 | 0.92 | 0.88 |
| alfa | ~0 | – | 0 (solo 9 casos) | ~0 |

Por grupo de trazas, en la misma validación:

| trazas | n | v1 las llama muón | v2 las llama muón |
|---|---|---|---|
| muones cortos (< 30 px) | 446 | 1 % | 31 % |
| muones largos | 3864 | 88 % | 87 % |
| electrones rectos cortos | 768 | 0.8 % | 3.1 % |
| otros electrones | 6019 | 9.2 % | 9.8 % |

El reentrenamiento recupera parte de los muones cortos sin afectar a los largos, pero el 65 % de los cortos
sigue saliendo como electrón. Son pocos en el entrenamiento y lo que los distingue es la energía por unidad de
largo, que la red ve sólo de forma indirecta. Los eventos cortos del registro serie pasan de 0 % a 66 %
reconocidos como artefacto. El modelo v1 queda en `salidas_anteriores/` (no versionado) y en el historial de git.

Estas métricas miden el acuerdo con las reglas, no con la física real (ver limitaciones).
Funciona bien en imágenes como las del run 43, que son el 98 % del entrenamiento.
En los darks de 2020 (sin binning, solo 9 archivos) fragmenta las trazas largas en varias cajas.
Por eso el método `auto` usa las reglas físicas en imágenes sin binning.

## Archivos ROOT

`particulas/root.py` lee el ROOT y arma en memoria el mismo FITS (una extensión por amplificador, con su
encabezado). Así, la calibración y la detección son exactamente las de un FITS. Formatos reconocidos:

| formato | contenido | calibración |
|---|---|---|
| `skipper2root` (estándar de Skipper-CCD) | `skPixTree` (x, y, ohdu, pix) o `skTablePixTree` (x, y, pix[n]) + encabezados en `headerTree_N` | igual que el FITS: overscan, `CCDNPRES`, `CCDNCOL`, `NBINCOL` |
| TTree o RNTuple genérico | ramas x/col, y/row, un valor (pix/charge/val/adc/…) y opcionalmente ohdu/hdu/amp | sin encabezado: pedestal de la zona activa, sin binning |
| histogramas 2D (TH2F/TH2D/…) | cada histograma es una imagen | como el genérico |

Validación: los 9 darks de 2020 en ROOT dan las mismas ganancias y **exactamente las mismas trazas** que
sus FITS (con reglas y con YOLO). Si un TH2 o TTree ya está en electrones, usar ganancia = 1.
Para pasar un ROOT a FITS sin detectar: `convertir_a_fits.py archivo.root`.

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
- **Muón vs. electrón en trazas rectas**: un electrón de alta energía que cruza todo el sensor deja la misma
  firma que un muón (recta con energía de MIP); ninguna regla los separa. Los muones de menos de 7 px
  (15 px con binning) no se recuperan.
- **Alfas**: son muy pocas (45 en todo el dataset), así que el detector las aprende mal.
- Las trazas que se cruzan sin que ninguna sea recta quedan en una sola caja.
- Cuando una partícula cruza un muón, el píxel del cruce puede salir como un depósito chico aparte. Las dos
  mitades de la partícula que cruza se vuelven a unir si quedan en la misma recta (ver abajo).
- **Tramos colineales** (`core._unir_colineales`, v0.7): un muón largo quedaba partido cuando cruzaba una columna
  enmascarada por la calibración, cuando su carga bajaba del umbral en algún píxel o donde lo cruzaba otra
  partícula. Cada tramo tenía sólo parte de la energía y los cortos terminaban como electrones. Ahora se unen
  dos tramos rectos casi paralelos (< 6°), sobre la misma recta y con un hueco chico (≤ 20 px físicos sin
  binning, ≤ 3 columnas con binning, sin contar las columnas enmascaradas del medio). Muones partidos:
  17 → 2 en los darks y 38 → 6 en el run 43 (las 6 restantes son líneas de artefacto); en los darks, 65 trazas
  que salían como electrones eran tramos de muones. La energía de la traza unida no incluye la de las
  columnas enmascaradas.
- La energía de trazas saturadas (alfas, muones muy horizontales en el run 43) está subestimada.
- La GTX 1660 no soporta bien FP16 (AMP), así que se entrenó en FP32 con YOLO11n y batch 8.

## Licencia

© 2026 Theo Del Compare y Santiago Romero. El código se distribuye bajo una licencia permisiva ([LICENSE](LICENSE)):
se puede usar, copiar, modificar y redistribuir libremente, manteniendo el aviso de copyright y sin garantías.
Es el texto estándar conocido como "licencia MIT" (el nombre viene de la institución donde se redactó; usarlo
no implica ninguna relación con ella).

Ojo: el detector usa [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), que tiene licencia
**AGPL-3.0**. La licencia de este repositorio cubre el código propio. Al redistribuir el conjunto (o los pesos `.pt`
entrenados con Ultralytics), o al ofrecer el applet como servicio público, rigen además los términos de la
AGPL-3.0 de Ultralytics, salvo que se tenga una licencia comercial de ellos.

## Entorno

`.venv` con Python 3.14, PyTorch 2.14 + CUDA 12.6, ultralytics 8.4 y Gradio 6.28 (ver `requirements.txt`).
