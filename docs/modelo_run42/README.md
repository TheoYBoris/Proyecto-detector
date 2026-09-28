# Modelo listo para usar: ajuste al run 42

El modelo distribuido en `listo_para_usar/modelo/detector_particulas.pt` incorpora
un ajuste de 10 epocas al run 42, partiendo del detector anterior. Las primeras
10 capas, incluidas sus estadisticas BatchNorm, permanecieron exactamente iguales.
El applet y la linea de comandos cargan estos pesos por defecto.

## Comparacion

Ambos modelos se evaluaron con la misma configuracion: imagen 640, batch 8,
FP32, confianza minima 0.001, IoU de NMS 0.7 y hasta 500 detecciones.

| Conjunto | Imagenes | mAP50 anterior | mAP50 actual | mAP50-95 anterior | mAP50-95 actual |
|---|---:|---:|---:|---:|---:|
| Run 42, validacion | 212 | 0.7131 | 0.7338 | 0.5637 | 0.6030 |
| Anterior, validacion | 300 | 0.7190 | 0.7390 | 0.5714 | 0.6099 |
| Run 42, prueba final | 212 | 0.7201 | 0.7414 | 0.5736 | 0.6126 |

La prueba final comprende 53 FITS completos, con cuatro amplificadores cada uno,
excluidos del entrenamiento y de la seleccion de la mejor epoca. La validacion
anterior ya participo en la seleccion de modelos: no es una prueba independiente.

| Clase en la prueba final | Trazas | mAP50-95 anterior | mAP50-95 actual |
|---|---:|---:|---:|
| Muon | 2954 | 0.8180 | 0.8411 |
| Electron | 4651 | 0.6749 | 0.7219 |
| Alfa | 5 | 0.0068 | 0.0026 |
| Puntual | 2437 | 0.6519 | 0.7274 |
| Artefacto | 636 | 0.7165 | 0.7698 |

La mejora es global y en cuatro clases. **Alfa sigue sin rendimiento util** y
hay muy pocos ejemplos para una conclusion fiable sobre esa clase.
Las etiquetas son automaticas, generadas por reglas fisicas: las metricas miden
acuerdo con esas reglas, no identificacion fisica confirmada. Se evaluo la red
YOLO; el applet tambien reconstruye trazas y aplica reglas como respaldo.
No se garantiza una mejora en cada imagen ni en otros instrumentos.

Las metricas completas y la huella de los pesos estan en [comparacion.json](comparacion.json).
El campo `original_intacto` registra la comprobacion realizada al finalizar el
ajuste, antes de sustituir el modelo distribuido por esta version.

## Procedimiento

- 702 FITS del run 42 reducidos a 351 archivos unicos mediante SHA-256.
- Separacion por exposicion completa, semilla 42: 245 train, 53 val y 53 test.
- Verificacion de ausencia de copias entre train, validacion y prueba final.
- 980 imagenes nuevas y 524 imagenes de entrenamiento anteriores como refuerzo,
  incluidos los darks de train y los archivos de train que contienen alfas.
- Etiquetas con `para_entrenar/criterios.yaml`, iguales a los parametros por defecto.
- YOLO11n, primeras 10 capas congeladas, AdamW, `lr0=0.0002`, `lrf=0.1`,
  calentamiento de una epoca, paciencia 5, FP32, batch 4, imagen 640.
- Solo reflexiones horizontal/vertical; sin cambios de color, escala ni mosaic.
- 10 epocas; mejor checkpoint en la epoca 10, elegido por validacion combinada.
- GPU GTX 1660 SUPER; unos 5.5 minutos de ajuste.

La prueba de integracion con `detectar.py` genero CSV y figura con 203 trazas
para `img341.fits`. Se verificaron la huella del checkpoint y las rutas del applet.

## Archivo local

Los pesos anteriores, datasets, checkpoints, logs y el script especifico del
experimento se conservaron en `salidas_anteriores/ajuste_run42/`, fuera de git.
Los pesos anteriores tambien se pueden recuperar del historial de git.
Ese archivo conserva rutas del experimento original; no es el kit de entrenamiento
ni un applet instalado. Para usar el modelo actual, abrir
`listo_para_usar/iniciar_applet.bat`.

`para_entrenar/` contiene el kit para otros usuarios. Su flujo principal parte
del modelo actual por defecto, o del modelo generico con `--partir-de base`.
