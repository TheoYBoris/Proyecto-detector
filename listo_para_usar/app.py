"""Applet web: los usuarios suben imagenes de Skipper-CCD (FITS, o PNG/JPG/TIFF/PDF) y reciben las
imagenes con cada traza identificada, mas el conteo de particulas por tipo.

    ..\\.venv\\Scripts\\python.exe app.py               # solo esta PC:      http://127.0.0.1:7860
    ..\\.venv\\Scripts\\python.exe app.py --red         # toda la red local: http://<IP-de-esta-PC>:7860
    ..\\.venv\\Scripts\\python.exe app.py --compartir   # link publico temporal (*.gradio.live, 1 semana)
    ..\\.venv\\Scripts\\python.exe app.py --modelo ..\\para_entrenar\\modelo_entrenado\\detector_particulas.pt
"""
import argparse
import os
import re
import sys
import tempfile
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import pandas as pd

from particulas.core import CLASES, DESCRIPCION
from particulas.dibujo import COLORES
from particulas.imagenes import FORMATOS, es_imagen, guardar_fits
from particulas.pipeline import METODOS, MODELO_DEFAULT, cargar_modelo, guardar_figura, procesar_archivo

NOMBRES = {"muon": "Muón", "electron": "Electrón", "alfa": "Alfa", "puntual": "Puntual", "artefacto": "Artefacto"}
EXTENSIONES = [".fits", ".fit", ".fts", ".fz", ".gz"] + sorted(FORMATOS)

RUTA_MODELO = MODELO_DEFAULT
MODELO = None


def modelo():
    global MODELO
    if MODELO is None:
        MODELO = cargar_modelo(RUTA_MODELO)
    return MODELO


def analizar_archivos(archivos, metodo, conf, ganancia, escala, binx, paneles, progress=gr.Progress()):
    if not archivos:
        raise gr.Error("Subí al menos un archivo (FITS, PNG, JPG, TIFF o PDF).")
    salida = tempfile.mkdtemp(prefix="particulas_")
    galeria, filas_conteo, todas, errores = [], [], [], []
    hubo_imagenes = False
    t0 = time.time()
    for i, f in enumerate(archivos):
        ruta = f if isinstance(f, str) else f.name
        nombre = os.path.basename(ruta)
        base = nombre.rsplit(".", 1)[0].replace(".fits", "")
        progress(i / len(archivos), desc=f"Analizando {nombre}")
        try:
            amps, dets, usado = procesar_archivo(
                ruta, metodo, modelo() if metodo != "reglas" else None, conf,
                ganancia if ganancia and ganancia > 0 else None,
                escala=float(escala or 1.0), binx=int(binx or 1), paneles=bool(paneles))
        except Exception as e:  # archivo corrupto, formato no reconocido, sin imagenes, etc.
            # no mostrar rutas internas del servidor: dejar solo el nombre del archivo
            detalle = re.sub(r"'[^']*[\\/]([^'\\/]+)'", r"'\1'", str(e))
            errores.append(f"**{nombre}**: no se pudo procesar — formato no reconocido o archivo dañado "
                           f"({type(e).__name__}: {detalle[:200]})")
            continue
        imagen = es_imagen(ruta)
        hubo_imagenes |= imagen
        filas = [dict(archivo=nombre, **d) for ds in dets for d in ds]
        todas += filas
        cuenta = pd.Series([d["clase"] for d in filas], dtype=object).value_counts()
        fila = {"Archivo": nombre, "Tipo": "Imagen (sin calibrar)" if imagen else "FITS calibrado",
                "Método": METODOS[usado].split(" (")[0], "Paneles": len(amps), "Binning": f"x{amps[0].binx}"}
        fila.update({NOMBRES[c]: int(cuenta.get(c, 0)) for c in CLASES})
        fila["Total"] = len(filas)
        filas_conteo.append(fila)

        png = os.path.join(salida, base + "_identificado.png")
        guardar_figura(amps, dets, png, titulo=f"{nombre}  -  {METODOS[usado]}")
        pd.DataFrame(filas).to_csv(os.path.join(salida, base + "_detecciones.csv"), index=False)
        if imagen:
            # la imagen convertida queda disponible como FITS para usarla con el resto de las herramientas
            guardar_fits(amps, os.path.join(salida, base + "_convertido.fits"), origen=nombre)
        resumen = ", ".join(f"{NOMBRES[c]}: {int(cuenta.get(c, 0))}" for c in CLASES if cuenta.get(c, 0))
        galeria.append((png, f"{nombre} — {len(filas)} trazas ({resumen})"))

    if not filas_conteo:
        raise gr.Error("Ningún archivo se pudo procesar. " + " ".join(e.replace("**", "") for e in errores))

    conteo = pd.DataFrame(filas_conteo)
    total = {"Archivo": "TOTAL", "Tipo": "", "Método": "", "Paneles": int(conteo["Paneles"].sum()), "Binning": ""}
    total.update({NOMBRES[c]: int(conteo[NOMBRES[c]].sum()) for c in CLASES})
    total["Total"] = int(conteo["Total"].sum())
    conteo = pd.concat([conteo, pd.DataFrame([total])], ignore_index=True)
    conteo.to_csv(os.path.join(salida, "conteo_particulas.csv"), index=False)
    pd.DataFrame(todas).to_csv(os.path.join(salida, "todas_las_detecciones.csv"), index=False)

    zip_path = os.path.join(salida, "resultados_particulas.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for n in sorted(os.listdir(salida)):
            if n != os.path.basename(zip_path):
                z.write(os.path.join(salida, n), n)

    barras = pd.DataFrame({"Partícula": [NOMBRES[c] for c in CLASES],
                           "Cantidad": [total[NOMBRES[c]] for c in CLASES]})
    lineas = [f"### {total['Total']} trazas identificadas en {len(filas_conteo)} archivo(s) "
              f"({time.time() - t0:.1f} s)",
              " · ".join(f"**{NOMBRES[c]}**: {total[NOMBRES[c]]}" for c in CLASES)]
    if hubo_imagenes:
        lineas.append("⚠️ **Imágenes PNG/JPG/TIFF/PDF:** no traen los valores crudos del sensor, así que no se "
                      "pueden calibrar. La clasificación usa la forma de las trazas, pero **la energía no está "
                      "disponible**. En figuras con escala saturada, el ruido puede contarse como depósitos "
                      "puntuales. El ZIP incluye cada imagen convertida a FITS.")
    if errores:
        lineas.append("**Archivos con problemas:**\n\n" + "\n".join(f"- {e}" for e in errores))
    progress(1.0)
    return "\n\n".join(lineas), conteo, barras, galeria, zip_path


LEYENDA = "\n".join(
    f"- <span style='color:{COLORES[c]};font-weight:700'>■ {NOMBRES[c]}</span>: {DESCRIPCION[c].split(': ', 1)[1]}"
    for c in CLASES)

AYUDA = """
**Formatos aceptados**
- **FITS de Skipper-CCD** (recomendado): una extensión por amplificador. Se calibra con el overscan y los
  encabezados `NBINCOL`, `CCDNPRES` y `CCDNCOL`. Con ganancia 0, se estima del pico de 1 electrón.
  Da la clasificación y la energía de cada traza.
- **PNG, JPG, TIFF (incluso 16 bits), BMP, WEBP y PDF**: se convierten a un mapa de *pseudo-electrones*
  (fondo y ruido medidos en la propia imagen, intensidad reescalada). La clasificación por forma funciona,
  pero **la energía no se puede medir**. De los PDF se extraen las imágenes incrustadas o, si no hay,
  se renderiza la página.

**Cómo obtener buenos resultados con imágenes**
- Mejor la imagen cruda (una traza = píxeles brillantes sobre fondo oscuro, o al revés) que una figura.
- Si es una figura con ejes (matplotlib, etc.), *Recortar ejes* detecta cada panel y descarta ejes, textos
  y barras de color. Con varios paneles, cada uno se analiza por separado.
- Las reglas de forma usan longitudes en píxeles del CCD. Si la imagen está ampliada o reducida, indicá la
  *escala* (píxeles de imagen por píxel del CCD); por ejemplo 2 si cada píxel del CCD ocupa 2×2 píxeles.
- Si las columnas del CCD estaban agrupadas (binning), indicalo.
- Evitá subir imágenes con anotaciones (cajas, flechas, texto sobre los datos): se detectan como trazas.

**Métodos**
- *Automático*: detector YOLO en imágenes con binning de columnas; reglas físicas sin binning.
- *Detector YOLO*: red neuronal entrenada sobre ~100 000 trazas.
- *Reglas físicas*: clasificación directa por forma (largo, ancho, curvatura) y energía.

**Energía (solo FITS):** carga total de la traza × 3.75 eV por par electrón-hueco; subestimada si hay saturación.

**Limitación:** las etiquetas de entrenamiento salen de reglas morfológicas, no de una verdad de campo.
Las alfas son escasas en el entrenamiento, así que el detector casi no las reconoce.
"""


def construir_app():
    with gr.Blocks(title="Identificador de partículas · Skipper-CCD") as app:
        gr.Markdown("# Identificador de partículas en imágenes Skipper-CCD\n"
                    "Subí una o varias imágenes (**FITS**, o también **PNG, JPG, TIFF o PDF**). Vas a recibir cada "
                    "imagen con las trazas encerradas e identificadas, más el conteo de partículas de cada tipo.")
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                archivos = gr.File(label="Imágenes (FITS, PNG, JPG, TIFF, PDF)", file_count="multiple",
                                   file_types=EXTENSIONES)
                metodo = gr.Radio([(v, k) for k, v in METODOS.items()], value="auto", label="Método")
                with gr.Accordion("Opciones para PNG / JPG / PDF", open=False):
                    paneles = gr.Checkbox(value=True, label="Recortar ejes de figuras (detectar paneles)")
                    escala = gr.Number(value=1.0, minimum=0.1, label="Escala: píxeles de imagen por píxel del CCD")
                    binx = gr.Number(value=1, minimum=1, precision=0, label="Binning de columnas del CCD")
                with gr.Accordion("Opciones avanzadas", open=False):
                    conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confianza mínima (solo YOLO)")
                    ganancia = gr.Number(value=0, label="Ganancia en ADU/e⁻ para FITS (0 = automática)", minimum=0)
                boton = gr.Button("Identificar partículas", variant="primary")
                gr.Markdown("**Clases**\n\n" + LEYENDA)
            with gr.Column(scale=3):
                resumen = gr.Markdown()
                with gr.Row():
                    barras = gr.BarPlot(x="Partícula", y="Cantidad", title="Total de partículas",
                                        sort=None, height=260)
                    descarga = gr.File(label="Descargar todo (PNG + CSV + FITS convertidos)")
                conteo = gr.Dataframe(label="Conteo por archivo", interactive=False, wrap=True,
                                      headers=["Archivo", "Tipo", "Método", "Paneles", "Binning"]
                                      + [NOMBRES[c] for c in CLASES] + ["Total"])
                galeria = gr.Gallery(label="Imágenes identificadas", columns=1, height="auto",
                                     object_fit="contain", preview=True)
        with gr.Accordion("Ayuda", open=False):
            gr.Markdown(AYUDA)
        boton.click(analizar_archivos, [archivos, metodo, conf, ganancia, escala, binx, paneles],
                    [resumen, conteo, barras, galeria, descarga])
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puerto", type=int, default=7860)
    ap.add_argument("--red", action="store_true", help="aceptar conexiones de otras PCs de la red local")
    ap.add_argument("--compartir", action="store_true", help="crear un link publico temporal de Gradio")
    ap.add_argument("--modelo", default=MODELO_DEFAULT,
                    help="detector a usar (por defecto el entrenado de listo_para_usar/modelo/)")
    a = ap.parse_args()
    global RUTA_MODELO
    RUTA_MODELO = os.path.abspath(a.modelo)
    if not os.path.exists(RUTA_MODELO):
        raise SystemExit(f"No existe el modelo {RUTA_MODELO}")
    print(f"Modelo: {RUTA_MODELO}")
    modelo()  # cargar el modelo al arrancar, no en la primera consulta
    app = construir_app()
    # una consulta a la vez: la GPU es una sola
    app.queue(default_concurrency_limit=1)
    app.launch(server_name="0.0.0.0" if a.red else "127.0.0.1", server_port=a.puerto,
               share=a.compartir, max_file_size="500mb", theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
