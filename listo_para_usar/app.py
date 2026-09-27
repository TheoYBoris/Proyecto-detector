"""Applet web: los usuarios suben imagenes de Skipper-CCD (FITS, ROOT, o PNG/JPG/TIFF/PDF) y reciben las
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
import threading
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import pandas as pd

from particulas.core import CLASES, DESCRIPCION, Params, cargar_criterios
from particulas.dibujo import COLORES
from particulas.imagenes import FORMATOS, es_imagen, guardar_fits
from particulas.instrumentos import (ENERGIA_BLOB_EV, INSTRUMENTOS, LINEAS_CU_KEV, RANGO_ALFA_MEV,
                                     detectar_instrumento, masa_amps)
from particulas.pipeline import METODOS, MODELO_DEFAULT, cargar_modelo, guardar_figura, procesar_archivo
from particulas.root import EXTENSIONES_ROOT, es_root, root_a_fits

NOMBRES = {"muon": "Muón", "electron": "Electrón", "alfa": "Alfa", "puntual": "Puntual", "artefacto": "Artefacto"}
EXTENSIONES = [".fits", ".fit", ".fts", ".fz", ".gz"] + sorted(EXTENSIONES_ROOT) + sorted(FORMATOS)

RUTA_MODELO = MODELO_DEFAULT
CRITERIOS = None          # Params de la reconstruccion y las reglas (--criterios); None = los del proyecto
NOTA_CONFIG = ""          # se muestra bajo el titulo cuando el modelo o los criterios no son los del proyecto
MODELO = None
_CANDADO = threading.Lock()


def modelo():
    """Carga el detector una sola vez. Arranca en segundo plano al abrir el applet; si alguien analiza antes
    de que termine, espera aca."""
    global MODELO
    with _CANDADO:
        if MODELO is None:
            import numpy as np
            m = cargar_modelo(RUTA_MODELO)
            # la primera prediccion inicializa la GPU (~3 s): mejor hacerla antes que el primer usuario
            m.predict(np.zeros((64, 64, 3), np.uint8), verbose=False)
            MODELO = m
    return MODELO


def espectro(filas, ruta_png):
    """Espectro de energia de las trazas (solo FITS calibrados), con referencias de los papers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    df = pd.DataFrame(filas)
    if df.empty or df["energia_kev"].notna().sum() == 0:
        return None
    df = df[df["energia_kev"] > 0]
    bins = np.logspace(np.log10(0.05), np.log10(3e4), 90)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for c in CLASES:
        e = df.loc[df["clase"] == c, "energia_kev"]
        if len(e):
            ax.hist(e, bins=bins, histtype="step", lw=1.6, color=COLORES[c], label=f"{NOMBRES[c]} ({len(e)})")
    ax.set_xscale("log"); ax.set_yscale("log")
    for n, e in LINEAS_CU_KEV.items():
        ax.axvline(e, color="0.4", ls="--", lw=0.9)
        ax.text(e, 0.97, f" {n}", rotation=90, va="top", fontsize=7, color="0.3",
                transform=ax.get_xaxis_transform())
    ax.axvline(ENERGIA_BLOB_EV / 1000, color="#0a84ff", ls=":", lw=1)
    ax.text(ENERGIA_BLOB_EV / 1000, 0.97, " 600 eV: difusión | blob", rotation=90, va="top", fontsize=7,
            color="#0a84ff", transform=ax.get_xaxis_transform())
    ax.axvspan(RANGO_ALFA_MEV[0] * 1e3, RANGO_ALFA_MEV[1] * 1e3, color="#ffcc00", alpha=0.12, lw=0)
    ax.set_xlabel("energía depositada [keV]  (3.75 eV por par e⁻-hueco)")
    ax.set_ylabel("trazas por bin")
    ax.set_title("Espectro de energía — las líneas del Cu (8.05 / 8.91 keV) sirven para verificar la calibración",
                 fontsize=9)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(ruta_png, dpi=130); plt.close(fig)
    return ruta_png


def analizar_archivos(archivos, metodo, conf, ganancia, escala, binx, paneles, instrumento, exposicion,
                      umbral_ev=0, progress=gr.Progress()):
    if not archivos:
        raise gr.Error("Subí al menos un archivo (FITS, ROOT, PNG, JPG, TIFF o PDF).")
    salida = tempfile.mkdtemp(prefix="particulas_")
    galeria, filas_conteo, filas_tasas, todas, errores, avisos = [], [], [], [], [], []
    notas_exp = set()
    hubo_imagenes = False
    t0 = time.time()
    for i, f in enumerate(archivos):
        ruta = f if isinstance(f, str) else f.name
        nombre = os.path.basename(ruta)
        base = nombre.rsplit(".", 1)[0].replace(".fits", "")
        progress(i / len(archivos), desc=f"Analizando {nombre}")
        imagen = es_imagen(ruta)
        inst = INSTRUMENTOS.get(instrumento, INSTRUMENTOS["auto"])
        if inst.clave == "auto":
            inst = INSTRUMENTOS["generico"] if imagen else detectar_instrumento(ruta)
        # binning de imagenes: el indicado, o el tipico del instrumento elegido
        bx = int(binx) if binx and binx >= 1 else inst.binx
        try:
            amps, dets, usado = procesar_archivo(
                ruta, metodo, modelo() if metodo != "reglas" else None, conf,
                ganancia if ganancia and ganancia > 0 else None,
                escala=float(escala or 1.0), binx=bx, paneles=bool(paneles),
                umbral_e=(float(umbral_ev) / (inst.ev_por_e)) if umbral_ev else None, criterios=CRITERIOS)
        except Exception as e:  # archivo corrupto, formato no reconocido, sin imagenes, etc.
            # no mostrar rutas internas del servidor: dejar solo el nombre del archivo
            detalle = re.sub(r"(?:[A-Za-z]:)?[\\/](?:[^\\/\n'\"]*[\\/])+", "", str(e))
            errores.append(f"**{nombre}**: no se pudo procesar — formato no reconocido o archivo dañado "
                           f"({type(e).__name__}: {detalle[:200]})")
            continue
        hubo_imagenes |= imagen
        filas = [dict(archivo=nombre, **d) for ds in dets for d in ds]
        todas += filas
        cuenta = pd.Series([d["clase"] for d in filas], dtype=object).value_counts()
        sub = pd.Series([d["subclase"] for d in filas if d["subclase"]], dtype=object).value_counts()
        fila = {"Archivo": nombre, "Instrumento": inst.nombre.split(" (")[0],
                "Tipo": "Imagen (sin calibrar)" if imagen else ("ROOT calibrado" if es_root(ruta) else "FITS calibrado"),
                "Método": METODOS[usado].split(" (")[0], "Paneles": len(amps), "Binning": f"x{amps[0].binx}"}
        fila.update({NOMBRES[c]: int(cuenta.get(c, 0)) for c in CLASES})
        fila["Blob (>600 eV)"] = int(sub.get("blob", 0))
        fila["Difusión (<600 eV)"] = int(sub.get("difusion", 0))
        fila["Total"] = len(filas)
        filas_conteo.append(fila)

        if not imagen:
            # ruido medido vs. publicado para el instrumento (p.ej. Atucha-II: 2 de 4 cuadrantes a 0.17 e-)
            if inst.ruido_e == inst.ruido_e:   # no NaN
                malos = [f"amp {a.hdu}: {a.ruido_e:.2f} e⁻" for a in amps if a.ruido_e > 1.3 * inst.ruido_e]
                if malos:
                    avisos.append(f"**{nombre}**: ruido mayor al publicado para {inst.nombre.split(' (')[0]} "
                                  f"({inst.ruido_e} e⁻) en {', '.join(malos)}. Esos amplificadores tienen menor "
                                  f"sensibilidad a depósitos de baja energía.")
            # tasas por masa y tiempo de exposicion
            exp_h = float(exposicion) if exposicion and exposicion > 0 else inst.exposicion_h
            masa = masa_amps(amps, inst)
            if exp_h == exp_h and masa > 0:
                if not (exposicion and exposicion > 0):
                    notas_exp.add(f"{inst.nombre.split(' (')[0]}: " + (inst.nota_exposicion or
                                  f"{inst.exposicion_h * 60:.0f} min por imagen (valor típico publicado)."))
                dias = exp_h / 24.0
                t = {"Archivo": nombre, "Masa activa [g]": round(masa, 4), "Exposición [h]": round(exp_h, 3)}
                t.update({f"{NOMBRES[c]} [ev/(g·día)]": round(cuenta.get(c, 0) / (masa * dias), 1) for c in CLASES})
                t["Difusión [ev/(g·día)]"] = round(sub.get("difusion", 0) / (masa * dias), 1)
                filas_tasas.append(t)

        png = os.path.join(salida, base + "_identificado.png")
        guardar_figura(amps, dets, png, titulo=f"{nombre}  -  {METODOS[usado]}")
        pd.DataFrame(filas).to_csv(os.path.join(salida, base + "_detecciones.csv"), index=False)
        if imagen:
            # la imagen convertida queda disponible como FITS para usarla con el resto de las herramientas
            guardar_fits(amps, os.path.join(salida, base + "_convertido.fits"), origen=nombre)
        elif es_root(ruta):
            # el ROOT tambien queda como FITS (mismos ADU y encabezados) para usarlo con otras herramientas
            root_a_fits(ruta, os.path.join(salida, base + "_convertido.fits"))
        resumen = ", ".join(f"{NOMBRES[c]}: {int(cuenta.get(c, 0))}" for c in CLASES if cuenta.get(c, 0))
        galeria.append((png, f"{nombre} — {len(filas)} trazas ({resumen})"))

    if not filas_conteo:
        raise gr.Error("Ningún archivo se pudo procesar. " + " ".join(e.replace("**", "") for e in errores))

    conteo = pd.DataFrame(filas_conteo)
    total = {"Archivo": "TOTAL", "Instrumento": "", "Tipo": "", "Método": "",
             "Paneles": int(conteo["Paneles"].sum()), "Binning": ""}
    for col in [NOMBRES[c] for c in CLASES] + ["Blob (>600 eV)", "Difusión (<600 eV)", "Total"]:
        total[col] = int(conteo[col].sum())
    conteo = pd.concat([conteo, pd.DataFrame([total])], ignore_index=True)
    conteo.to_csv(os.path.join(salida, "conteo_particulas.csv"), index=False)
    pd.DataFrame(todas).to_csv(os.path.join(salida, "todas_las_detecciones.csv"), index=False)
    tasas = pd.DataFrame(filas_tasas)
    if len(tasas):
        tasas.to_csv(os.path.join(salida, "tasas_por_masa.csv"), index=False)
    png_espectro = espectro([d for d in todas if d["energia_kev"] == d["energia_kev"]],
                            os.path.join(salida, "espectro_energia.png"))

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
    if total["Blob (>600 eV)"] + total["Difusión (<600 eV)"]:
        lineas.append(f"Depósitos puntuales (convención CONNIE): **{total['Blob (>600 eV)']} blobs** (> 600 eV) y "
                      f"**{total['Difusión (<600 eV)']} de difusión** (< 600 eV; incluyen los candidatos a CEvNS).")
    if notas_exp:
        lineas.append("ℹ️ **Tasas:** exposición supuesta — " + " ".join(sorted(notas_exp)) +
                      " Si tus imágenes tienen otra exposición, indicala en *Opciones avanzadas*.")
    if avisos:
        lineas.append("ℹ️ " + "\n\nℹ️ ".join(avisos))
    if hubo_imagenes:
        lineas.append("⚠️ **Imágenes PNG/JPG/TIFF/PDF:** no traen los valores crudos del sensor, así que no se "
                      "pueden calibrar. La clasificación usa la forma de las trazas, pero **la energía no está "
                      "disponible**. En figuras con escala saturada, el ruido puede contarse como depósitos "
                      "puntuales. El ZIP incluye cada imagen convertida a FITS.")
    if errores:
        lineas.append("**Archivos con problemas:**\n\n" + "\n".join(f"- {e}" for e in errores))
    progress(1.0)
    return ("\n\n".join(lineas), conteo, barras, galeria, zip_path, png_espectro,
            tasas if len(tasas) else None)


LEYENDA = "\n".join(
    f"- <span style='color:{COLORES[c]};font-weight:700'>■ {NOMBRES[c]}</span>: {DESCRIPCION[c].split(': ', 1)[1]}"
    for c in CLASES)

AYUDA = """
**Formatos aceptados**
- **FITS de Skipper-CCD** (recomendado): una extensión por amplificador. Se calibra con el overscan y los
  encabezados `NBINCOL`, `CCDNPRES` y `CCDNCOL`. Con ganancia 0, se estima del pico de 1 electrón.
  Da la clasificación y la energía de cada traza.
- **ROOT** (CERN): se lee el formato de `skipper2root` (árbol `skPixTree` con x, y, ohdu, pix, y los
  encabezados en `headerTree_N`), cualquier TTree con ramas x, y y valor de píxel, o histogramas 2D (TH2).
  Se calibra igual que un FITS y da la misma clasificación y energía. El ZIP incluye cada ROOT convertido
  a FITS. Los TH2 y TTree sin encabezado se calibran sin overscan; si ya están en electrones, usá ganancia 1.
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
- *Detector YOLO*: las trazas salen de la reconstrucción por píxeles (que separa las partículas que se cruzan y
  no parte los muones) y una red neuronal entrenada sobre ~100 000 trazas clasifica cada una. Las que la red
  no ve las clasifican las reglas (columna `origen` del CSV).
- *Reglas físicas*: clasificación directa por forma (largo, ancho, curvatura) y energía.

**Muones cortos.** Un muón cruza los 675 µm del sensor, así que su traza mide 45·tan(ángulo) píxeles: los que
llegan casi perpendiculares al CCD dejan trazas cortas. Una traza recta de menos de 30 px se cuenta como muón
si su energía es la de una partícula de mínima ionización que cruza todo el espesor (7–30 px sin binning,
15–30 px con binning; por debajo no se distinguen de otros depósitos). Las líneas horizontales cortas sin
difusión vertical (σy < 0.3 px) y sin esa energía se cuentan como artefactos del registro serie.

**Instrumento.** En *Automático* se reconoce por la geometría del CCD en el encabezado FITS. Los perfiles usan
las especificaciones publicadas; todos son Skipper-CCD de 15 µm de píxel y 675 µm de espesor.
- *Atucha-II* (Depaoli et al., JHEP 10 (2024) 155): 6144×1024 px, binning ×10, 300 muestras, ruido 0.17 e⁻
  (2 de los 4 cuadrantes), 53.7 min de lectura por imagen. El CCD está **vertical**, con el eje x de la imagen
  hacia arriba: la mayoría de los muones dejan trazas largas a lo largo de x. En el run 43 hay una imagen de
  limpieza entre imágenes científicas, así que la exposición media por píxel es ~32 min (inferido de los
  tiempos de lectura).
- *CONNIE* (PRL 134 (2025) 071801; Mirthis, ICHEP 2026): 1022×682 px, 400 muestras, ruido 0.15 e⁻.

El perfil se usa para avisar si un amplificador tiene más ruido que el publicado, para calcular la masa
activa y las tasas, y como binning por defecto de las imágenes sin encabezado.

**Energía (solo FITS):** carga total de la traza × 3.75 eV por par electrón-hueco; subestimada si hay
saturación. En el espectro, las líneas de fluorescencia del cobre (Kα 8.05 keV, Kβ 8.91 keV) permiten
verificar la calibración, como hace Atucha-II.

**Depósitos puntuales (convención CONNIE):** *blob* si superan 600 eV (electrones o fotones de baja energía)
y *difusión* si están por debajo (rayos X, gammas de baja energía; incluyen los candidatos a CEvNS).

**Tasas:** eventos / (masa activa × exposición), en eventos por gramo por día. La masa sale de la zona activa
leída (píxeles × binning × 15 µm × 15 µm × 675 µm × 2.329 g/cm³). La exposición es la del perfil del
instrumento, o la que indiques.

**Limitación:** las etiquetas de entrenamiento salen de reglas morfológicas, no de una verdad de campo.
Las alfas son escasas en el entrenamiento, así que el detector casi no las reconoce.
"""


def construir_app():
    with gr.Blocks(title="Identificador de partículas · Skipper-CCD") as app:
        gr.Markdown("# Identificador de partículas en imágenes Skipper-CCD\n"
                    "Subí una o varias imágenes (**FITS** o **ROOT**, o también **PNG, JPG, TIFF o PDF**). Vas a recibir cada "
                    "imagen con las trazas encerradas e identificadas, más el conteo de partículas de cada tipo."
                    + (f"\n\n{NOTA_CONFIG}" if NOTA_CONFIG else ""))
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                archivos = gr.File(label="Imágenes (FITS, ROOT, PNG, JPG, TIFF, PDF)", file_count="multiple",
                                   file_types=EXTENSIONES)
                metodo = gr.Radio([(v, k) for k, v in METODOS.items()], value="auto", label="Método")
                instrumento = gr.Dropdown([(v.nombre, k) for k, v in INSTRUMENTOS.items()], value="auto",
                                          label="Instrumento")
                with gr.Accordion("Opciones para PNG / JPG / PDF", open=False):
                    paneles = gr.Checkbox(value=True, label="Recortar ejes de figuras (detectar paneles)")
                    escala = gr.Number(value=1.0, minimum=0.1, label="Escala: píxeles de imagen por píxel del CCD")
                    binx = gr.Number(value=0, minimum=0, precision=0,
                                     label="Binning de columnas del CCD (0 = el del instrumento)")
                with gr.Accordion("Opciones avanzadas", open=False):
                    conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confianza mínima (solo YOLO)")
                    ganancia = gr.Number(value=0, label="Ganancia en ADU/e⁻ para FITS/ROOT (0 = automática)", minimum=0)
                    exposicion = gr.Number(value=0, minimum=0,
                                           label="Exposición por imagen en horas, para las tasas (0 = la del instrumento)")
                    e_min = (CRITERIOS or Params()).min_energia_e
                    umbral = gr.Dropdown([(f"{e_min * 3.75:.0f} eV ({e_min:.0f} e⁻) — el de los criterios, como el "
                                           "entrenamiento", 0),
                                          ("45 eV (12 e⁻) — umbral de Atucha-II (JHEP 2024)", 45),
                                          ("15 eV (4 e⁻) — umbral de CONNIE (PRL 2025)", 15)],
                                         value=0, label="Energía mínima por evento (FITS)")
                boton = gr.Button("Identificar partículas", variant="primary")
                gr.Markdown("**Clases**\n\n" + LEYENDA)
            with gr.Column(scale=3):
                resumen = gr.Markdown()
                with gr.Row():
                    barras = gr.BarPlot(x="Partícula", y="Cantidad", title="Total de partículas",
                                        sort=None, height=260)
                    descarga = gr.File(label="Descargar todo (PNG + CSV + FITS convertidos desde ROOT/imágenes)")
                conteo = gr.Dataframe(label="Conteo por archivo", interactive=False, wrap=True,
                                      headers=["Archivo", "Instrumento", "Tipo", "Método", "Paneles", "Binning"]
                                      + [NOMBRES[c] for c in CLASES]
                                      + ["Blob (>600 eV)", "Difusión (<600 eV)", "Total"])
                galeria = gr.Gallery(label="Imágenes identificadas", columns=1, height="auto",
                                     object_fit="contain", preview=True)
                with gr.Accordion("Espectro de energía y tasas (FITS calibrados)", open=True):
                    img_espectro = gr.Image(label="Espectro de energía", type="filepath", show_label=False)
                    tasas = gr.Dataframe(label="Tasas por masa activa y tiempo de exposición",
                                         interactive=False, wrap=True)
        with gr.Accordion("Ayuda", open=False):
            gr.Markdown(AYUDA)
        boton.click(analizar_archivos,
                    [archivos, metodo, conf, ganancia, escala, binx, paneles, instrumento, exposicion, umbral],
                    [resumen, conteo, barras, galeria, descarga, img_espectro, tasas])
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puerto", type=int, default=None,
                    help="puerto (por defecto el primero libre desde 7860)")
    ap.add_argument("--red", action="store_true", help="aceptar conexiones de otras PCs de la red local")
    ap.add_argument("--compartir", action="store_true", help="crear un link publico temporal de Gradio")
    ap.add_argument("--modelo", default=MODELO_DEFAULT,
                    help="detector a usar (por defecto el entrenado de listo_para_usar/modelo/)")
    ap.add_argument("--criterios", default=None,
                    help="archivo YAML de criterios (ver para_entrenar/criterios.yaml); por defecto los del proyecto")
    ap.add_argument("--sin-navegador", action="store_true", help="no abrir el navegador al arrancar")
    a = ap.parse_args()
    global RUTA_MODELO, CRITERIOS, NOTA_CONFIG
    RUTA_MODELO = os.path.abspath(a.modelo)
    if not os.path.exists(RUTA_MODELO):
        raise SystemExit(f"No existe el modelo {RUTA_MODELO}")
    try:
        CRITERIOS = cargar_criterios(a.criterios) if a.criterios else None
    except (OSError, ValueError) as e:
        raise SystemExit(f"Error en el archivo de criterios: {e}")
    print(f"Modelo: {RUTA_MODELO}")
    if a.criterios:
        print(f"Criterios: {os.path.abspath(a.criterios)}")
    if RUTA_MODELO != os.path.abspath(MODELO_DEFAULT) or a.criterios:
        NOTA_CONFIG = (f"*Modelo: `{os.path.basename(os.path.dirname(RUTA_MODELO))}/{os.path.basename(RUTA_MODELO)}`"
                       + (f" · Criterios: `{os.path.basename(a.criterios)}`" if a.criterios else "") + "*")
    # el modelo se carga en segundo plano: la pagina aparece antes, y si alguien analiza enseguida, espera
    threading.Thread(target=modelo, daemon=True).start()
    app = construir_app()
    # una consulta a la vez: la GPU es una sola
    app.queue(default_concurrency_limit=1)
    # el navegador lo abre Gradio cuando el servidor ya escucha (abrirlo antes daba "conexion rechazada")
    app.launch(server_name="0.0.0.0" if a.red else "127.0.0.1", server_port=a.puerto,
               inbrowser=not a.sin_navegador, share=a.compartir, max_file_size="500mb", theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
