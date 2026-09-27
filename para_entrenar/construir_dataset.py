"""PASO 2 - Construye el dataset YOLO a partir de tus imagenes, auto-etiquetadas con las reglas fisicas.

    ..\\.venv\\Scripts\\python.exe construir_dataset.py --datos "C:/mis_datos/run1/*.fits" "C:/mis_datos/run2"

Cada --datos es un GRUPO (una carpeta o un patron con comodines). Acepta FITS de Skipper-CCD y tambien
PNG/JPG/TIFF/PDF (sin calibrar). La separacion train/val se hace por grupo, y los grupos chicos se
repiten en train para que el detector no los ignore. Sin --datos se usan los del experimento original
(datos/201211 y datos/proc_corr_proc).

Cada amplificador/panel es una imagen. Salida (en para_entrenar/dataset/):
    images/{train,val}/*.png
    labels/{train,val}/*.txt     (formato YOLO: clase xc yc w h, normalizado; editables en CVAT/Label Studio)
    data.yaml
    clusters.csv                 (todas las trazas con sus variables fisicas)
"""
import argparse
import glob
import os
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)

import pandas as pd
from PIL import Image

from particulas.core import CLASES, Params, clasificar, encontrar_clusters, imagen_rgb
from particulas.imagenes import FORMATOS
from particulas.pipeline import cargar_amps

MIN_CAJA = 5       # px: las cajas mas pequenas se agrandan (YOLO detecta mal objetos de 1-2 px)
REPETIR_CHICO = 4  # grupos con < 10 % de los archivos del grupo mas grande se repiten en train
REPETIR_ALFA = 8   # imagenes con alfas (muy raras) se repiten en train
EXT_FITS = {".fits", ".fit", ".fts", ".fz"}
DATOS_ORIGINALES = [os.path.join(RAIZ, "datos", "201211", "proc_*.fits"),
                    os.path.join(RAIZ, "datos", "proc_corr_proc", "*.fits")]


def expandir(patron):
    """Carpeta o patron -> lista de archivos soportados."""
    if os.path.isdir(patron):
        patron = os.path.join(patron, "*")
    ok = EXT_FITS | FORMATOS
    return sorted(f for f in glob.glob(patron) if os.path.splitext(f)[1].lower() in ok
                  or f.lower().endswith(".fits.gz"))


def cajas_yolo(cl, h, w):
    lineas = []
    for c in cl:
        x0, y0, x1, y1 = c.x0 - 1, c.y0 - 1, c.x1 + 1, c.y1 + 1
        if x1 - x0 < MIN_CAJA:
            m = (MIN_CAJA - (x1 - x0)) / 2; x0 -= m; x1 += m
        if y1 - y0 < MIN_CAJA:
            m = (MIN_CAJA - (y1 - y0)) / 2; y0 -= m; y1 += m
        x0, x1 = max(0, x0), min(w, x1); y0, y1 = max(0, y0), min(h, y1)
        # la imagen PNG se guarda con la fila 0 arriba -> invertir eje y
        yc = h - (y0 + y1) / 2
        lineas.append(f"{CLASES.index(c.clase)} {(x0 + x1) / 2 / w:.6f} {yc / h:.6f} "
                      f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return lineas


def procesar(args):
    f, split, out, rep, grupo = args
    filas = []
    base = f"g{grupo}_" + os.path.basename(f).split(".")[0]
    try:
        amps = cargar_amps(f)
    except Exception as e:
        print(f"  (se omite {os.path.basename(f)}: {type(e).__name__}: {e})", flush=True)
        return filas
    for amp in amps:
        # imagenes sin calibrar: mismas reglas que en la deteccion (ver pipeline.detectar_reglas)
        p = Params() if amp.calibrada else replace(Params(), alfa_min_energia_e=1.0e6, min_pixeles=3)
        cl = encontrar_clusters(amp, p)
        for c in cl:
            c.clase = clasificar(c, p)
        rgb = imagen_rgb(amp)[::-1]          # origin="lower" -> fila 0 abajo, como en los visores
        h, w = rgb.shape[:2]
        lab = cajas_yolo(cl, h, w)
        for k in range(rep):
            nombre = f"{base}_amp{amp.hdu}" + (f"_r{k}" if k else "")
            Image.fromarray(rgb).save(os.path.join(out, "images", split, nombre + ".png"))
            with open(os.path.join(out, "labels", split, nombre + ".txt"), "w") as fh:
                fh.write("\n".join(lab))
        for c in cl:
            d = c.dict(); d.update(archivo=os.path.basename(f), grupo=grupo, amp=amp.hdu, split=split,
                                   binx=amp.binx, ganancia=amp.ganancia, calibrada=amp.calibrada)
            filas.append(d)
    return filas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", nargs="+", default=None,
                    help="carpetas o patrones (cada uno es un grupo). Por defecto: datos del experimento original")
    ap.add_argument("--out", default=os.path.join(AQUI, "dataset"))
    ap.add_argument("--val", type=float, default=0.15, help="fraccion de archivos de cada grupo para validacion")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()

    grupos = [expandir(p) for p in (a.datos or DATOS_ORIGINALES)]
    for p, g in zip(a.datos or DATOS_ORIGINALES, grupos):
        print(f"grupo: {p} -> {len(g)} archivos")
    grupos = [g for g in grupos if g]
    if not grupos:
        sys.exit("No se encontraron archivos. Revisa las rutas de --datos.")
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    for s in ("train", "val"):
        os.makedirs(os.path.join(a.out, "images", s), exist_ok=True)
        os.makedirs(os.path.join(a.out, "labels", s), exist_ok=True)

    random.seed(0)
    mayor = max(len(g) for g in grupos)
    tareas = []
    for i, grupo in enumerate(grupos):
        rep = REPETIR_CHICO if len(grupo) < 0.1 * mayor else 1
        g = grupo[:]; random.shuffle(g)
        nval = max(1, round(len(g) * a.val)) if len(g) > 1 else 0
        tareas += [(f, "val", a.out, 1, i) for f in g[:nval]]
        tareas += [(f, "train", a.out, rep, i) for f in g[nval:]]
    if not any(t[1] == "val" for t in tareas):
        # YOLO necesita al menos una imagen de validacion: se toma un archivo del grupo mas grande
        k = max(range(len(tareas)), key=lambda j: len(grupos[tareas[j][4]]))
        f, _, out, _, gi = tareas[k]
        tareas[k] = (f, "val", out, 1, gi)
        if len(tareas) == 1:
            tareas.append((f, "train", out, 1, gi))
            print("AVISO: un solo archivo; se usa para entrenar y validar (solo sirve como prueba).")

    filas = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(procesar, tareas, chunksize=4), 1):
            filas += r
            if i % 50 == 0 or i == len(tareas):
                print(f"{i}/{len(tareas)} archivos", flush=True)

    # clases raras (alfas): replicar en train las imagenes que las contienen
    id_alfa = str(CLASES.index("alfa"))
    ldir, idir = os.path.join(a.out, "labels", "train"), os.path.join(a.out, "images", "train")
    for txt in os.listdir(ldir):
        with open(os.path.join(ldir, txt)) as fh:
            if not any(l.split()[0] == id_alfa for l in fh if l.strip()):
                continue
        base = txt[:-4]
        for k in range(1, REPETIR_ALFA):
            shutil.copy(os.path.join(ldir, txt), os.path.join(ldir, f"{base}_a{k}.txt"))
            shutil.copy(os.path.join(idir, base + ".png"), os.path.join(idir, f"{base}_a{k}.png"))

    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(a.out, "clusters.csv"), index=False)
    with open(os.path.join(a.out, "data.yaml"), "w") as fh:
        fh.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\n"
                 f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASES)))
    if len(df):
        print(df.groupby(["split", "clase"]).size().unstack(fill_value=0))
    print(f"\nDataset listo en {a.out}. Siguiente paso: entrenar.py")


if __name__ == "__main__":
    main()
