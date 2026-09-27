"""Detecta y clasifica trazas de particulas con el modelo entrenado (linea de comandos).

    ..\\.venv\\Scripts\\python.exe detectar.py nueva_imagen.fits
    ..\\.venv\\Scripts\\python.exe detectar.py "carpeta/*.fits" foto.png --conf 0.3 --metodo reglas

Acepta FITS de Skipper-CCD y PNG/JPG/TIFF/PDF. Para cada archivo guarda en detecciones/:
    <nombre>.png   imagen con las cajas coloreadas por tipo de particula
    <nombre>.csv   una fila por traza: amplificador, clase, confianza, caja, energia (keV)
Con --reglas tambien dibuja la clasificacion por reglas fisicas (para comparar).
Para una interfaz web (subir archivos desde el navegador) ver app.py.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from particulas.pipeline import METODOS, MODELO_DEFAULT, cargar_modelo, guardar_figura, procesar_fits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--modelo", default=MODELO_DEFAULT)
    ap.add_argument("--metodo", choices=list(METODOS), default="auto")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--ganancia", type=float, default=None, help="ADU/e- (si no, se estima del pico de 1 e-)")
    ap.add_argument("--reglas", action="store_true", help="dibujar tambien la clasificacion por reglas")
    ap.add_argument("--out", default="detecciones")
    a = ap.parse_args()

    model = cargar_modelo(a.modelo) if a.metodo != "reglas" else None
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(p for pat in a.files for p in (glob.glob(pat) or [pat]))
    for f in paths:
        base = os.path.basename(f).split(".")[0]
        amps, dets, usado = procesar_fits(f, a.metodo, model, a.conf, a.ganancia)
        filas = [d for ds in dets for d in ds]
        pd.DataFrame(filas).to_csv(os.path.join(a.out, base + ".csv"), index=False)
        guardar_figura(amps, dets, os.path.join(a.out, base + ".png"), f"{os.path.basename(f)}  -  {METODOS[usado]}")
        if a.reglas and usado != "reglas":
            amps_r, dets_r, _ = procesar_fits(f, "reglas", ganancia=a.ganancia)
            guardar_figura(amps_r, dets_r, os.path.join(a.out, base + "_reglas.png"),
                           f"{os.path.basename(f)}  -  {METODOS['reglas']}")
        resumen = pd.Series([d["clase"] for d in filas]).value_counts().to_dict() if filas else {}
        print(f"{base}: {len(filas)} trazas {resumen}")


if __name__ == "__main__":
    main()
