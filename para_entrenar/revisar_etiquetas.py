"""PASO 1 - Aplica la reconstruccion + reglas fisicas a tus imagenes y dibuja las cajas etiquetadas.
Sirve para revisar (y ajustar en criterios.yaml) las reglas ANTES de generar el dataset: las etiquetas de
entrenamiento salen de estas reglas.

    ..\\.venv\\Scripts\\python.exe revisar_etiquetas.py mis_datos --criterios criterios.yaml --max 5

Acepta carpetas, patrones con comodines o archivos: FITS/ROOT de Skipper-CCD y PNG/JPG/TIFF/PDF.
Las figuras quedan en para_entrenar/revision_etiquetas/.
"""
import argparse
import glob
import os
import sys
from collections import Counter

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))

from particulas.core import cargar_criterios
from particulas.pipeline import METODOS, guardar_figura, procesar_archivo
from construir_dataset import expandir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="carpetas, patrones o archivos")
    ap.add_argument("--criterios", default=None, help="archivo de criterios (por defecto los del proyecto)")
    ap.add_argument("--max", type=int, default=10, help="maximo de archivos a revisar")
    ap.add_argument("--out", default=os.path.join(AQUI, "revision_etiquetas"))
    a = ap.parse_args()

    criterios = cargar_criterios(a.criterios)
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(f for pat in a.files for f in (expandir(pat) or glob.glob(pat) or [pat]))[: a.max]
    total = Counter()
    for f in paths:
        amps, dets, _ = procesar_archivo(f, "reglas", criterios=criterios)
        cont = Counter(d["clase"] for ds in dets for d in ds)
        total += cont
        calib = [f"g={amp.ganancia:.0f} ruido={amp.ruido_e:.2f}e" if amp.calibrada else "sin calibrar"
                 for amp in amps]
        print(os.path.basename(f), dict(cont), calib)
        guardar_figura(amps, dets, os.path.join(a.out, os.path.basename(f).split(".")[0] + ".png"),
                       titulo=f"{os.path.basename(f)}  -  {METODOS['reglas']}")
    print("TOTAL", dict(total), f"\nFiguras en {a.out}")


if __name__ == "__main__":
    main()
