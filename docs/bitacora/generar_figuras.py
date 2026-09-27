"""Regenera las figuras de la bitacora a partir de los datos y del modelo actual.

    ..\\..\\.venv\\Scripts\\python.exe generar_figuras.py --datos ..\\..\\..\\datos

Salida: figuras/deteccion_atucha.png, figuras/espectro_atucha.png, figuras/resumen.txt (numeros citados).
"""
import argparse
import glob
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(AQUI)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from particulas.core import CLASES
from particulas.dibujo import COLORES, dibujar
from particulas.instrumentos import ENERGIA_BLOB_EV, LINEAS_CU_KEV, detectar_instrumento, masa_amps
from particulas.pipeline import cargar_modelo, figura, procesar_archivo

NOMBRES = {"muon": "Muón", "electron": "Electrón", "alfa": "Alfa", "puntual": "Puntual", "artefacto": "Artefacto"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default=os.path.join(AQUI, "..", "..", "..", "datos"))
    ap.add_argument("--n", type=int, default=30, help="imagenes de Atucha-II para el espectro")
    a = ap.parse_args()
    out = os.path.join(AQUI, "figuras")
    os.makedirs(out, exist_ok=True)

    archivos = sorted(glob.glob(os.path.join(a.datos, "proc_corr_proc", "*.fits")))[: a.n]
    if not archivos:
        sys.exit("No se encontraron datos de Atucha-II en " + a.datos)
    model = cargar_modelo()
    filas, ruidos, ganancias, masas = [], [], [], []
    for i, f in enumerate(archivos):
        amps, dets, usado = procesar_archivo(f, "yolo", model)
        ruidos.append([x.ruido_e for x in amps]); ganancias.append([x.ganancia for x in amps])
        masas.append(masa_amps(amps, detectar_instrumento(f)))
        filas += [d for ds in dets for d in ds]
        if i == 0:
            # figura de ejemplo: amplificadores 0 y 1 (los de ruido nominal)
            # sin textos por caja (ilegibles al tamano de la pagina): solo cajas coloreadas + leyenda
            paneles = [(amp, [(d["x0"], d["y0"], d["x1"], d["y1"], d["clase"], "") for d in ds])
                       for amp, ds in zip(amps[:2], dets[:2])]
            fig = dibujar(paneles, titulo="")
            fig.set_size_inches(10, 5.4)
            fig.savefig(os.path.join(out, "deteccion_atucha.png"), dpi=200); plt.close(fig)
    df = pd.DataFrame(filas)

    bins = np.logspace(np.log10(0.2), np.log10(3e4), 80)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for c in CLASES:
        e = df.loc[df["clase"] == c, "energia_kev"]
        if len(e):
            ax.hist(e, bins=bins, histtype="step", lw=1.4, color=COLORES[c], label=f"{NOMBRES[c]} ({len(e)})")
    for n, e in LINEAS_CU_KEV.items():
        ax.axvline(e, color="0.35", ls="--", lw=0.8)
    ax.text(LINEAS_CU_KEV["Cu Kα"], 0.98, " Cu K", va="top", fontsize=7, color="0.3", transform=ax.get_xaxis_transform())
    ax.axvline(ENERGIA_BLOB_EV / 1000, color="#0a84ff", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("energía depositada [keV]"); ax.set_ylabel("trazas por bin")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "espectro_atucha.png"), dpi=200); plt.close(fig)

    # numeros que cita la bitacora
    mu = df.loc[df["clase"] == "muon", "energia_kev"]
    h, e = np.histogram(mu, bins=np.logspace(1, 4, 60))
    pico_mu = np.sqrt(e[np.argmax(h)] * e[np.argmax(h) + 1])
    ru, ga = np.array(ruidos), np.array(ganancias)
    with open(os.path.join(out, "resumen.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"imagenes: {len(archivos)}\n")
        fh.write(f"trazas: {len(df)}  por imagen: {len(df) / len(archivos):.0f}\n")
        fh.write("por clase: " + str(df["clase"].value_counts().to_dict()) + "\n")
        fh.write("por imagen: " + str((df["clase"].value_counts() / len(archivos)).round(1).to_dict()) + "\n")
        fh.write(f"ruido medio por amp [e]: {np.round(ru.mean(0), 3).tolist()}  std: {np.round(ru.std(0), 3).tolist()}\n")
        fh.write(f"ganancia media por amp [ADU/e]: {np.round(ga.mean(0), 0).tolist()}  std: {np.round(ga.std(0), 0).tolist()}\n")
        fh.write(f"masa activa [g]: {np.mean(masas):.4f}\n")
        fh.write(f"pico del espectro de muones [keV]: {pico_mu:.0f}\n")
        fh.write(f"mediana energia muones [keV]: {mu.median():.0f}\n")
    print(open(os.path.join(out, "resumen.txt"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
