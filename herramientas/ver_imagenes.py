"""Visor de imagenes Skipper-CCD (FITS con 4 extensiones = 4 amplificadores).

Uso:
    ..\\.venv\\Scripts\\python.exe ver_imagenes.py archivo.fits [...]        # guarda PNG en imagenes_png/
    ..\\.venv\\Scripts\\python.exe ver_imagenes.py archivo.fits --show       # ventana interactiva (zoom/pan)
    ..\\.venv\\Scripts\\python.exe ver_imagenes.py "..\\datos\\proc_corr_proc\\*.fits"   # admite comodines
    ..\\.venv\\Scripts\\python.exe ver_imagenes.py archivo.root               # lee skPixTree de los ROOT crudos
"""
import argparse
import glob
import os

import numpy as np
import matplotlib

OUT_DIR = "imagenes_png"


def load_fits(path):
    from astropy.io import fits
    with fits.open(path) as h:
        return [hdu.data.astype(float) for hdu in h if hdu.data is not None]


def load_root(path):
    import uproot
    with uproot.open(path) as f:
        t = f["skPixTree"].arrays(["x", "y", "ohdu", "pix"], library="np")
    imgs = []
    for ohdu in np.unique(t["ohdu"]):
        m = t["ohdu"] == ohdu
        x, y = t["x"][m], t["y"][m]
        img = np.full((y.max() + 1, x.max() + 1), np.nan)
        img[y, x] = t["pix"][m]
        imgs.append(img)
    return imgs


def plot(path, imgs, lo=1, hi=99.5):
    import matplotlib.pyplot as plt
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), squeeze=False)
    for i, (ax, img) in enumerate(zip(axes[0], imgs)):
        # Restar el pedestal (mediana por fila) y escalar por percentiles para que
        # se vean las trazas de particulas sobre el ruido.
        img = img - np.nanmedian(img, axis=1, keepdims=True)
        vmin, vmax = np.nanpercentile(img, [lo, hi])
        im = ax.imshow(img, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                       interpolation="nearest", aspect="auto")
        ax.set_title(f"Amplificador {i}")
        ax.set_xlabel("columna")
        if i == 0:
            ax.set_ylabel("fila")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(os.path.basename(path), fontsize=10)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--show", action="store_true", help="abrir ventana interactiva en vez de guardar PNG")
    ap.add_argument("--lo", type=float, default=1, help="percentil inferior de la escala de color")
    ap.add_argument("--hi", type=float, default=99.5, help="percentil superior de la escala de color")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = sorted(p for pat in args.files for p in (glob.glob(pat) or [pat]))
    os.makedirs(OUT_DIR, exist_ok=True)
    for p in paths:
        imgs = load_root(p) if p.endswith(".root") else load_fits(p)
        fig = plot(p, imgs, args.lo, args.hi)
        if args.show:
            plt.show()
        else:
            out = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(p))[0] + ".png")
            fig.savefig(out, dpi=110)
            plt.close(fig)
            print(out)


if __name__ == "__main__":
    main()
