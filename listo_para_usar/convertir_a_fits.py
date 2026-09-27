"""Convierte a FITS los archivos que no lo son, para usarlos con el resto del algoritmo (detectar.py,
revisar_etiquetas.py, app.py) o con cualquier herramienta de FITS (ds9, astropy, ...).

    ..\\.venv\\Scripts\\python.exe convertir_a_fits.py foto.png
    ..\\.venv\\Scripts\\python.exe convertir_a_fits.py "figuras/*.pdf" --escala 2 --sin-paneles
    ..\\.venv\\Scripts\\python.exe convertir_a_fits.py "datos/*.root"

- ROOT (skipper2root, TTree x/y/pix o TH2): FITS con los mismos ADU y encabezados, una extension por
  amplificador. Se calibra despues exactamente igual que el FITS original del run.
- PNG/JPG/TIFF/BMP/WEBP/PDF: una imagen no conserva los ADU del sensor, asi que el FITS resultante NO esta
  calibrado (BUNIT='PSEUDO-E'). La geometria de las trazas es valida; la energia no.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from particulas.imagenes import cargar_imagen, guardar_fits
from particulas.root import es_root, root_a_fits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--escala", type=float, default=1.0, help="pixeles de imagen por pixel del CCD")
    ap.add_argument("--binx", type=int, default=1, help="binning de columnas del CCD")
    ap.add_argument("--sin-paneles", action="store_true", help="no recortar ejes de figuras")
    ap.add_argument("--out", default="convertidos")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in sorted(p for pat in a.files for p in (glob.glob(pat) or [pat])):
        dest = os.path.join(a.out, os.path.splitext(os.path.basename(f))[0] + ".fits")
        if es_root(f):
            root_a_fits(f, dest)
            print(f"{f} -> {dest}  (ROOT: ADU sin calibrar)")
            continue
        amps = cargar_imagen(f, a.escala, a.binx, paneles=not a.sin_paneles)
        guardar_fits(amps, dest, origen=f)
        print(f"{f} -> {dest}  ({len(amps)} panel(es): {', '.join(x.etiqueta for x in amps)})")


if __name__ == "__main__":
    main()
