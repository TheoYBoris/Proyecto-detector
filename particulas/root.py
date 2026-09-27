"""Entrada desde archivos ROOT (CERN).

Se reconocen tres formas de guardar una imagen de CCD en ROOT, en este orden:

    1. skipper2root (formato estandar de los Skipper-CCD, p.ej. los darks de 201211/):
         skPixTree       x, y, ohdu, pix   -> un pixel por entrada (promedio de las NSAMP muestras, en ADU)
         skTablePixTree  x, y, pix[n]      -> alternativa si no hay skPixTree (una columna por amplificador)
         headerTree_k    el encabezado FITS de la extension k, una rama por palabra clave
    2. Un TTree cualquiera con ramas x, y y un valor de pixel (pix/charge/val/adc/value/...), y opcionalmente
       una rama de amplificador (ohdu/hdu/amp/ext).
    3. Histogramas 2D (TH2F/TH2D/TH2I/...): cada histograma es una imagen.

El resultado es un HDUList de FITS en memoria (una extension por amplificador, con su encabezado), asi que
la calibracion (pedestal de overscan, ganancia del pico de 1 e-) es la misma que para un FITS. El ROOT de
skipper2root y el FITS del mismo run dan exactamente la misma imagen.

Los histogramas y los TTree genericos no traen encabezado: se calibran sin overscan (pedestal de la zona
activa) y sin binning. Si ya estan en electrones, usar ganancia = 1.
"""
import os
import warnings

import numpy as np
from astropy.io import fits

EXTENSIONES_ROOT = {".root"}

RAMAS_X = ("x", "col", "ix")
RAMAS_Y = ("y", "row", "iy")
RAMAS_VALOR = ("pix", "charge", "val", "value", "adc", "ene", "e", "q", "signal")
RAMAS_AMP = ("ohdu", "hdu", "amp", "ext", "ccd")
# palabras clave que astropy genera a partir de los datos (en skipper2root describen las muestras crudas)
CLAVES_ESTRUCTURA = {"", "SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND", "XTENSION",
                     "PCOUNT", "GCOUNT", "COMMENT", "HISTORY", "END"}


def es_root(ruta):
    return isinstance(ruta, str) and os.path.splitext(ruta)[1].lower() in EXTENSIONES_ROOT


# ----------------------------------------------------------------------------- encabezados

def _encabezados(f):
    """{indice: fits.Header} desde los headerTree_k de skipper2root."""
    salida = {}
    for nombre in f.keys(cycle=False):
        if not nombre.startswith("headerTree_"):
            continue
        try:
            k = int(nombre.rsplit("_", 1)[1])
            arr = f[nombre].arrays(library="np")
        except Exception:
            continue
        h = fits.Header()
        with warnings.catch_warnings():
            # las claves de mas de 8 letras (DATESTART, DELAY_...) pasan a tarjetas HIERARCH
            warnings.simplefilter("ignore")
            for clave, v in arr.items():
                if clave.upper() in CLAVES_ESTRUCTURA or not len(v):
                    continue
                valor = v[0].decode(errors="replace") if isinstance(v[0], bytes) else str(v[0])
                try:
                    h[clave] = valor.strip()[:60]
                except (ValueError, KeyError):
                    pass
        salida[k] = h
    return salida


# ----------------------------------------------------------------------------- imagenes

def _rama(ramas, opciones):
    nombres = {b.lower(): b for b in ramas}
    return next((nombres[o] for o in opciones if o in nombres), None)


def _es_escalar(tree, rama):
    """Un numero por entrada (pix) o un arreglo (pix[4] de skTablePixTree)."""
    try:
        return np.asarray(tree.arrays([rama], entry_stop=1, library="np")[rama]).ndim == 1
    except Exception:
        return False


def _arbol_de_pixeles(f):
    """(arbol, rama x, rama y, rama valor, rama amplificador o None, es tabla) o None.
    Acepta TTree y RNTuple (el formato que reemplaza a TTree desde ROOT 6.34)."""
    arboles = {n: f[n] for n, c in f.classnames(cycle=False).items() if c in ("TTree", "ROOT::RNTuple")}
    orden = sorted(arboles, key=lambda n: (n != "skPixTree", n != "skTablePixTree", n))
    for n in orden:
        t = arboles[n]
        ramas = list(t.keys())
        bx, by, bv = _rama(ramas, RAMAS_X), _rama(ramas, RAMAS_Y), _rama(ramas, RAMAS_VALOR)
        if not (bx and by and bv):
            continue
        if _es_escalar(t, bv):
            return t, bx, by, bv, _rama(ramas, RAMAS_AMP), False
        # skTablePixTree: pix[n] con un valor por amplificador
        return t, bx, by, bv, None, True
    return None


def _imagenes_de_arbol(t, bx, by, bv, bamp, tabla):
    """{amplificador: imagen 2D (NaN donde no hay pixel)} leyendo el arbol por partes."""
    ramas = [bx, by, bv] + ([bamp] if bamp else [])
    # 1ra pasada: tamano de cada imagen (solo coordenadas)
    tam = {}
    for ch in t.iterate([bx, by] + ([bamp] if bamp else []), library="np", step_size="200 MB"):
        amps = ch[bamp] if bamp else np.zeros(len(ch[bx]), int)
        for a in np.unique(amps):
            m = amps == a
            ny, nx = int(ch[by][m].max()) + 1, int(ch[bx][m].max()) + 1
            py, px = tam.get(int(a), (0, 0))
            tam[int(a)] = (max(py, ny), max(px, nx))
    if not tam:
        return {}
    # 2da pasada: llenar
    imgs = {}
    for ch in t.iterate(ramas, library="np", step_size="200 MB"):
        x, y, v = ch[bx].astype(int), ch[by].astype(int), ch[bv]
        if tabla:
            v = np.asarray(v, float).reshape(len(x), -1)
            for a in range(v.shape[1]):
                img = imgs.setdefault(a, np.full(tam[0], np.nan))
                img[y, x] = v[:, a]
            continue
        amps = ch[bamp].astype(int) if bamp else np.zeros(len(x), int)
        for a in np.unique(amps):
            m = amps == a
            img = imgs.setdefault(int(a), np.full(tam[int(a)], np.nan))
            img[y[m], x[m]] = v[m]
    return imgs


def _histogramas(f):
    """[(nombre, imagen 2D)] de los TH2 del archivo (fila 0 = primer bin de y, como en FITS)."""
    salida = []
    for n, c in f.classnames(cycle=False).items():
        if c.startswith("TH2") and "Poly" not in c:
            try:
                salida.append((n, np.asarray(f[n].values(flow=False), float).T))
            except Exception:
                continue
    return salida


def leer_root(ruta):
    """ROOT -> fits.HDUList en memoria (una extension por amplificador)."""
    import uproot
    with uproot.open(ruta) as f:
        heads = _encabezados(f)
        base = _arbol_de_pixeles(f)
        imagenes = []
        if base:
            imgs = _imagenes_de_arbol(*base)
            # skipper2root numera ohdu desde 1 y headerTree desde 0: se aparean por orden
            imagenes = [(f"ohdu {a}", imgs[a]) for a in sorted(imgs)]
        if not imagenes:
            imagenes = _histogramas(f)
    if not imagenes:
        raise ValueError("el archivo ROOT no tiene imagenes reconocibles (skPixTree, TTree x/y/pix o TH2)")
    hdus = []
    for k, (nombre, img) in enumerate(imagenes):
        h = heads.get(k, fits.Header()).copy()
        h["ROOTOBJ"] = (nombre[:60], "objeto ROOT de origen")
        datos = img.astype(np.float32)
        hdus.append(fits.PrimaryHDU(datos, h) if k == 0 else fits.ImageHDU(datos, h))
    return fits.HDUList(hdus)


def leer_encabezado_root(ruta):
    """Encabezado principal (headerTree_0) sin leer los pixeles. Vacio si el archivo no lo trae."""
    import uproot
    with uproot.open(ruta) as f:
        return _encabezados(f).get(0, fits.Header())


def root_a_fits(ruta, ruta_fits):
    """Guarda el ROOT como FITS (mismos ADU y encabezados), legible por cualquier herramienta de FITS."""
    hdul = leer_root(ruta)
    hdul[0].header["ORIGEN"] = os.path.basename(ruta)[:68]
    hdul[0].header["COMMENT"] = "Convertido desde ROOT: ADU sin calibrar, igual que el FITS original."
    hdul.writeto(ruta_fits, overwrite=True)
    return ruta_fits
