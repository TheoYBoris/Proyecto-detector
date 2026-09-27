"""Entrada desde imagenes comunes (PNG, JPG, TIFF, BMP) y PDF.

Una imagen exportada NO conserva los valores crudos del sensor (ADU), ni overscan, ni encabezados:
no se puede calibrar en electrones. Lo que si se conserva es la GEOMETRIA de las trazas, que es lo que
usan las reglas y el detector para clasificar. Por eso la imagen se convierte en un mapa de
"pseudo-electrones":

    1. se pasa a escala de grises (luminancia) y, si hace falta, se invierte (trazas oscuras sobre blanco)
    2. si es una figura con ejes (p.ej. matplotlib), se recorta cada panel por dentro de sus ejes
    3. se resta un fondo suave y se mide el ruido de la propia imagen
    4. la intensidad sobre el ruido se lleva a pseudo-electrones con una escala logaritmica:
       fondo -> 0, maximo -> 2e4. Los umbrales de la reconstruccion (20 e semilla, 4 e crecimiento)
       quedan en ~30 % y ~16 % del rango de la imagen.

Las energias resultantes NO son fisicas y no se informan. El resultado puede guardarse como FITS
(BUNIT='PSEUDO-E'), que el resto del algoritmo lee sin recalibrar.
"""
import os

import numpy as np
from astropy.io import fits
from scipy import ndimage as ndi

from .core import Amp

FORMATOS_IMAGEN = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
FORMATOS = FORMATOS_IMAGEN | {".pdf"}
E_MAX = 2.0e4


def es_imagen(ruta):
    return os.path.splitext(ruta)[1].lower() in FORMATOS


# ----------------------------------------------------------------------------- lectura

def _a_gris(arr):
    """Array de PIL/PyMuPDF -> float [0,1] en luminancia."""
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        a = a / 255.0
    elif a.dtype == np.uint16:
        a = a / 65535.0
    else:
        a = a.astype(float)
        rng = np.nanmax(a) - np.nanmin(a)
        a = (a - np.nanmin(a)) / (rng if rng > 0 else 1.0)
    if a.ndim == 3:
        a = a[..., :3]
        a = a @ np.array([0.2126, 0.7152, 0.0722])[: a.shape[2]] if a.shape[2] == 3 else a[..., 0]
    return np.nan_to_num(a.astype(float))


def _pil_a_array(im):
    from PIL import Image
    if im.mode in ("I;16", "I;16B", "I;16L"):
        return np.array(im, dtype=np.uint16)
    if im.mode in ("I", "F"):
        return np.array(im, dtype=np.float64)
    if im.mode in ("RGBA", "LA", "P", "PA"):
        # transparencia -> se compone sobre el color de fondo mas comun para no crear bordes falsos
        im = im.convert("RGBA")
        fondo = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(fondo, im)
    return np.array(im.convert("RGB") if im.mode not in ("L", "RGB") else im)


def leer_rasters(ruta, dpi_pdf=200):
    """Lista de (etiqueta, imagen gris float) de un archivo de imagen o PDF."""
    ext = os.path.splitext(ruta)[1].lower()
    salida = []
    if ext == ".pdf":
        import pymupdf
        with pymupdf.open(ruta) as doc:
            for n, page in enumerate(doc, 1):
                imgs = page.get_images(full=True)
                if imgs:
                    # imagenes incrustadas a su resolucion nativa (mejor que renderizar la pagina)
                    for k, info in enumerate(imgs, 1):
                        pix = pymupdf.Pixmap(doc, info[0])
                        if pix.n - pix.alpha >= 4:          # CMYK -> RGB
                            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                        a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
                        if pix.alpha:
                            a = a[..., :-1]
                        if min(a.shape[:2]) >= 32:
                            salida.append((f"pag{n}_img{k}", _a_gris(a)))
                else:
                    pix = page.get_pixmap(dpi=dpi_pdf, colorspace=pymupdf.csRGB, alpha=False)
                    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3)
                    salida.append((f"pag{n}", _a_gris(a)))
    else:
        from PIL import Image, ImageSequence
        with Image.open(ruta) as im:
            frames = list(ImageSequence.Iterator(im))
            for k, fr in enumerate(frames, 1):
                salida.append((f"cuadro{k}" if len(frames) > 1 else "", _a_gris(_pil_a_array(fr.copy()))))
    if not salida:
        raise ValueError("no se encontraron imagenes en el archivo")
    return salida


# ----------------------------------------------------------------------------- paneles de figuras

def detectar_paneles(g):
    """Rectangulos (r0, r1, c0, c1) dentro de los ejes de cada panel de una figura. [] si no hay ejes."""
    H, W = g.shape
    # solo figuras: tienen un marco exterior uniforme (fondo de la figura). Una imagen de datos
    # crudos tiene ruido en los bordes y no se recorta.
    b = max(2, int(0.02 * min(H, W)))
    borde = np.concatenate([g[:b].ravel(), g[-b:].ravel(), g[:, :b].ravel(), g[:, -b:].ravel()])
    moda = np.median(borde)
    if np.mean(np.abs(borde - moda) < 0.02) < 0.6:
        return []
    # cada panel (imagen + su marco de ejes) es una region grande conexa distinta del fondo de la
    # figura; textos y numeros de los ejes son regiones chicas; la barra de color es muy alargada
    lab, n = ndi.label(np.abs(g - moda) > 0.05, structure=np.ones((3, 3)))
    paneles = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h * w < 0.04 * H * W or max(h, w) / max(1, min(h, w)) > 6:
            continue
        # un panel es un rectangulo lleno de datos; la caja de una traza grande esta casi vacia
        if np.mean(lab[sl] == i) < 0.5:
            continue
        r0, r1, c0, c1 = _ajustar_a_datos(g, moda, sl[0].start, sl[0].stop, sl[1].start, sl[1].stop)
        if r1 - r0 >= 16 and c1 - c0 >= 16:
            paneles.append((r0, r1, c0, c1))
    return sorted(paneles, key=lambda p: (p[0] // max(1, H // 4), p[2]))


def _es_borde(v, moda):
    """Fila/columna que no es dato: mayormente fondo de figura, o casi constante (linea de eje)."""
    return np.mean(np.abs(v - moda) < 0.05) > 0.5 or np.std(v) < 0.04 or np.mean(np.abs(np.diff(v)) < 0.02) > 0.8


def _ajustar_a_datos(g, moda, r0, r1, c0, c1, max_paso=12):
    """Mueve cada borde de la caja hacia adentro saltando marcas de ticks y lineas de eje."""
    for _ in range(max_paso):
        cambio = False
        if _es_borde(g[r0, c0:c1], moda): r0 += 1; cambio = True
        if _es_borde(g[r1 - 1, c0:c1], moda): r1 -= 1; cambio = True
        if _es_borde(g[r0:r1, c0], moda): c0 += 1; cambio = True
        if _es_borde(g[r0:r1, c1 - 1], moda): c1 -= 1; cambio = True
        if not cambio or r1 - r0 < 16 or c1 - c0 < 16:
            break
    return r0, r1, c0, c1


# ----------------------------------------------------------------------------- pseudo-electrones

def _fondo_suave(g, bloque=32):
    H, W = g.shape
    if min(H, W) < 4 * bloque:
        return np.full_like(g, np.median(g))
    hb, wb = H // bloque, W // bloque
    med = np.median(g[: hb * bloque, : wb * bloque].reshape(hb, bloque, wb, bloque), axis=(1, 3))
    med = ndi.median_filter(med, size=3)
    return ndi.zoom(med, (H / hb, W / wb), order=1)[:H, :W]


def a_pseudo_electrones(g, escala=1.0):
    """Imagen gris [0,1] -> (mapa de pseudo-electrones, ruido estimado, invertida?)."""
    if escala and abs(escala - 1.0) > 1e-3:
        from PIL import Image
        h, w = g.shape
        nuevo = (max(8, round(w / escala)), max(8, round(h / escala)))
        g = np.asarray(Image.fromarray(g.astype(np.float32), mode="F").resize(nuevo, Image.BOX), float)
    bg = np.median(g)
    # la senal es la cola mas extendida: trazas claras sobre oscuro o oscuras sobre claro
    arriba, abajo = np.percentile(g, 99.7) - bg, bg - np.percentile(g, 0.3)
    invertida = abajo > arriba
    if invertida:
        g = 1.0 - g
    r = g - _fondo_suave(g)
    neg = r[r < 0]
    sigma = 1.4826 * np.median(np.abs(neg)) if neg.size > 100 else 0.0
    sigma = max(sigma, 0.5 / 255)                  # imagenes renderizadas sin ruido
    tope = max(np.percentile(r, 99.99), r.max() * 0.95, 6 * sigma)
    norm = np.clip((r - 3 * sigma) / (tope - 3 * sigma), 0, 1)
    return 10 ** (np.log10(E_MAX + 1) * norm) - 1, sigma, invertida


def cargar_imagen(ruta, escala=1.0, binx=1, paneles=True):
    """Archivo de imagen/PDF -> lista de Amp (uno por panel/pagina) en pseudo-electrones."""
    amps = []
    for etiqueta, g in leer_rasters(ruta):
        recortes = detectar_paneles(g) if paneles else []
        trozos = [g[r0:r1, c0:c1] for r0, r1, c0, c1 in recortes] or [g]
        for k, t in enumerate(trozos):
            e, sigma, inv = a_pseudo_electrones(t, escala)
            # PNG: fila 0 arriba. El resto del codigo usa fila 0 abajo (como FITS)
            nombre = " ".join(x for x in (etiqueta, f"panel{k + 1}" if len(trozos) > 1 else "") if x)
            amps.append(Amp(archivo=ruta, hdu=len(amps), electrones=e[::-1].copy(), binx=int(binx), x0=0,
                            ganancia=float("nan"), ruido_e=float("nan"), calibrada=False,
                            etiqueta=nombre or f"imagen{len(amps) + 1}"))
    return amps


# ----------------------------------------------------------------------------- FITS convertido

def guardar_fits(amps, ruta_fits, origen=""):
    """Guarda los mapas de pseudo-electrones como FITS (una extension por panel)."""
    hdus = []
    for i, a in enumerate(amps):
        h = fits.Header()
        h["BUNIT"] = ("PSEUDO-E", "pseudo-electrones: NO calibrado")
        h["NBINCOL"] = (a.binx, "binning de columnas")
        h["CCDNPRES"] = (0, "sin prescan")
        # sin comentario: los nombres de archivo largos no entran en una tarjeta FITS de 80 caracteres
        h["ORIGEN"] = os.path.basename(origen or a.archivo)[:68]
        h["PANEL"] = (a.etiqueta[:40], "pagina/panel de origen")
        h["COMMENT"] = "Convertido desde imagen: geometria valida, energia NO fisica."
        datos = a.electrones.astype(np.float32)
        hdus.append(fits.PrimaryHDU(datos, h) if i == 0 else fits.ImageHDU(datos, h))
    fits.HDUList(hdus).writeto(ruta_fits, overwrite=True)
    return ruta_fits
