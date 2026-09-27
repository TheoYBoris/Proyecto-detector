"""Calibracion, reconstruccion de clusters y clasificacion morfologica para Skipper-CCD.

Flujo para cada extension (amplificador) de un FITS:
    ADU --(pedestal de overscan, ganancia del pico de 1 e-)--> electrones
    electrones --(umbral con histeresis)--> clusters (una traza = un cluster)
    cluster --(geometria en pixeles fisicos + energia)--> clase

La geometria se calcula en pixeles FISICOS: si la imagen tiene binning de columnas
(NBINCOL > 1, p.ej. el run 43 con NBINCOL=10) la coordenada x se multiplica por el binning.
"""
from dataclasses import dataclass, field, asdict

import numpy as np
from astropy.io import fits
from scipy import ndimage as ndi

EV_POR_ELECTRON = 3.75          # energia media para crear un par e-h en Si (eV)
PIXEL_UM = 15.0                  # tamano de pixel (um)

CLASES = ["muon", "electron", "alfa", "puntual", "artefacto"]
DESCRIPCION = {
    "muon": "muon: traza larga y recta (MIP que atraviesa el CCD)",
    "electron": "electrón: traza curva/irregular ('gusano', Compton o beta)",
    "alfa": "alfa: mancha compacta, redonda y muy energética (MeV)",
    "puntual": "puntual: depósito pequeño limitado por difusión (rayos X / baja energía)",
    "artefacto": "artefacto: línea de 1 fila/columna (registro serie, columna caliente)",
}


@dataclass
class Params:
    semilla_e: float = 20.0      # un cluster necesita al menos un pixel >= semilla_e
    crecer_e: float = 4.0        # y se extiende por pixeles vecinos >= crecer_e
    min_energia_e: float = 60.0  # clusters con menos carga total se descartan
    min_pixeles: int = 1         # clusters con menos pixeles se descartan (imagenes sin calibrar: 3)
    # --- reglas de clasificacion (unidades: pixeles fisicos, electrones) ---
    muon_min_largo: float = 30.0
    muon_max_ancho_rel: float = 0.10   # ancho/largo maximo para considerarse recto
    muon_min_rectitud: float = 0.75     # distancia extremos / longitud del esqueleto
    muon_max_curvatura: float = 0.03    # sagita / largo
    alfa_min_energia_e: float = 2.67e5  # ~1 MeV (alfas de U/Th: 4-8 MeV, pero el nucleo satura)
    alfa_min_redondez: float = 0.5      # lado menor / lado mayor de la caja fisica
    alfa_max_tamano: float = 60.0       # px fisicos; un electron de 1 MeV recorre ~2 mm (>100 px)
    puntual_max_largo: float = 7.0
    artefacto_min_largo: float = 12.0


# ----------------------------------------------------------------------------- calibracion

def _hint(h, k, default=None):
    try:
        return int(str(h.get(k)).strip())
    except (TypeError, ValueError):
        return default


@dataclass
class Amp:
    """Una extension del FITS ya calibrada."""
    archivo: str
    hdu: int
    electrones: np.ndarray       # imagen en e- (NaN/columnas malas -> 0)
    binx: int                    # binning de columnas
    x0: int                      # primera columna activa (en la imagen original)
    ganancia: float              # ADU / e-
    ruido_e: float               # sigma de lectura en e-
    columnas_malas: list = field(default_factory=list)
    calibrada: bool = True       # False: viene de una imagen PNG/JPG/PDF (pseudo-electrones)
    etiqueta: str = ""           # nombre del panel/pagina (imagenes)
    filas_activas: int = 0       # filas fisicas del cuadrante (CCDNROW/2); el resto es overscan vertical


def _modelo_poisson(x, N, lam, g, s, mu):
    """Picos de n electrones: gaussianas de ancho s en mu + n*g, con pesos de Poisson(lam)."""
    from scipy.stats import poisson
    y = np.zeros_like(x)
    for n in range(8):
        y += poisson.pmf(n, lam) * np.exp(-0.5 * ((x - mu - n * g) / s) ** 2)
    return N * y


def _estimar_ganancia(activo, sigma, default):
    """Ajuste del histograma de pixeles: ruido gaussiano (x) Poisson de corriente oscura.
    Se prueban varias ganancias iniciales y se queda el mejor chi2."""
    from scipy.optimize import curve_fit
    v = activo[np.isfinite(activo)].ravel()
    hi = 30 * sigma
    v = v[(v > -6 * sigma) & (v < hi)]
    h, e = np.histogram(v, bins=300, range=(-6 * sigma, hi))
    c = 0.5 * (e[1:] + e[:-1])
    err = np.sqrt(np.maximum(h, 1))
    mejor = (np.inf, default)
    for g0 in sigma * np.array([3.5, 4.5, 5.5, 7.0, 9.0]):
        try:
            p, _ = curve_fit(_modelo_poisson, c, h, p0=[h.max(), 0.3, g0, sigma, 0.0], sigma=err,
                             bounds=([0, 1e-4, 2.5 * sigma, 0.5 * sigma, -2 * sigma],
                                     [np.inf, 5, 15 * sigma, 2 * sigma, 2 * sigma]), maxfev=4000)
        except (RuntimeError, ValueError):
            continue
        chi2 = (((h - _modelo_poisson(c, *p)) / err) ** 2).sum()
        if chi2 < mejor[0]:
            mejor = (chi2, float(p[2]))
    return mejor[1]


def calibrar(archivo, hdu, ganancia=None, hdul=None):
    """hdul: HDUList ya abierto (p.ej. un ROOT leido en memoria); si no, se abre 'archivo'."""
    if hdul is None:
        with fits.open(archivo) as f:
            return calibrar(archivo, hdu, ganancia, f)
    h0 = hdul[0].header
    hh = hdul[hdu].header
    d = hdul[hdu].data.astype(np.float64)
    if str(hh.get("BUNIT", "")).strip().upper() == "PSEUDO-E":
        # FITS convertido desde una imagen (particulas/imagenes.py): ya esta en pseudo-electrones
        return Amp(archivo, hdu, np.nan_to_num(d), _hint(hh, "NBINCOL", 1) or 1, 0, float("nan"),
                   float("nan"), calibrada=False, etiqueta=str(hh.get("PANEL", "")))
    binx = _hint(h0, "NBINCOL", 1) or 1
    npres = _hint(h0, "CCDNPRES", 0) or 0
    ccdncol = _hint(h0, "CCDNCOL")
    ncol = d.shape[1]
    x0 = int(np.ceil(npres / binx))
    x1 = ncol
    if ccdncol:
        x1 = min(ncol, int((npres + ccdncol // 2) // binx))
    over = d[:, x1 + 1:] if ncol - x1 > 6 else None

    # pedestal por fila desde el overscan (o desde la zona activa si no hay overscan)
    ref = over if over is not None else d[:, x0:x1]
    ped = np.nanmedian(ref, axis=1, keepdims=True)
    ped = np.where(np.isfinite(ped), ped, np.nanmedian(ped))
    act = d[:, x0:x1] - ped
    r = (over - ped) if over is not None else act[act < 0]
    r = r[np.isfinite(r)]
    sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
    if over is None:                              # solo lado negativo: sigma ~ mediana(|x|)/0.6745
        sigma = 1.4826 * np.median(np.abs(r))

    # columnas iniciales/finales con pedestal anomalo (bordes del prescan) y columnas calientes
    # (se compara con la mediana tipica de columna: la corriente oscura desplaza todas por igual)
    colmed = np.nanmedian(act, axis=0)
    dev = colmed - np.nanmedian(colmed)
    mad = 1.4826 * np.nanmedian(np.abs(dev))
    malas = np.where(np.abs(dev) > max(3 * sigma, 6 * mad))[0]
    act[:, malas] = 0.0

    g = ganancia or _estimar_ganancia(act, sigma, default=500.0 if binx == 1 else 870.0)
    e = np.nan_to_num(act / g, nan=0.0)
    ccdnrow = _hint(h0, "CCDNROW")
    filas = min(e.shape[0], ccdnrow // 2) if ccdnrow else e.shape[0]
    return Amp(archivo, hdu, e, binx, x0, g, sigma / g, [int(m + x0) for m in malas], filas_activas=filas)


def abrir_hdul(archivo):
    """FITS o ROOT (particulas/root.py) -> HDUList. Un ROOT se lee entero a memoria."""
    from .root import es_root, leer_root
    return leer_root(archivo) if es_root(archivo) else fits.open(archivo)


def calibrar_fits(archivo, ganancia=None):
    """FITS o ROOT de Skipper-CCD -> lista de Amp calibrados (uno por extension 2D)."""
    with abrir_hdul(archivo) as f:
        idx = [i for i, x in enumerate(f) if x.data is not None and x.data.ndim == 2]
        amps = [calibrar(archivo, i, ganancia, f) for i in idx]
    for k, a in enumerate(amps):   # numerar amplificadores 0..n-1 aunque el primario este vacio
        a.hdu = k
    return amps


# ----------------------------------------------------------------------------- clusters

@dataclass
class Cluster:
    y0: int; x0: int; y1: int; x1: int   # caja en coordenadas de la imagen ORIGINAL (x1,y1 exclusivos)
    npix: int
    energia_e: float
    max_e: float
    largo: float                 # extension a lo largo del eje principal (px fisicos)
    ancho: float                 # extension transversal (px fisicos)
    rectitud: float              # distancia entre extremos / longitud de la traza
    filas: int
    cols_fis: int
    binx: int = 1
    curvatura: float = 0.0       # sagita / largo (0 = recta)
    clase: str = ""

    @property
    def energia_kev(self):
        return self.energia_e * EV_POR_ELECTRON / 1000.0

    def dict(self):
        d = asdict(self); d["energia_kev"] = self.energia_kev
        return d


def _rectitud(mask_fis):
    """Cociente distancia-extremos / largo del camino (1 = recta, <1 = curva)."""
    from skimage.morphology import skeletonize
    sk = skeletonize(np.pad(mask_fis, 1))
    ys, xs = np.nonzero(sk)
    if len(ys) < 3:
        return 1.0
    # extremos a lo largo del eje principal
    pts = np.c_[ys, xs].astype(float)
    c = pts - pts.mean(0)
    u = np.linalg.svd(c, full_matrices=False)[2][0]
    p = c @ u
    a, b = pts[np.argmin(p)], pts[np.argmax(p)]
    # una recta digital 8-conexa tiene (distancia de Chebyshev + 1) pixeles
    return float(min(1.0, (np.abs(a - b).max() + 1) / len(ys)))


def _pts_fisicos(ys, xs, binx):
    """Coordenadas (y, x) en pixeles fisicos (centro de cada columna binneada)."""
    return np.c_[ys + 0.5, (xs + 0.5) * binx]


def _medir(m, e, y0, x0, amp):
    """Variables fisicas de un cluster dado por su mascara m (recorte con origen y0, x0)."""
    ys, xs = np.nonzero(m)
    w = e[ys, xs]
    E = float(w.sum())
    pts = _pts_fisicos(ys, xs, amp.binx)
    largo = ancho = 1.0
    curv = 0.0
    if len(pts) >= 2:
        c = pts - np.average(pts, axis=0, weights=w)
        vt = np.linalg.svd(c, full_matrices=False)[2]
        proj = c @ vt.T
        # largo conservador: con binning los extremos en x se conocen solo a +-binx/2
        largo = float(max(ys.max() - ys.min() + 1,
                          np.ptp(proj[:, 0]) + 1 - (amp.binx - 1) * abs(vt[0, 1])))
        if proj.shape[1] > 1 and len(pts) >= 6:
            # sagita: desvio transversal ajustado con una parabola a lo largo del eje
            s, t = proj[:, 0], proj[:, 1]
            a2 = np.polyfit(s, t, 2, w=np.sqrt(w))[0]
            curv = float(abs(a2) * (np.ptp(s) / 2) ** 2 / max(largo, 1.0))
        # ancho equivalente = sqrt(12) * dispersion transversal ponderada por carga (una banda
        # uniforme de ancho a tiene sigma = a/sqrt(12)). Robusto frente a pixeles sueltos.
        # Con binning, x esta cuantizada en pasos de binx columnas: se resta esa varianza.
        var_t = np.average(proj[:, 1] ** 2, weights=w) if proj.shape[1] > 1 else 0.0
        var_t -= (amp.binx ** 2 - 1) / 12.0 * vt[1, 1] ** 2 if len(vt) > 1 else 0.0
        ancho = float(np.sqrt(12 * max(var_t, 1 / 12)))
    m_fis = np.repeat(m, amp.binx, axis=1) if amp.binx > 1 else m
    rect = _rectitud(m_fis) if largo > 6 else 1.0
    ys0, xs0 = ys.min(), xs.min()
    return Cluster(int(y0 + ys0), int(x0 + xs0 + amp.x0), int(y0 + ys.max() + 1), int(x0 + xs.max() + 1 + amp.x0),
                   int(len(ys)), E, float(w.max()), largo, min(ancho, largo), rect,
                   int(ys.max() - ys0 + 1), int((xs.max() - xs0 + 1) * amp.binx), amp.binx, curv)


def _separar_recta(m, e, amp, p):
    """Si el cluster contiene una traza recta dominante (p.ej. un muon que toca/cruza otra traza),
    la separa con RANSAC. Devuelve lista de mascaras (la recta primero) o None."""
    from skimage.measure import LineModelND, ransac
    ys, xs = np.nonzero(m)
    if len(ys) < 30:
        return None
    pts = _pts_fisicos(ys, xs, amp.binx)
    tol = 2.5 + amp.binx / 2.0
    try:
        modelo, inl = ransac(pts, LineModelND, min_samples=2, residual_threshold=tol,
                             max_trials=300, rng=0)
    except Exception:
        return None
    if inl is None or inl.sum() < 20:
        return None
    rm = np.zeros_like(m); rm[ys[inl], xs[inl]] = True
    # quedarse con el tramo conexo (tolerando huecos de 1 px) mas cargado
    lab, n = ndi.label(ndi.binary_dilation(rm, np.ones((3, 3))) & m, np.ones((3, 3)))
    if n == 0:
        return None
    cargas = ndi.sum(e, lab, index=np.arange(1, n + 1))
    recta = lab == (1 + int(np.argmax(cargas)))
    ry, rx = np.nonzero(recta)
    rp = _pts_fisicos(ry, rx, amp.binx)
    extension = np.ptp((rp - rp.mean(0)) @ modelo.direction) + 1
    if extension < max(40, p.muon_min_largo) or e[recta].sum() < 0.35 * e[m].sum():
        return None
    resto = m & ~recta
    lab, n = ndi.label(resto, np.ones((3, 3)))
    partes = [recta]
    for i in range(1, n + 1):
        pi = lab == i
        # fragmentos chicos pegados a la recta = halo de difusion de la misma traza
        if pi.sum() < 8 or e[pi].sum() < 3 * p.min_energia_e:
            partes[0] = partes[0] | pi
        else:
            partes.append(pi)
    return partes if len(partes) > 1 else None


def encontrar_clusters(amp: Amp, p: Params = Params()):
    e = amp.electrones
    fuerte = e >= p.semilla_e
    debil = e >= p.crecer_e
    # histeresis: componentes de 'debil' que contienen alguna semilla
    lab, n = ndi.label(debil, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, bool); keep[np.unique(lab[fuerte])] = True; keep[0] = False
    mask = keep[lab]
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        es = np.where(m, e[sl], 0.0)
        if es.sum() < p.min_energia_e:
            continue
        pendientes = [m]
        for _ in range(4):                       # hasta 3 rectas separadas por cluster
            c = _medir(pendientes[-1], es, sl[0].start, sl[1].start, amp)
            if c.largo < 40 or clasificar(c, p) == "muon":
                break
            partes = _separar_recta(pendientes[-1], es, amp, p)
            if partes is None:
                break
            pendientes = pendientes[:-1] + partes[:1] + partes[1:]
            # seguir intentando sobre la parte restante mas grande
            pendientes.sort(key=lambda q: q.sum())
        for q in pendientes:
            if es[q].sum() >= p.min_energia_e and q.sum() >= p.min_pixeles:
                out.append(_medir(q, es, sl[0].start, sl[1].start, amp))
    return out


# ----------------------------------------------------------------------------- clasificacion

def clasificar(c: Cluster, p: Params = Params()):
    L, W = c.largo, c.ancho
    # artefactos: lineas de una sola fila (registro serie) o una sola columna fisica/binneada
    cols = c.cols_fis // c.binx                      # columnas en la imagen (binneadas)
    # con binning, un deposito chico de 1 fila puede ocupar 2-3 columnas: exigir una linea larga
    if c.filas <= 1 and cols >= (p.artefacto_min_largo if c.binx == 1 else 6):
        return "artefacto"
    if c.cols_fis <= 1 and c.filas >= p.artefacto_min_largo:
        return "artefacto"
    # alfa: mucha energia en una mancha compacta y redonda (caja fisica casi cuadrada). La energia
    # suele estar subestimada porque el nucleo satura el ADC, por eso el umbral es moderado.
    lado_max, lado_min = max(c.filas, c.cols_fis), min(c.filas, c.cols_fis)
    if (c.energia_e >= p.alfa_min_energia_e and lado_max <= p.alfa_max_tamano
            and lado_min / lado_max >= p.alfa_min_redondez and c.npix >= 15):
        return "alfa"
    if L <= p.puntual_max_largo:
        return "puntual"
    # la rectitud por esqueleto solo es fiable sin binning (con binning aparecen escalones)
    recto = c.curvatura <= p.muon_max_curvatura and (c.binx > 1 or c.rectitud >= p.muon_min_rectitud)
    if L >= p.muon_min_largo and W / L <= p.muon_max_ancho_rel and recto:
        return "muon"
    return "electron"


def analizar(archivo, p: Params = Params(), ganancia=None):
    """Devuelve [(Amp, [Cluster, ...]), ...] con la clase asignada por reglas."""
    res = []
    for amp in calibrar_fits(archivo, ganancia):
        cl = encontrar_clusters(amp, p)
        for c in cl:
            c.clase = clasificar(c, p)
        res.append((amp, cl))
    return res


# ----------------------------------------------------------------------------- imagen para la red

E_SAT = 2.0e4


def imagen_rgb(amp: Amp):
    """3 canales uint8: log(E) global, E lineal de baja energia, mascara >= 4 e-.
    Se devuelve la imagen COMPLETA (mismas coordenadas que el FITS), con 0 fuera de la zona activa."""
    e = np.clip(amp.electrones, 0, None)
    full = np.zeros((e.shape[0], e.shape[1] + amp.x0))
    full[:, amp.x0:] = e
    c0 = np.log1p(full) / np.log1p(E_SAT)
    c1 = full / 60.0
    c2 = (full >= 4.0).astype(float)
    rgb = np.stack([c0, c1, c2], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
