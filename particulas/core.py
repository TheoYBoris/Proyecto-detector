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
ESPESOR_UM = 675.0               # espesor del sensor (um)

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
    # --- muones cortos: casi perpendiculares al CCD. Un muon cruza los 675 um de espesor, asi que su
    # largo proyectado es 45*tan(alfa) px y su energia la de una MIP a lo largo de sqrt((15 L)^2 + 675^2) um.
    # Banda de energia = p5 .. 1.5*p98 de los muones largos del mismo tipo de imagen (bitacora, v0.5).
    # Validado contra la distribucion angular cos^2 (MC): 7-30 px sin binning (darks, CCD horizontal) y
    # 15-30 px con binning x10 (Atucha, CCD vertical); con binning, de 7 a 15 px el exceso no son muones.
    muon_corto_min_largo: float = 7.0
    muon_corto_min_largo_binning: float = 15.0
    mip_kev_um: tuple = (0.126, 0.57)           # sin binning
    mip_kev_um_binning: tuple = (0.236, 0.79)   # con binning de columnas
    # --- eventos del registro serie: sin difusion vertical (sigma_y < 0.3, corte de Atucha-II [JHEP24])
    sre_max_sigma_y: float = 0.3
    # --- union de tramos colineales de una misma traza (core._unir_colineales)
    unir_colineales: bool = True
    union_min_largo: float = 10.0     # px fisicos de cada tramo
    union_min_cos: float = 0.995      # ~5.7 grados
    union_max_hueco: float = 20.0     # px fisicos (con binning: al menos 3 columnas)
    usar_energia: bool = True                   # False en imagenes sin calibrar (energia no fisica)


def params_para(amp, p: Params = None):
    """Parametros de clasificacion para un amplificador. Sin calibracion (PNG/JPG/PDF) no hay energia
    real: cualquier mancha brillante satura la escala de pseudo-electrones. Para "alfa" se exige un nucleo
    saturado grande (~50 px), un pixel suelto no se distingue del ruido (>= 3 pixeles) y no se usan los
    criterios basados en energia (muon corto, registro serie)."""
    from dataclasses import replace
    p = p or Params()
    if getattr(amp, "calibrada", True):
        return p
    return replace(p, alfa_min_energia_e=1.0e6, min_pixeles=3, usar_energia=False)


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
    sigma_y: float = 1.0         # dispersion de la carga en filas (px); < 0.3 = sin difusion (registro serie)
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
    # sigma_y de cada columna (una traza inclinada no infla la dispersion), promediada con la carga
    sy_col, pesos = [], []
    for xc in np.unique(xs):
        sel = xs == xc
        wc, yc = w[sel], ys[sel]
        sy_col.append(np.sqrt(np.average((yc - np.average(yc, weights=wc)) ** 2, weights=wc)))
        pesos.append(wc.sum())
    sigma_y = float(np.average(sy_col, weights=pesos))
    return Cluster(int(y0 + ys0), int(x0 + xs0 + amp.x0), int(y0 + ys.max() + 1), int(x0 + xs.max() + 1 + amp.x0),
                   int(len(ys)), E, float(w.max()), largo, min(ancho, largo), rect,
                   int(ys.max() - ys0 + 1), int((xs.max() - xs0 + 1) * amp.binx), amp.binx, curv, sigma_y)


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


def _recta_de(mask_idx, amp):
    """(punto medio, direccion, t_min, t_max) en px fisicos si los pixeles forman una recta; si no, None."""
    ys, xs = mask_idx
    if len(ys) < 3:
        return None
    pts = _pts_fisicos(ys, xs, amp.binx)
    m = pts.mean(0)
    u = np.linalg.svd(pts - m, full_matrices=False)[2][0]
    t = (pts - m) @ u
    d = np.abs((pts - m) @ np.array([-u[1], u[0]]))
    if np.percentile(d, 90) >= 3 + amp.binx / 2:
        return None
    # con binning, una mancha de pocas columnas mide varios binx px fisicos de ancho y su "recta" sale
    # horizontal. Una recta de pendiente uy/ux tiene ~binx*|uy/ux| filas por columna (+ su ancho): si la
    # forma no coincide con la direccion, no es una recta
    if amp.binx > 1:
        esperado = amp.binx * abs(u[0]) / max(abs(u[1]), 0.05) + 3
        if len(ys) / len(np.unique(xs)) > 1.5 * esperado:
            return None
    return m, u, t.min(), t.max()


def _unir_colineales(out, mapa, e, amp, p):
    """Une tramos rectos colineales de una misma traza. Un muon largo queda partido cuando cruza una columna
    enmascarada por la calibracion, cuando su carga baja del umbral en algun pixel o donde lo cruza otra
    particula; cada tramo tiene solo parte de la energia y los cortos terminaban como electrones.
    Se unen dos tramos rectos (>= union_min_largo px) casi paralelos (|cos| >= union_min_cos), a menos de
    3 + binx/2 px de la misma recta y con un hueco <= max(union_max_hueco, 3*binx) px fisicos entre ellos."""
    idx = {k: np.nonzero(mapa == k + 1) for k in range(len(out))}
    rectas = {k: r for k, c in enumerate(out) if c.largo >= p.union_min_largo
              for r in [_recta_de(idx[k], amp)] if r is not None}
    padre = list(range(len(out)))

    def raiz(k):
        while padre[k] != k:
            padre[k] = padre[padre[k]]; k = padre[k]
        return k

    tol = 3 + amp.binx / 2
    hueco_max = max(p.union_max_hueco, 3 * amp.binx)
    malas = [c - amp.x0 for c in amp.columnas_malas]     # columnas enmascaradas, en indices de 'e'
    ks = sorted(rectas)
    for i, a in enumerate(ks):
        ma, ua, ta0, ta1 = rectas[a]
        for b in ks[i + 1:]:
            mb, ub, tb0, tb1 = rectas[b]
            cos = ua @ ub
            if abs(cos) < p.union_min_cos:
                continue
            if abs((mb - ma) @ np.array([-ua[1], ua[0]])) > tol:
                continue
            sb = (mb - ma) @ ua
            lo, hi = sorted((sb + tb0 * cos, sb + tb1 * cos))
            hueco = max(lo - ta1, ta0 - hi)
            if hueco < -5:          # se superponen a lo largo de la recta: trazas paralelas, no tramos
                continue
            # las columnas enmascaradas que caen en el hueco no cuentan (ahi la carga se borro)
            s0, s1 = (ta1, lo) if lo > ta1 else (hi, ta0)
            xa_, xb_ = sorted(((ma + ua * s0)[1] / amp.binx, (ma + ua * s1)[1] / amp.binx))
            n_malas = sum(1 for c in malas if xa_ - 1 <= c <= xb_ + 1) if hueco > 0 else 0
            hueco -= n_malas * amp.binx / max(abs(ua[1]), 0.1)
            if hueco <= hueco_max:
                padre[raiz(b)] = raiz(a)
    grupos = {}
    for k in range(len(out)):
        grupos.setdefault(raiz(k), []).append(k)
    if all(len(g) == 1 for g in grupos.values()):
        return out, mapa
    nuevo_out, nuevo_mapa = [], np.zeros_like(mapa)
    for g in sorted(grupos.values(), key=min):
        ys = np.concatenate([idx[k][0] for k in g]); xs = np.concatenate([idx[k][1] for k in g])
        if len(g) == 1:
            c = out[g[0]]
        else:
            y0, x0 = ys.min(), xs.min()
            m = np.zeros((ys.max() - y0 + 1, xs.max() - x0 + 1), bool)
            m[ys - y0, xs - x0] = True
            es = np.where(m, e[y0:ys.max() + 1, x0:xs.max() + 1], 0.0)
            c = _medir(m, es, y0, x0, amp)
        nuevo_out.append(c)
        nuevo_mapa[ys, xs] = len(nuevo_out)
    return nuevo_out, nuevo_mapa


def encontrar_clusters(amp: Amp, p: Params = Params(), con_mapa=False):
    """Clusters (una traza cada uno). con_mapa=True devuelve ademas un mapa del tamano de la imagen con
    el numero de cluster (1..n) de cada pixel, 0 = sin traza."""
    e = amp.electrones
    mapa = np.zeros(e.shape, np.int32)
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
            # tambien se intenta con los que ya parecen muones: una particula que toca un muon largo casi
            # no cambia su ancho/largo, y sin esto quedaba pegada al muon (_separar_recta solo separa
            # restos de >= 8 px y >= 3*min_energia_e; el halo de la propia traza se queda en la recta)
            if c.largo < 40:
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
                mapa[sl][q] = len(out)
    if p.unir_colineales:
        # se repite: al unir dos tramos la recta se ajusta mejor y puede alcanzar a un tercero
        for _ in range(5):
            n_antes = len(out)
            out, mapa = _unir_colineales(out, mapa, e, amp, p)
            if len(out) == n_antes:
                break
    return (out, mapa) if con_mapa else out


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
    if p.usar_energia and L < p.muon_min_largo:
        lo, hi = p.mip_kev_um_binning if c.binx > 1 else p.mip_kev_um
        kev_um = c.energia_kev / np.hypot(L * PIXEL_UM, ESPESOR_UM)
        # linea horizontal corta sin difusion y sin energia de muon: carga del registro serie
        if c.sigma_y < p.sre_max_sigma_y and c.filas <= 2 and kev_um < lo:
            return "artefacto"
        # traza recta corta con la energia de una MIP que cruza todo el espesor: muon casi perpendicular
        min_largo = p.muon_corto_min_largo_binning if c.binx > 1 else p.muon_corto_min_largo
        if recto and L >= min_largo and lo <= kev_um <= hi:
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
