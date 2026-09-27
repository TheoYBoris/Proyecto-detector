"""Deteccion completa sobre un FITS: calibracion -> detector (YOLO o reglas) -> cajas + figura.
Lo usan detectar.py (linea de comandos) y app.py (applet web)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import (CLASES, EV_POR_ELECTRON, Params, calibrar_fits, clasificar, encontrar_clusters, imagen_rgb,
                   params_para)
from .dibujo import dibujar

# modelo ya entrenado que usa el kit listo_para_usar/
MODELO_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "listo_para_usar", "modelo", "detector_particulas.pt")

METODOS = {
    "auto": "Automático (YOLO con binning, reglas sin binning)",
    "yolo": "Detector YOLO (red neuronal que clasifica las trazas reconstruidas)",
    "reglas": "Reglas físicas (morfología)",
}


def cargar_modelo(ruta=MODELO_DEFAULT):
    from ultralytics import YOLO
    return YOLO(ruta)


def params_umbral(umbral_e=None):
    """Params con una energia minima por evento distinta (en e-). Referencias: Atucha-II trabaja desde
    12 e- = 45 eV [JHEP24] y CONNIE desde ~15 eV [PRL25]. La semilla nunca supera el umbral."""
    p = Params()
    if umbral_e and umbral_e > 0:
        from dataclasses import replace
        p = replace(p, min_energia_e=float(umbral_e), semilla_e=min(p.semilla_e, max(2.0, float(umbral_e))))
    return p


def detectar_yolo(model, amps, conf=0.25, imgsz=640, prm: Params = None):
    """Por amplificador, lista de dicts con las detecciones (coordenadas del FITS)."""
    prm = prm or Params()
    imgs = [imagen_rgb(a)[::-1] for a in amps]
    # NMS agnostico: una traza recibe una sola caja aunque dos clases compitan por ella
    res = model.predict(imgs, conf=conf, imgsz=imgsz, max_det=500, agnostic_nms=True, verbose=False)
    salida = []
    for amp, img, r in zip(amps, imgs, res):
        h = img.shape[0]
        dets = []
        for (x0, y0, x1, y1), c, p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                                          r.boxes.conf.cpu().numpy()):
            # volver a coordenadas del FITS (fila 0 abajo)
            fy0, fy1 = h - y1, h - y0
            xa, xb = int(max(amp.x0, np.floor(x0))), int(np.ceil(x1))
            ya, yb = int(max(0, np.floor(fy0))), int(np.ceil(fy1))
            sub = amp.electrones[ya:yb, xa - amp.x0:xb - amp.x0]
            e = float(sub[sub >= prm.crecer_e].sum()) if sub.size else 0.0
            # consistencia fisica: descartar cajas sin un deposito real (ruido/corriente oscura)
            if not sub.size or sub.max() < prm.semilla_e or e < prm.min_energia_e:
                continue
            dets.append(dict(amp=amp.hdu, clase=CLASES[int(c)], confianza=float(p),
                             x0=float(x0), x1=float(x1), y0=float(fy0), y1=float(fy1),
                             energia_e=e, energia_kev=e * EV_POR_ELECTRON / 1000))
        salida.append(dets)
    return salida


def detectar_reglas(amps, p: Params = Params()):
    salida = []
    for amp in amps:
        pa = params_para(amp, p)
        dets = []
        for c in encontrar_clusters(amp, pa):
            dets.append(dict(amp=amp.hdu, clase=clasificar(c, pa), confianza=float("nan"), origen="reglas",
                             x0=float(c.x0), x1=float(c.x1), y0=float(c.y0), y1=float(c.y1),
                             energia_e=c.energia_e, energia_kev=c.energia_kev))
        salida.append(dets)
    return salida


IOU_MIN = 0.3        # caja de la red que corresponde a una traza reconstruida


def _caja_etiqueta(c):
    """Caja con la que se entreno la red para un cluster (construir_dataset.cajas_yolo): +-1 px, minimo 5 px."""
    x0, y0, x1, y1 = c.x0 - 1, c.y0 - 1, c.x1 + 1, c.y1 + 1
    if x1 - x0 < 5:
        m = (5 - (x1 - x0)) / 2; x0 -= m; x1 += m
    if y1 - y0 < 5:
        m = (5 - (y1 - y0)) / 2; y0 -= m; y1 += m
    return x0, y0, x1, y1


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def detectar_hibrido(model, amps, conf=0.25, imgsz=640, prm: Params = None):
    """Red neuronal + reconstruccion. Las trazas (pixeles, caja, energia) salen de la reconstruccion por
    pixeles de las reglas, que separa las particulas que se cruzan y no parte los muones; la red solo
    clasifica: cada traza toma la clase de la caja de la red que mejor se le superpone (IoU >= 0.3). Si la
    red no tiene caja para esa traza (p.ej. depositos bajo el umbral con que se entreno), clasifican las
    reglas. Asi un muon largo que otra particula cruza no queda partido en varias cajas."""
    prm = prm or Params()
    cajas = detectar_yolo(model, amps, conf, imgsz, prm)
    salida = []
    for amp, ds in zip(amps, cajas):
        pa = params_para(amp, prm)
        bs = [(d["x0"], d["y0"], d["x1"], d["y1"]) for d in ds]
        dets = []
        for c in encontrar_clusters(amp, pa):
            b = _caja_etiqueta(c)
            ious = [_iou(b, x) for x in bs]
            k = int(np.argmax(ious)) if ious else -1
            if k >= 0 and ious[k] >= IOU_MIN:
                clase, confianza, origen = ds[k]["clase"], ds[k]["confianza"], "red"
            else:
                clase, confianza, origen = clasificar(c, pa), float("nan"), "reglas"
            dets.append(dict(amp=amp.hdu, clase=clase, confianza=confianza, origen=origen,
                             x0=float(c.x0), x1=float(c.x1), y0=float(c.y0), y1=float(c.y1),
                             energia_e=c.energia_e, energia_kev=c.energia_kev))
        salida.append(dets)
    return salida


def cargar_amps(ruta, ganancia=None, escala=1.0, binx=1, paneles=True):
    """FITS (calibrado o convertido), ROOT o imagen PNG/JPG/TIFF/PDF -> lista de Amp."""
    from .imagenes import cargar_imagen, es_imagen
    if es_imagen(ruta):
        return cargar_imagen(ruta, escala=escala, binx=binx, paneles=paneles)
    return calibrar_fits(ruta, ganancia)


def procesar_archivo(ruta, metodo="auto", model=None, conf=0.25, ganancia=None,
                     escala=1.0, binx=1, paneles=True, umbral_e=None):
    """Devuelve (amps, detecciones por amplificador, metodo efectivamente usado).
    umbral_e: energia minima por evento en e- (por defecto 60 e- = 225 eV)."""
    amps = cargar_amps(ruta, ganancia, escala, binx, paneles)
    if not amps:
        raise ValueError("el archivo no contiene imagenes")
    prm = params_umbral(umbral_e)
    usado = metodo
    if metodo == "auto":
        # el detector se entreno casi solo con imagenes binneadas; sin binning las reglas son mas fiables
        usado = "yolo" if amps[0].binx > 1 else "reglas"
    if usado == "yolo":
        if model is None:
            model = cargar_modelo()
        # las trazas salen de la reconstruccion y la red las clasifica (ver detectar_hibrido). Los depositos
        # bajo el umbral de entrenamiento (60 e-) no tienen caja de la red y los clasifican las reglas.
        dets = detectar_hibrido(model, amps, conf, prm=prm)
    else:
        dets = detectar_reglas(amps, prm)
    from .instrumentos import subclase_puntual
    for amp, ds in zip(amps, dets):
        for d in ds:
            # imagenes sin calibrar: la energia en pseudo-electrones no es fisica -> no se informa
            if not amp.calibrada:
                d["energia_e"] = d["energia_kev"] = float("nan")
            # convencion CONNIE: puntual -> blob (> 600 eV) o difusion (< 600 eV, candidatos a CEvNS)
            d["subclase"] = subclase_puntual(d["energia_kev"]) if d["clase"] == "puntual" else ""
    return amps, dets, usado


procesar_fits = procesar_archivo  # compatibilidad


def figura(amps, dets, titulo=""):
    paneles = []
    for amp, ds in zip(amps, dets):
        cajas = []
        for d in ds:
            conf = "" if np.isnan(d["confianza"]) else f" {d['confianza']:.2f}"
            ener = "" if np.isnan(d["energia_kev"]) else f" {d['energia_kev']:.0f}keV"
            cajas.append((d["x0"], d["y0"], d["x1"], d["y1"], d["clase"], f"{d['clase']}{conf}{ener}"))
        paneles.append((amp, cajas))
    return dibujar(paneles, titulo=titulo)


def guardar_figura(amps, dets, ruta_png, titulo="", dpi=150):
    fig = figura(amps, dets, titulo)
    fig.savefig(ruta_png, dpi=dpi)
    plt.close(fig)
