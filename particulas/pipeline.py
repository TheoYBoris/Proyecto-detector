"""Deteccion completa sobre un FITS: calibracion -> detector (YOLO o reglas) -> cajas + figura.
Lo usan detectar.py (linea de comandos) y app.py (applet web)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import CLASES, EV_POR_ELECTRON, Params, calibrar_fits, clasificar, encontrar_clusters, imagen_rgb
from .dibujo import dibujar

# modelo ya entrenado que usa el kit listo_para_usar/
MODELO_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "listo_para_usar", "modelo", "detector_particulas.pt")

METODOS = {
    "auto": "Automático (YOLO con binning, reglas sin binning)",
    "yolo": "Detector YOLO (red neuronal)",
    "reglas": "Reglas físicas (morfología)",
}


def cargar_modelo(ruta=MODELO_DEFAULT):
    from ultralytics import YOLO
    return YOLO(ruta)


UMBRAL_ENTRENAMIENTO_E = Params().min_energia_e   # el detector vio eventos de >= 60 e- (225 eV)


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
    from dataclasses import replace
    salida = []
    for amp in amps:
        pa = p
        if not amp.calibrada:
            # sin calibracion no hay energia real: cualquier mancha brillante satura la escala de
            # pseudo-electrones. Para "alfa" se exige un nucleo saturado grande (~50 px).
            # Un pixel suelto no se distingue del ruido de la imagen: se piden >= 3 pixeles.
            pa = replace(p, alfa_min_energia_e=1.0e6, min_pixeles=3)
        dets = []
        for c in encontrar_clusters(amp, pa):
            dets.append(dict(amp=amp.hdu, clase=clasificar(c, pa), confianza=float("nan"),
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


def _agregar_baja_energia(amps, dets, prm):
    """Modo hibrido: los depositos por debajo del umbral con que se entreno YOLO se toman de las reglas,
    salvo los que ya caen dentro de una caja del detector."""
    for amp, ds, extra in zip(amps, dets, detectar_reglas(amps, prm)):
        for d in extra:
            if d["energia_e"] >= UMBRAL_ENTRENAMIENTO_E:
                continue
            cx, cy = (d["x0"] + d["x1"]) / 2, (d["y0"] + d["y1"]) / 2
            if not any(b["x0"] <= cx <= b["x1"] and b["y0"] <= cy <= b["y1"] for b in ds):
                ds.append(d)
    return dets


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
        dets = detectar_yolo(model, amps, conf, prm=prm)
        if prm.min_energia_e < UMBRAL_ENTRENAMIENTO_E:
            dets = _agregar_baja_energia(amps, dets, prm)
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
