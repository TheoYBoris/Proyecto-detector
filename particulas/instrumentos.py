"""Perfiles de instrumento con las especificaciones publicadas de cada experimento con Skipper-CCD.

Fuentes (carpeta Papers/ del proyecto):
  [JHEP24]  E. Depaoli et al., "Deployment and performance of a Low-Energy-Threshold Skipper-CCD inside
            a nuclear reactor", JHEP 10 (2024) 155.                                        -> Atucha-II
  [PRL25]   CONNIE & Atucha-II Collaborations, "Search for Reactor-Produced Millicharged Particles with
            Skipper-CCDs at the CONNIE and Atucha-II Experiments", PRL 134 (2025) 071801.
  [ENIAC25] S. Mirthis et al., "Particle Tracking Classification in the CONNIE Experiment" (2025).
  [ICHEP26] S. Mirthis, "Event Classification and Annotated Dataset in CONNIE", ICHEP 2026.   -> CONNIE
"""
from dataclasses import dataclass

import numpy as np

DENSIDAD_SI = 2.329            # g/cm^3
ENERGIA_BLOB_EV = 600.0        # [ENIAC25]: blob (> 600 eV) vs. diffusion hit (< 600 eV)
LINEAS_CU_KEV = {"Cu Kα": 8.048, "Cu Kβ": 8.905}   # fluorescencia del cobre usada para calibrar [JHEP24]
RANGO_ALFA_MEV = (3.0, 9.0)    # [ENIAC25]


@dataclass
class Instrumento:
    clave: str
    nombre: str
    pixel_um: float = 15.0          # tamano de pixel
    espesor_um: float = 675.0       # espesor del silicio
    ev_por_e: float = 3.75          # energia por par electron-hueco
    ruido_e: float = float("nan")   # ruido de lectura publicado (e-)
    binx: int = 1                   # binning de columnas tipico (para imagenes sin encabezado)
    nsamp: int = 0                  # muestras por pixel
    exposicion_h: float = float("nan")  # exposicion media por pixel de cada imagen
    nota_exposicion: str = ""       # de donde sale la exposicion (se muestra junto a las tasas)
    orientacion: str = ""           # posicion del plano del CCD (define el largo tipico de los muones)
    referencia: str = ""

    def masa_g(self, filas, cols_fisicas):
        """Masa activa de silicio de un amplificador (g)."""
        area_cm2 = filas * cols_fisicas * (self.pixel_um * 1e-4) ** 2
        return area_cm2 * self.espesor_um * 1e-4 * DENSIDAD_SI


INSTRUMENTOS = {
    "auto": Instrumento("auto", "Automático (según el encabezado FITS)"),
    # Exposicion: el run 43 (mar-abr 2025) tiene solo RUNID impares, 53.7 min de lectura y 10.1 min de hueco
    # entre imagenes, donde va la imagen par: el modo de [JHEP24] con una lectura rapida de limpieza entre
    # imagenes cientificas. Un pixel leido en la fraccion f de la imagen se limpio en la fraccion f de la
    # limpieza: exposicion = 10.1 + (53.7 - T_limpieza) f -> media 32 min si toda la pausa es limpieza,
    # hasta 37 min si la limpieza es mas corta. Inferido de los tiempos: confirmar con una imagen par.
    "atucha": Instrumento(
        "atucha", "Atucha-II (CNEA, 12 m del núcleo)", ruido_e=0.17, binx=10, nsamp=300,
        exposicion_h=32 / 60,
        nota_exposicion="32 min por imagen: modo con imagen de limpieza entre imágenes científicas (run 43, "
                        "inferido de los tiempos de lectura; rango 32–37 min). En lectura continua sería 53.7 min.",
        orientacion="vertical, lado largo (eje x de la imagen) hacia arriba [JHEP24, Fig. 2]: los muones "
                    "casi verticales dejan trazas largas a lo largo de x",
        referencia="Depaoli et al., JHEP 10 (2024) 155"),
    "connie": Instrumento(
        "connie", "CONNIE (Angra 2, 30 m del núcleo)", ruido_e=0.15, binx=1, nsamp=400,
        exposicion_h=2.0, referencia="CONNIE, PRL 134 (2025) 071801; Mirthis, ICHEP 2026"),
    "generico": Instrumento("generico", "Skipper-CCD genérico (15 µm, 675 µm)"),
}


def detectar_instrumento(archivo):
    """Reconoce el instrumento por la geometria del CCD en el encabezado. 'generico' si no coincide."""
    try:
        from astropy.io import fits
        from .root import es_root, leer_encabezado_root
        h = leer_encabezado_root(archivo) if es_root(archivo) else fits.getheader(archivo, 0)
        ncol, nrow = int(str(h.get("CCDNCOL", 0)).strip()), int(str(h.get("CCDNROW", 0)).strip())
        nsamp = int(str(h.get("NSAMP", 0)).strip())
    except Exception:
        return INSTRUMENTOS["generico"]
    if ncol == 6144 and nrow == 1024:          # 9.216 x 1.536 cm2, 6.29 MPx [JHEP24]
        return INSTRUMENTOS["atucha"]
    if {ncol, nrow} == {1022, 682} or (nsamp == 400 and 600 <= min(ncol, nrow) <= 1100):
        return INSTRUMENTOS["connie"]           # 1022 x 682 px [ENIAC25]
    return INSTRUMENTOS["generico"]


def subclase_puntual(energia_kev):
    """Convencion de CONNIE para depositos pequenos: blob (> 600 eV) o diffusion hit (< 600 eV).
    Los diffusion hits incluyen los candidatos a CEvNS."""
    if energia_kev is None or not np.isfinite(energia_kev):
        return ""
    return "blob" if energia_kev * 1000 > ENERGIA_BLOB_EV else "difusion"


def masa_amps(amps, inst):
    """Masa activa total (g) de los amplificadores calibrados."""
    total = 0.0
    for a in amps:
        filas, cols = a.electrones.shape
        filas = a.filas_activas or filas          # sin el overscan vertical
        total += inst.masa_g(filas, cols * a.binx)
    return total
