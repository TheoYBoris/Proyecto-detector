import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .core import CLASES

COLORES = {"muon": "#ff3b30", "electron": "#34c759", "alfa": "#ffcc00",
           "puntual": "#0a84ff", "artefacto": "#bf5af2"}


def dibujar(amps_y_cajas, titulo="", energia=True):
    """amps_y_cajas: [(Amp, [(x0, y0, x1, y1, clase, texto_extra), ...]), ...]"""
    n = len(amps_y_cajas)
    # hasta 4 paneles por fila (imagenes/PDF pueden traer muchos)
    ncol = min(n, 4)
    nfil = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nfil, ncol, figsize=(5.2 * ncol, 6.2 * nfil), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes = [axes.ravel()]
    for ax, (amp, cajas) in zip(axes[0], amps_y_cajas):
        full = np.zeros((amp.electrones.shape[0], amp.electrones.shape[1] + amp.x0))
        full[:, amp.x0:] = amp.electrones
        ax.imshow(np.log10(np.clip(full, 0.5, None)), origin="lower", cmap="gray_r",
                  vmin=np.log10(0.5), vmax=np.log10(2e4), interpolation="nearest", aspect="auto")
        for x0, y0, x1, y1, clase, extra in cajas:
            col = COLORES.get(clase, "w")
            ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, lw=0.9, ec=col))
            if extra and (y1 - y0 > 6 or x1 - x0 > 6):
                ax.text(x0, y1 + 0.5, extra, color=col, fontsize=5, va="bottom")
        if getattr(amp, "calibrada", True):
            ax.set_title(f"amp {amp.hdu}  (bin x{amp.binx}, g={amp.ganancia:.0f} ADU/e, "
                         f"ruido={amp.ruido_e:.2f} e)", fontsize=8)
        else:
            ax.set_title(f"{amp.etiqueta or 'imagen'}  (sin calibrar: energia no disponible)", fontsize=8)
        ax.set_xlabel("columna"); ax.set_ylabel("fila")
    handles = [Rectangle((0, 0), 1, 1, fill=False, ec=COLORES[c], label=c) for c in CLASES]
    fig.legend(handles=handles, loc="upper right", ncol=len(CLASES), fontsize=8, frameon=False)
    fig.suptitle(titulo, fontsize=9, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
