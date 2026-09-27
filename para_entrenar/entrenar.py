"""PASO 3 - Entrena el detector YOLO sobre para_entrenar/dataset/ (generado por construir_dataset.py).

    ..\\.venv\\Scripts\\python.exe entrenar.py --epochs 60

Parte del modelo SIN ENTRENAR en particulas (modelo_base/yolo11n.pt: pesos genericos de ultralytics).
El mejor modelo queda en para_entrenar/modelo_entrenado/detector_particulas.pt. Para usarlo en el applet:
    ..\\.venv\\Scripts\\python.exe ..\\listo_para_usar\\app.py --modelo modelo_entrenado\\detector_particulas.pt
"""
import argparse
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

AQUI = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default=str(AQUI / "modelo_base" / "yolo11n.pt"),
                    help="punto de partida (por defecto el modelo sin entrenar)")
    ap.add_argument("--datos", default=str(AQUI / "dataset" / "data.yaml"))
    ap.add_argument("--epochs", type=int, default=60)
    # GTX 1660 (6 GB): con 16 desborda la memoria y va 10x mas lento. Con mas VRAM se puede subir.
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--salida", default=str(AQUI / "modelo_entrenado" / "detector_particulas.pt"))
    a = ap.parse_args()

    if not os.path.exists(a.datos):
        raise SystemExit(f"No existe {a.datos}. Primero corre construir_dataset.py")
    model = YOLO(a.modelo)
    r = model.train(
        data=a.datos, epochs=a.epochs, batch=a.batch, imgsz=a.imgsz,
        project=str(AQUI / "runs"), name="particulas", exist_ok=True, workers=4, max_det=500, cache="ram",
        # Los canales codifican energia: nada de alterar color/brillo.
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        # La clase depende de la longitud y forma de la traza: sin rotaciones ni escalados fuertes
        # (con binning de columnas, rotar mezclaria ejes con distinta escala).
        degrees=0.0, scale=0.1, shear=0.0, perspective=0.0,
        # Sin mosaic: cada imagen ya tiene ~150 trazas; mosaic las cuadruplica y el asignador de
        # YOLO desborda una GPU de 6 GB.
        fliplr=0.5, flipud=0.5, mosaic=0.0,
        patience=20, plots=True,
    )
    best = Path(r.save_dir) / "weights" / "best.pt"
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, a.salida)
    print(f"\nModelo guardado en {a.salida}")
    print(f"Probarlo en el applet:  ..\\.venv\\Scripts\\python.exe ..\\listo_para_usar\\app.py --modelo \"{a.salida}\"")


if __name__ == "__main__":
    main()
