"""TODO EN UNO: de tus imagenes a tu propio applet.

    1. Deja tus imagenes en para_entrenar/mis_datos/  (FITS, ROOT, PNG/JPG/TIFF o PDF). Podes hacer una
       subcarpeta por run o por detector: cada subcarpeta se valida por separado.
    2. (Opcional) Ajusta los criterios en para_entrenar/criterios.yaml.
    3. Doble clic en ENTRENAR_CON_MIS_DATOS.bat   (o:  ..\\.venv\\Scripts\\python.exe entrenar_todo.py)

Al terminar queda para_entrenar/mi_applet/ con tu modelo, tus criterios, un resumen (RESULTADOS.txt) y
iniciar_applet.bat para abrir el applet con todo eso.

Otras opciones:
    entrenar_todo.py --revisar                  solo dibuja las etiquetas automaticas (para ajustar criterios)
    entrenar_todo.py --datos D:\\run7 --nombre applet_run7 --epochs 40
    entrenar_todo.py --partir-de base           entrenar desde el modelo generico en vez del del proyecto
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
PY = sys.executable
MODELOS = {"proyecto": os.path.join(RAIZ, "listo_para_usar", "modelo", "detector_particulas.pt"),
           "base": os.path.join(AQUI, "modelo_base", "yolo11n.pt")}


def paso(n, texto):
    print(f"\n{'=' * 78}\n PASO {n}: {texto}\n{'=' * 78}", flush=True)


def correr(args):
    r = subprocess.run([PY] + args, cwd=AQUI)
    if r.returncode != 0:
        sys.exit(f"\nFallo el paso anterior (codigo {r.returncode}). Revisa el mensaje de error de arriba.")


def grupos_de(datos):
    """Cada subcarpeta con imagenes es un grupo; los archivos sueltos en la carpeta principal, otro."""
    from construir_dataset import expandir
    grupos = []
    if expandir(datos):
        grupos.append(datos)
    for sub in sorted(glob.glob(os.path.join(datos, "*"))):
        if os.path.isdir(sub) and expandir(sub):
            grupos.append(sub)
    return grupos


def resumen(carpeta, dataset, runs):
    """RESULTADOS.txt: cuantas trazas de cada clase se usaron y como le fue al modelo en validacion."""
    import pandas as pd
    lineas = []
    cl = pd.read_csv(os.path.join(dataset, "clusters.csv"))
    tabla = cl.groupby(["split", "clase"]).size().unstack(fill_value=0)
    lineas += ["TRAZAS USADAS (etiquetas de los criterios)", tabla.to_string(), ""]
    res = os.path.join(runs, "particulas", "results.csv")
    if os.path.exists(res):
        r = pd.read_csv(res)
        r.columns = [c.strip() for c in r.columns]
        mejor = r.loc[r["metrics/mAP50(B)"].idxmax()]
        lineas += ["VALIDACION (imagenes que el modelo no vio al entrenar)",
                   f"  mejor epoca: {int(mejor['epoch'])} de {len(r)}",
                   f"  mAP50: {mejor['metrics/mAP50(B)']:.3f}   precision: {mejor['metrics/precision(B)']:.3f}   "
                   f"recall: {mejor['metrics/recall(B)']:.3f}",
                   "  (miden cuanto coincide el modelo con los criterios, no con la particula real)", ""]
        ult = r.iloc[-1]
        if ult["val/box_loss"] > 1.2 * r["val/box_loss"].min() and ult["train/box_loss"] < r["train/box_loss"].iloc[0]:
            lineas.append("AVISO: la perdida de validacion subio al final mientras la de entrenamiento bajaba: "
                          "posible sobreajuste. Con pocos datos, conviene --epochs menor o mas imagenes.")
    n_val = len(os.listdir(os.path.join(dataset, "images", "val")))
    pocas = [c for c in tabla.columns if tabla.loc["train", c] < 50] if "train" in tabla.index else []
    if n_val < 8:
        lineas.append(f"AVISO: solo {n_val} imagenes de validacion; las metricas son poco confiables.")
    if pocas:
        lineas.append(f"AVISO: pocas trazas de entrenamiento para {pocas} (< 50): el modelo va a reconocerlas mal.")
    with open(os.path.join(carpeta, "RESULTADOS.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")
    print("\n".join(lineas))


def crear_applet(carpeta, criterios):
    shutil.copy(criterios, os.path.join(carpeta, "criterios.yaml"))
    bat = (
        "@echo off\r\n"
        "rem Applet con el modelo y los criterios de esta carpeta (generado por entrenar_todo.py).\r\n"
        "rem Opciones: --red (otras PCs de la red), --compartir (link publico temporal)\r\n"
        "cd /d \"%~dp0\"\r\n"
        "echo Iniciando el applet... el navegador se abre solo en unos segundos.\r\n"
        "\"..\\..\\.venv\\Scripts\\python.exe\" \"..\\..\\listo_para_usar\\app.py\" "
        "--modelo modelo\\detector_particulas.pt --criterios criterios.yaml %*\r\n"
        "pause\r\n")
    with open(os.path.join(carpeta, "iniciar_applet.bat"), "w", newline="") as fh:
        fh.write(bat)
    with open(os.path.join(carpeta, "LEEME.txt"), "w", encoding="utf-8") as fh:
        fh.write("Tu applet personalizado\n=======================\n\n"
                 "iniciar_applet.bat         abre el applet con tu modelo y tus criterios (doble clic)\n"
                 "modelo\\                   el detector entrenado con tus datos\n"
                 "criterios.yaml             los criterios con los que se etiqueto (el applet usa los mismos)\n"
                 "RESULTADOS.txt             cuantas trazas se usaron y como le fue al modelo en validacion\n"
                 "runs\\particulas\\          curvas y metricas del entrenamiento\n"
                 "dataset\\                  imagenes y etiquetas usadas (se puede borrar para ahorrar espacio)\n\n"
                 "Para usarlo en otra PC: copiar esta carpeta dentro de para_entrenar\\ del proyecto instalado.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", default=os.path.join(AQUI, "mis_datos"), help="carpeta con tus imagenes")
    ap.add_argument("--criterios", default=os.path.join(AQUI, "criterios.yaml"))
    ap.add_argument("--nombre", default="mi_applet", help="carpeta de salida dentro de para_entrenar/")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8, help="imagenes por paso (8 entra en una GPU de 6 GB)")
    ap.add_argument("--partir-de", choices=list(MODELOS), default="proyecto",
                    help="'proyecto': el detector ya entrenado (aprende rapido, mejor con pocos datos); "
                         "'base': el modelo generico sin entrenar en particulas")
    ap.add_argument("--revisar", action="store_true", help="solo dibujar las etiquetas automaticas y terminar")
    a = ap.parse_args()

    from particulas.core import cargar_criterios
    try:
        cargar_criterios(a.criterios)
    except (OSError, ValueError) as e:
        sys.exit(f"Error en el archivo de criterios: {e}")
    grupos = grupos_de(a.datos)
    if not grupos:
        sys.exit(f"No hay imagenes en {a.datos}. Copia ahi tus archivos (FITS, ROOT, PNG, JPG, TIFF o PDF),\n"
                 "sueltos o en subcarpetas, y volve a correr.")
    carpeta = os.path.join(AQUI, a.nombre)
    dataset, runs = os.path.join(carpeta, "dataset"), os.path.join(carpeta, "runs")
    print(f"Datos: {a.datos}  ({len(grupos)} grupo(s))\nCriterios: {a.criterios}\nSalida: {carpeta}")

    paso(1, "revisar las etiquetas automaticas (figuras de muestra)")
    revision = os.path.join(carpeta, "revision_etiquetas")
    correr(["revisar_etiquetas.py", *grupos, "--criterios", a.criterios, "--max", "3", "--out", revision])
    print(f"Figuras en {revision}. Si las etiquetas no te convencen, ajusta criterios.yaml y volve a correr.")
    if a.revisar:
        return

    paso(2, "construir el dataset (imagenes + etiquetas de los criterios)")
    correr(["construir_dataset.py", "--datos", *grupos, "--criterios", a.criterios, "--val-min", "2",
            "--out", dataset])

    paso(3, f"entrenar ({a.epochs} epocas, partiendo del modelo '{a.partir_de}')")
    modelo = os.path.join(carpeta, "modelo", "detector_particulas.pt")
    correr(["entrenar.py", "--modelo", MODELOS[a.partir_de], "--datos", os.path.join(dataset, "data.yaml"),
            "--epochs", str(a.epochs), "--batch", str(a.batch), "--salida", modelo, "--runs", runs])

    paso(4, "armar tu applet")
    crear_applet(carpeta, a.criterios)
    resumen(carpeta, dataset, runs)
    print(f"\nListo. Tu applet: {os.path.join(carpeta, 'iniciar_applet.bat')}")


if __name__ == "__main__":
    main()
