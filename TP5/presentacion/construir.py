"""Constructor de la presentacion (version trazadora).

Arma una diapositiva con la linea RESULTADO del reporte del Ej 1a y su grafico
latente, para probar la cadena completa: salidas regeneradas -> .pptx valido.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from pptx_minimo import ErrorDeLayout, Presentacion  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Construye la presentacion .pptx del TP5")
    ap.add_argument("--salida", default="output", help="raiz de las salidas regeneradas")
    ap.add_argument("--destino", default="presentacion/TP5-autoencoders.pptx")
    args = ap.parse_args(argv)

    raiz = Path(args.salida)
    reporte = raiz / "ej1a" / "salida-ej1a.txt"
    grafico = raiz / "ej1a" / "latent.png"
    for ruta in (reporte, grafico):
        if not ruta.is_file():
            print(f"FALTA ENTRADA: {ruta}; se crea con: bash scripts/regenerar.sh 1a")
            return 1
    linea = next(
        l.rstrip("\n")
        for l in reporte.read_text(encoding="utf-8").splitlines(keepends=True)
        if l.startswith("RESULTADO:")
    )

    pres = Presentacion("TP5: autoencoders")
    try:
        d = pres.diapositiva("Ej 1a: tracer")
        d.codigo(0.5, 1.7, 12.333, 0.7, [linea], fuente="ej1a/salida-ej1a.txt", tam=14)
        d.imagen(grafico, 0.5, 2.6, 12.333, 4.4, alt="output/ej1a/latent.png")
        pres.guardar(args.destino)
    except ErrorDeLayout as e:
        print(f"LAYOUT: {e}")
        return 1
    print(f"PRESENTACION: {pres.n_diapositivas} diapositivas, {pres.n_imagenes} imagenes -> {args.destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
