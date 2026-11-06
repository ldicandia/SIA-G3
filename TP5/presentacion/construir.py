"""Constructor de la presentacion del TP5.

Uso:
    python3 -B presentacion/construir.py [--salida DIR] [--destino ARCHIVO] [--solo PREFIJO[,PREFIJO...]]

Descubre las secciones presentacion/secciones/seccion_*.py en orden de nombre;
cada una define agregar(pres, salidas). Las cifras de las diapositivas entran
solo por lector.Salidas, a partir de lo que regenera scripts/regenerar.sh.

Codigos de salida: 0 bien; 1 falta una entrada o el layout no cierra;
2 un prefijo de --solo no coincide con ninguna seccion.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from lector import FaltaEntrada, Salidas  # noqa: E402
from pptx_minimo import ErrorDeLayout, Presentacion  # noqa: E402

TITULO_PRESENTACION = "TP5: autoencoders (básico, denoising y variacional)"


def descubrir_secciones() -> list[Path]:
    return sorted((AQUI / "secciones").glob("seccion_*.py"), key=lambda p: p.name)


def cargar(ruta: Path):
    spec = importlib.util.spec_from_file_location(f"secciones_{ruta.stem}", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    if not hasattr(modulo, "agregar"):
        raise SystemExit(f"la seccion {ruta.name} no define agregar(pres, salidas)")
    return modulo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Construye la presentacion .pptx del TP5")
    ap.add_argument("--salida", default="output", help="raiz de las salidas regeneradas")
    ap.add_argument("--destino", default="presentacion/TP5-autoencoders.pptx")
    ap.add_argument(
        "--solo",
        default=None,
        help="prefijos, separados por comas, de los nombres de archivo de las secciones a incluir",
    )
    args = ap.parse_args(argv)

    secciones = descubrir_secciones()
    if args.solo is not None:
        prefijos = [p.strip() for p in args.solo.split(",") if p.strip()]
        if not prefijos:
            print("--solo no tiene ningun prefijo")
            return 2
        for p in prefijos:
            if not any(s.name.startswith(p) for s in secciones):
                print(f"--solo: el prefijo '{p}' no coincide con ninguna seccion")
                return 2
        secciones = [s for s in secciones if any(s.name.startswith(p) for p in prefijos)]

    salidas = Salidas(args.salida)
    pres = Presentacion(TITULO_PRESENTACION)
    try:
        for ruta in secciones:
            cargar(ruta).agregar(pres, salidas)
        pres.guardar(args.destino)
    except FaltaEntrada as e:
        print(f"FALTA ENTRADA: {e}")
        return 1
    except ErrorDeLayout as e:
        print(f"LAYOUT: {e}")
        return 1
    except ValueError as e:
        print(f"ENTRADA INVALIDA: {e}")
        return 1
    print(
        f"PRESENTACION: {pres.n_diapositivas} diapositivas, {pres.n_imagenes} imagenes -> {args.destino}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
