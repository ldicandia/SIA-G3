#!/usr/bin/env bash
# Regenera los resultados, construye la presentacion y verifica que sea consistente con ellos.
#
#   bash scripts/presentacion.sh [--salida RAIZ] [--destino ARCHIVO] [--sin-regenerar]
#
# Pasos, en orden:
#   1. bash scripts/regenerar.sh todo --salida RAIZ   (se omite con --sin-regenerar)
#   2. python3 -B presentacion/construir.py --salida RAIZ --destino ARCHIVO
#   3. python3 -B presentacion/verificar.py --pptx ARCHIVO --salida RAIZ
#
# RAIZ y ARCHIVO son relativos al directorio TP5 (por defecto "output" y
# "presentacion/TP5-autoencoders.pptx"). Con --sin-regenerar se usan las salidas que ya hay en RAIZ.
# Solo hace falta Python 3 y su biblioteca estandar; no se instala ni se borra nada.
set -euo pipefail

USO='uso: bash scripts/presentacion.sh [--salida RAIZ] [--destino ARCHIVO] [--sin-regenerar]'

# Ir a la raiz de TP5 (el directorio padre del que contiene este script), para que toda ruta
# relativa, incluidas las de --salida y --destino, sea relativa a TP5.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RAIZ="output"
DESTINO="presentacion/TP5-autoencoders.pptx"
REGENERAR=1
while [ "$#" -gt 0 ]; do
    case "$1" in
        --salida)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                echo "$USO" >&2
                exit 2
            fi
            RAIZ="$2"
            shift 2
            ;;
        --destino)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                echo "$USO" >&2
                exit 2
            fi
            DESTINO="$2"
            shift 2
            ;;
        --sin-regenerar)
            REGENERAR=0
            shift
            ;;
        *)
            echo "$USO" >&2
            exit 2
            ;;
    esac
done
RAIZ="${RAIZ%/}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "falta python3 en el PATH: la presentacion se construye solo con Python 3 y su biblioteca estandar" >&2
    exit 2
fi

INICIO=$SECONDS

if [ "$REGENERAR" -eq 1 ]; then
    if ! bash scripts/regenerar.sh todo --salida "$RAIZ"; then
        echo "PRESENTACION FALLA: paso regenerar los resultados (bash scripts/regenerar.sh todo --salida $RAIZ)" >&2
        exit 1
    fi
else
    for requerido in "$RAIZ/ej1a/salida-ej1a.txt" "$RAIZ/ej1b/salida-ej1b.txt" "$RAIZ/ej2/salida-ej2.txt"; do
        if [ ! -f "$requerido" ]; then
            echo "PRESENTACION FALLA: falta $requerido; se crea con: bash scripts/regenerar.sh todo --salida $RAIZ" >&2
            exit 1
        fi
    done
fi

if ! python3 -B presentacion/construir.py --salida "$RAIZ" --destino "$DESTINO"; then
    echo "PRESENTACION FALLA: paso construir la presentacion (presentacion/construir.py)" >&2
    exit 1
fi

if ! python3 -B presentacion/verificar.py --pptx "$DESTINO" --salida "$RAIZ"; then
    echo "PRESENTACION FALLA: paso verificar la consistencia (presentacion/verificar.py)" >&2
    exit 1
fi

echo "PRESENTACION OK: $DESTINO ($((SECONDS - INICIO)) s)"
