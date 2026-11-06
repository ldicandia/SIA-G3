#!/usr/bin/env bash
# Regenera los resultados y graficos de un ejercicio del TP5 con un solo comando.
#
#   bash scripts/regenerar.sh <1a|1b|2|todo> [--salida RAIZ]
#
# RAIZ es relativa al directorio TP5 (por defecto "output"). Cada ejercicio escribe en
# RAIZ/ej1a, RAIZ/ej1b o RAIZ/ej2 y deja el reporte impreso de cada subcomando en
# RAIZ/ejX/salida-SUBCOMANDO.txt.
set -euo pipefail

USO='uso: bash scripts/regenerar.sh <1a|1b|2|todo> [--salida RAIZ]'

# Ir a la raiz de TP5 (el directorio padre del que contiene este script), para que toda ruta
# relativa, incluida la de --salida, sea relativa a TP5.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

EJERCICIO="${1:-}"
if [ -z "$EJERCICIO" ]; then
    echo "$USO" >&2
    exit 2
fi
shift || true

RAIZ="output"
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
        *)
            echo "$USO" >&2
            exit 2
            ;;
    esac
done
RAIZ="${RAIZ%/}"

case "$EJERCICIO" in
    2) ;;
    *)
        echo "$USO" >&2
        exit 2
        ;;
esac

if ! command -v cargo >/dev/null 2>&1; then
    echo "falta cargo en el PATH: instale Rust (ver la seccion Requisitos del README.md)" >&2
    exit 2
fi

CONSTRUIDO=0
construir() {
    if [ "$CONSTRUIDO" -eq 0 ]; then
        echo "compilando: cargo build --release --locked"
        cargo build --release --locked
        CONSTRUIDO=1
    fi
}

# correr SUBCOMANDO DIR [ARGS...]: corre el subcomando escribiendo en DIR y guarda el reporte
# impreso (salida estandar y de error) en DIR/salida-SUBCOMANDO.txt. pipefail propaga el fallo.
correr() {
    local sub="$1" dir="$2"
    shift 2
    mkdir -p "$dir"
    echo "== cargo run --release -- $sub $* --out $dir"
    cargo run --release --locked -q -- "$sub" "$@" --out "$dir" 2>&1 | tee "$dir/salida-$sub.txt"
}

# borrar_esperados DIR NOMBRE...: borra solo esos nombres exactos de archivo dentro de DIR
# (nunca un directorio ni un comodin), para que su presencia posterior pruebe que los escribio
# esta corrida y no codigo anterior.
borrar_esperados() {
    local dir="$1" nombre
    shift
    for nombre in "$@"; do
        rm -f -- "$dir/$nombre"
    done
}

# verificar_esperados EJ DIR NOMBRE...: exige que cada archivo exista y no este vacio.
verificar_esperados() {
    local ej="$1" dir="$2" nombre
    shift 2
    for nombre in "$@"; do
        if [ ! -s "$dir/$nombre" ]; then
            echo "REPRODUCCION FALLA: ejercicio $ej: falta o esta vacio $dir/$nombre" >&2
            exit 1
        fi
    done
}

ej2() {
    local inicio=$SECONDS
    local dir="$RAIZ/ej2"
    local esperados=(emojis.png vae_loss.csv vae_loss.png vae_reconstruction.png vae_manifold.png
        vae_new_emoji.png plain_manifold.png salida-ej2.txt)
    construir
    mkdir -p "$dir"
    borrar_esperados "$dir" "${esperados[@]}"
    correr ej2 "$dir"
    verificar_esperados 2 "$dir" "${esperados[@]}"
    echo "REPRODUCCION OK: ejercicio 2 (${#esperados[@]} archivos en $dir, $((SECONDS - inicio)) s)"
}

case "$EJERCICIO" in
    2) ej2 ;;
esac
