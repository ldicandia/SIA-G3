#!/usr/bin/env bash
# Prueba el checkout limpio: copia solo las fuentes versionadas a un directorio temporal, las
# compila con las dependencias fijadas (--locked) y regenera un ejercicio ahi.
#
#   bash scripts/verificar_checkout_limpio.sh <1a|1b|2|todo> [--conservar]
set -euo pipefail

USO='uso: bash scripts/verificar_checkout_limpio.sh <1a|1b|2|todo> [--conservar]'

cd "$(dirname "${BASH_SOURCE[0]}")/.."

EJERCICIO="${1:-}"
if [ -z "$EJERCICIO" ]; then
    echo "$USO" >&2
    exit 2
fi
shift || true

CONSERVAR=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --conservar)
            CONSERVAR=1
            shift
            ;;
        *)
            echo "$USO" >&2
            exit 2
            ;;
    esac
done

case "$EJERCICIO" in
    2) ;;
    *)
        echo "$USO" >&2
        exit 2
        ;;
esac

TMP="$(mktemp -d "${TMPDIR:-/tmp}/tp5-limpio.XXXXXX")"

# Borrado protegido: solo se borra si la variable no esta vacia, es un directorio existente y su
# nombre cumple el patron tp5-limpio.*
limpiar() {
    if [ "$CONSERVAR" -eq 1 ]; then
        echo "directorio temporal conservado: $TMP"
        return 0
    fi
    if [ -n "${TMP:-}" ] && [ -d "$TMP" ]; then
        case "$(basename "$TMP")" in
            tp5-limpio.*) rm -rf -- "$TMP" ;;
        esac
    fi
}
trap limpiar EXIT

COPIA="$TMP/tp5"
mkdir -p "$COPIA"

# Archivos versionados mas los nuevos no ignorados (asi la prueba sirve antes de confirmar los
# archivos nuevos). Los borrados del arbol de trabajo se ignoran.
git ls-files -z --cached --others --exclude-standard >"$TMP/lista"
tar --null --ignore-failed-read -T "$TMP/lista" -cf - | tar -xf - -C "$COPIA"
COPIADOS="$(find "$COPIA" -type f | wc -l | tr -d ' ')"

for requerido in Cargo.toml Cargo.lock src/main.rs docs/font.h scripts/regenerar.sh; do
    if [ ! -f "$COPIA/$requerido" ]; then
        echo "CHECKOUT LIMPIO FALLA: paso copiar: la copia no contiene $requerido" >&2
        exit 1
    fi
done
for prohibido in target output .planning; do
    if [ -e "$COPIA/$prohibido" ]; then
        echo "CHECKOUT LIMPIO FALLA: paso copiar: la copia contiene $prohibido" >&2
        exit 1
    fi
done
echo "copiados $COPIADOS archivos a $COPIA"

SALIDA_TMP="salida-limpia"

if ! (cd "$COPIA" && cargo build --release --locked); then
    echo "CHECKOUT LIMPIO FALLA: paso compilar con cargo build --release --locked" >&2
    exit 1
fi
if ! (cd "$COPIA" && bash scripts/regenerar.sh "$EJERCICIO" --salida "$SALIDA_TMP"); then
    echo "CHECKOUT LIMPIO FALLA: paso regenerar el ejercicio $EJERCICIO" >&2
    exit 1
fi

echo "CHECKOUT LIMPIO OK: $EJERCICIO ($COPIADOS archivos copiados)"
