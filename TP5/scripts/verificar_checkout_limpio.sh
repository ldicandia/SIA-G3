#!/usr/bin/env bash
# Prueba el checkout limpio: copia solo las fuentes versionadas a un directorio temporal, las
# compila con las dependencias fijadas (--locked) y regenera un ejercicio ahi.
#
#   bash scripts/verificar_checkout_limpio.sh <1a|1b|2|todo> [--conservar] [--comparar RAIZ_REFERENCIA]
#
# Con --comparar, cada archivo regenerado en la copia se compara con el mismo archivo de
# RAIZ_REFERENCIA (la raiz de una corrida previa de regenerar.sh en el arbol de trabajo): los
# CSV, Markdown y PNG deben tener el mismo SHA-256 y los reportes salida-*.txt deben ser iguales
# una vez reemplazada la raiz de salida por RAIZ y sin las lineas con tiempos (" corridas en ").
set -euo pipefail

USO='uso: bash scripts/verificar_checkout_limpio.sh <1a|1b|2|todo> [--conservar] [--comparar RAIZ_REFERENCIA]'

cd "$(dirname "${BASH_SOURCE[0]}")/.."

EJERCICIO="${1:-}"
if [ -z "$EJERCICIO" ]; then
    echo "$USO" >&2
    exit 2
fi
shift || true

CONSERVAR=0
REFERENCIA=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --conservar)
            CONSERVAR=1
            shift
            ;;
        --comparar)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                echo "$USO" >&2
                exit 2
            fi
            REFERENCIA="${2%/}"
            shift 2
            ;;
        *)
            echo "$USO" >&2
            exit 2
            ;;
    esac
done

case "$EJERCICIO" in
    1a | 1b | 2 | todo) ;;
    *)
        echo "$USO" >&2
        exit 2
        ;;
esac

REFERENCIA_ABS=""
if [ -n "$REFERENCIA" ]; then
    if [ ! -d "$REFERENCIA" ]; then
        echo "CHECKOUT LIMPIO FALLA: paso comparar: no existe la raiz de referencia $REFERENCIA" >&2
        exit 2
    fi
    REFERENCIA_ABS="$(cd "$REFERENCIA" && pwd)"
fi

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

if [ -z "$REFERENCIA" ]; then
    echo "CHECKOUT LIMPIO OK: $EJERCICIO ($COPIADOS archivos copiados)"
    exit 0
fi

# Reemplaza "RAIZ_TEXTUAL/" por "RAIZ/" y descarta las lineas con tiempos.
normalizar() {
    local archivo="$1" raiz="$2" patron
    patron="$(printf '%s' "$raiz" | sed 's/[][\.*^$|/]/\\&/g')"
    sed "s|${patron}/|RAIZ/|g" "$archivo" | grep -v ' corridas en ' || true
}

COMPARADOS=0
DIFERENCIAS=0
while IFS= read -r -d '' archivo; do
    rel="${archivo#"$COPIA/$SALIDA_TMP"/}"
    ref="$REFERENCIA_ABS/$rel"
    if [ ! -f "$ref" ]; then
        echo "DIFERENCIA: falta en la referencia: $REFERENCIA/$rel" >&2
        DIFERENCIAS=$((DIFERENCIAS + 1))
        continue
    fi
    COMPARADOS=$((COMPARADOS + 1))
    case "$(basename "$rel")" in
        salida-*)
            if ! diff <(normalizar "$archivo" "$SALIDA_TMP") <(normalizar "$ref" "$REFERENCIA") >/dev/null; then
                echo "DIFERENCIA: reporte distinto tras normalizar: $rel" >&2
                diff <(normalizar "$archivo" "$SALIDA_TMP") <(normalizar "$ref" "$REFERENCIA") | head -20 >&2 || true
                DIFERENCIAS=$((DIFERENCIAS + 1))
            fi
            ;;
        *)
            if [ "$(sha256sum <"$archivo" | cut -d' ' -f1)" != "$(sha256sum <"$ref" | cut -d' ' -f1)" ]; then
                echo "DIFERENCIA: SHA-256 distinto: $rel" >&2
                DIFERENCIAS=$((DIFERENCIAS + 1))
            fi
            ;;
    esac
done < <(find "$COPIA/$SALIDA_TMP" -type f -print0 | sort -z)

if [ "$DIFERENCIAS" -gt 0 ]; then
    echo "CHECKOUT LIMPIO FALLA: paso comparar: $DIFERENCIAS diferencias con $REFERENCIA" >&2
    exit 1
fi
if [ "$COMPARADOS" -eq 0 ]; then
    echo "CHECKOUT LIMPIO FALLA: paso comparar: no se comparo ningun archivo" >&2
    exit 1
fi
echo "CHECKOUT LIMPIO OK: $EJERCICIO ($COPIADOS archivos copiados, $COMPARADOS comparados)"
