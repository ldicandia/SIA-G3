#!/usr/bin/env bash
# Descarga las texturas de los bloques de mena del Ej 2 desde los servidores oficiales de Mojang.
#
#   bash scripts/descargar_menas.sh [--destino DIR]
#
# Las texturas son de Mojang y no se versionan: este script baja el cliente oficial de Minecraft
# 1.21.4 (el mismo archivo que baja el launcher), verifica su SHA-1, extrae las 18 menas y las 3
# rocas base a DIR (por defecto datos/menas, relativo a TP5) y verifica el SHA-256 de cada PNG, asi
# que el dataset es identico en cualquier maquina. Si las 21 texturas ya estan y sus hashes
# coinciden, no descarga nada. Necesita python3 (solo biblioteca estandar) y acceso a internet.
set -euo pipefail

USO='uso: bash scripts/descargar_menas.sh [--destino DIR]'
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DESTINO="datos/menas"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --destino)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                echo "$USO" >&2
                exit 2
            fi
            DESTINO="$2"
            shift 2
            ;;
        *)
            echo "$USO" >&2
            exit 2
            ;;
    esac
done

if ! command -v python3 >/dev/null 2>&1; then
    echo "falta python3 en el PATH (ver la seccion Requisitos del README.md)" >&2
    exit 2
fi

python3 -B - "$DESTINO" <<'PY'
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile

destino = sys.argv[1]
VERSION = "1.21.4"
MANIFIESTO = "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json"
JAR_SHA1 = "a7e5a6024bfd3cd614625aa05629adf760020304"
PREFIJO = "assets/minecraft/textures/block/"
TEXTURAS = {
    "coal_ore": "5088136494a0109d597c1bcdce6f8ba74b4284703ee71a95f4134db03e7df5f2",
    "iron_ore": "d3a50d2c689f8ade6445488f2edc68ca107c343ef09df8eaa27e51aa83d5a5ed",
    "copper_ore": "b83ac68ddccbd37b2fd61f80443e0c7b43fc5bbf9d33dec5b40a167fd0456ad6",
    "gold_ore": "e9228a6ea34328cf37dadef5953f4474ccc7321ab34bf1656dda3effb02eb69d",
    "redstone_ore": "99239d089f094c96afb8b47d98557851f3eb880e47d4ebc5fd3c79659896b1bf",
    "emerald_ore": "692ef7e7622bab160bc3624bf73697ad4ae7b86173213ab7d2367069ae5c1972",
    "lapis_ore": "190000ef0594f76f279cbbd0acff448ab1f9a32e788cc81beea380963082563d",
    "diamond_ore": "79abdde719a9bdf99279ab3f19b90da8e949c6c7cb7dcf453e0c3ce1f568577f",
    "deepslate_coal_ore": "a6e4d4c144d85be16fc67080fcc819e4c5d065af89ab900346b34e4ca34bdc88",
    "deepslate_iron_ore": "4565d56881e85ceef38107f0ba37c588788cb96ba1c8e269ddc716d972e7a951",
    "deepslate_copper_ore": "37d6d3fa66a15e1acda7c52910b0904ab8d1027163fcc81d5a5cd5733148d6e4",
    "deepslate_gold_ore": "f468e3f00c7393aa01c234058095610820e7dbdc0b26eb64044beadac647701c",
    "deepslate_redstone_ore": "5cd3c3b20045c4ab9fe5d3803a9304ec4d723209c51d2bd609126c90a6eb74af",
    "deepslate_emerald_ore": "29054e35459ee1a9e150d5f43432032095dde90291e9aca52f57b908ea2bef05",
    "deepslate_lapis_ore": "e8855090fee951faa4ee5e29714f033aa56778f155eaaeb72a3b97f33382db19",
    "deepslate_diamond_ore": "7ff158066e1a9a9832b52b1eec7a54071c95478b18bd16207b322f41756753b0",
    "nether_gold_ore": "2e8b34d2f640c171bf26e63ffe279f29d2b3acad34d60fd240c6d46bec5aedc0",
    "nether_quartz_ore": "b8083d01cbaf82ad2f6101646dcfa64bcedaa49f1c9363dca478ee0a1c378978",
    "stone": "33ac70ba8cb701087642a12c5fd29c1ba3006dc4730e9b014ed3a19ce7cdda7b",
    "deepslate": "b4c5e7f5538608a371fd4c1f6b398e4b76f723702d10144432636a85a1a61832",
    "netherrack": "3c0208b749c5393b9bad9d57ed3d830326a02b8cbd8444e96d4d0d277f85625f",
}


def sha256(datos):
    return hashlib.sha256(datos).hexdigest()


def completas():
    for nombre, esperado in TEXTURAS.items():
        ruta = os.path.join(destino, nombre + ".png")
        if not os.path.isfile(ruta):
            return False
        with open(ruta, "rb") as f:
            if sha256(f.read()) != esperado:
                return False
    return True


def bajar(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


if completas():
    print(f"MENAS OK: {len(TEXTURAS)} texturas ya presentes en {destino}")
    sys.exit(0)

try:
    manifiesto = json.loads(bajar(MANIFIESTO))
    version = next((v for v in manifiesto["versions"] if v["id"] == VERSION), None)
    if version is None:
        sys.exit(f"el manifiesto de Mojang no lista la version {VERSION}")
    cliente = json.loads(bajar(version["url"]))["downloads"]["client"]
    print(f"descargando el cliente de Minecraft {VERSION} ({cliente['size'] / 1e6:.1f} MB) desde {cliente['url']}")
    jar = bajar(cliente["url"])
except OSError as e:
    sys.exit(f"no se pudo descargar desde los servidores de Mojang: {e}")

if hashlib.sha1(jar).hexdigest() != JAR_SHA1:
    sys.exit(f"el SHA-1 del cliente descargado no es {JAR_SHA1}")

os.makedirs(destino, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(jar)) as z:
    for nombre, esperado in TEXTURAS.items():
        datos = z.read(PREFIJO + nombre + ".png")
        if sha256(datos) != esperado:
            sys.exit(f"{nombre}.png no tiene el SHA-256 esperado")
        with open(os.path.join(destino, nombre + ".png"), "wb") as f:
            f.write(datos)
print(f"MENAS OK: {len(TEXTURAS)} texturas extraidas y verificadas en {destino}")
PY
