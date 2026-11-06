"""Lector de las salidas regeneradas por scripts/regenerar.sh.

Toda cifra de la presentacion entra por aqui: las diapositivas no tipean
numeros, los leen de los reportes impresos, los CSV y las tablas Markdown que
escribe el codigo. Si falta una entrada la construccion se detiene con un
mensaje que nombra el archivo y el comando que lo crea.

Solo biblioteca estandar.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

EJERCICIOS = ("ej1a", "ej1b", "ej2")
_PNG = b"\x89PNG\r\n\x1a\n"


class FaltaEntrada(Exception):
    """Un archivo de salida no existe: hay que correr regenerar.sh."""


def campo(linea: str, clave: str) -> str:
    """Valor de `clave: valor` dentro de una linea con campos separados por ' | '."""
    prefijo = f"{clave}: "
    for parte in linea.split(" | "):
        if parte.startswith(prefijo):
            return parte[len(prefijo):].strip()
    raise ValueError(f"no hay campo '{clave}' en la linea: {linea!r}")


def numero(linea: str, patron: str) -> str:
    """Primer grupo de captura de la expresion regular `patron` sobre `linea`."""
    m = re.search(patron, linea)
    if m is None:
        raise ValueError(f"el patron {patron!r} no coincide con la linea: {linea!r}")
    if m.lastindex is None:
        raise ValueError(f"el patron {patron!r} no tiene grupo de captura")
    return m.group(1)


def _celdas(linea: str) -> list[str]:
    interior = linea.strip()
    if interior.startswith("|"):
        interior = interior[1:]
    if interior.endswith("|"):
        interior = interior[:-1]
    return [c.strip() for c in interior.split("|")]


def _es_separador(celdas: list[str]) -> bool:
    return all(re.fullmatch(r":?-+:?", c) for c in celdas)


class Salidas:
    """Acceso de solo lectura a la raiz de salidas (por defecto `output`)."""

    def __init__(self, raiz) -> None:
        self.raiz = Path(raiz)

    @staticmethod
    def _validar(ej: str) -> None:
        if ej not in EJERCICIOS:
            raise ValueError(f"ejercicio desconocido {ej!r}; se esperaba uno de {EJERCICIOS}")

    def ruta(self, ej: str, nombre: str) -> Path:
        self._validar(ej)
        ruta = self.raiz / ej / nombre
        if not ruta.is_file():
            raise FaltaEntrada(
                f"falta {ruta}; se crea con: bash scripts/regenerar.sh {ej[2:]}"
            )
        return ruta

    def lineas(self, ej: str, nombre: str) -> list[str]:
        """Lineas sin el salto final y sin ningun otro recorte (los espacios iniciales importan)."""
        texto = self.ruta(ej, nombre).read_text(encoding="utf-8")
        return texto.splitlines()

    def lineas_con_prefijo(self, ej: str, nombre: str, prefijo: str) -> list[str]:
        return [l for l in self.lineas(ej, nombre) if l.startswith(prefijo)]

    def linea(self, ej: str, nombre: str, prefijo: str) -> str:
        """La unica linea que empieza con `prefijo`; error si hay cero o varias."""
        halladas = self.lineas_con_prefijo(ej, nombre, prefijo)
        if len(halladas) != 1:
            raise ValueError(
                f"{ej}/{nombre}: se esperaba exactamente una linea que empiece con {prefijo!r} "
                f"y hay {len(halladas)}"
            )
        return halladas[0]

    def bloque(self, ej: str, nombre: str, prefijo: str, n: int) -> list[str]:
        """La primera linea que empieza con `prefijo` y las n-1 siguientes."""
        todas = self.lineas(ej, nombre)
        for i, l in enumerate(todas):
            if l.startswith(prefijo):
                if i + n > len(todas):
                    raise ValueError(
                        f"{ej}/{nombre}: el bloque que empieza con {prefijo!r} "
                        f"tiene menos de {n} lineas"
                    )
                return todas[i : i + n]
        raise ValueError(f"{ej}/{nombre}: ninguna linea empieza con {prefijo!r}")

    def csv(self, ej: str, nombre: str) -> list[dict[str, str]]:
        with self.ruta(ej, nombre).open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def tabla_md(self, ej: str, nombre: str, estudio: str) -> tuple[list[str], list[list[str]]]:
        """Encabezado y filas de la tabla bajo `## Estudio: ESTUDIO`."""
        titulo = f"## Estudio: {estudio}"
        todas = self.lineas(ej, nombre)
        for i, l in enumerate(todas):
            if l.rstrip() == titulo:
                break
        else:
            raise ValueError(f"{ej}/{nombre}: no hay la seccion {titulo!r}")
        tabla: list[list[str]] = []
        for l in todas[i + 1 :]:
            if l.startswith("## "):
                break
            if l.lstrip().startswith("|"):
                tabla.append(_celdas(l))
            elif tabla:
                break
        if not tabla:
            raise ValueError(f"{ej}/{nombre}: la seccion {titulo!r} no tiene tabla")
        encabezado = tabla[0]
        filas = [c for c in tabla[1:] if not _es_separador(c)]
        return encabezado, filas

    def png(self, ej: str, nombre: str) -> Path:
        ruta = self.ruta(ej, nombre)
        with ruta.open("rb") as f:
            if f.read(8) != _PNG:
                raise ValueError(f"{ruta} no es un PNG")
        return ruta

    def alt(self, ej: str, nombre: str) -> str:
        """Texto alternativo canonico `output/EJ/NOMBRE`, igual sea cual sea la raiz."""
        self._validar(ej)
        return f"output/{ej}/{nombre}"
