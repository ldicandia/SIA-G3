"""Seccion 99: diapositiva de cierre, la cadena de reproduccion.

Sin cifras: en la diapositiva solo los comandos; lo que los explica va a las notas del
orador. Es la ultima seccion por el orden de nombre de archivo.
"""

from __future__ import annotations

from disposicion import ANCHO_UTIL
from pptx_minimo import CONTENIDO_Y_IN, MARGEN_IN

TITULO = "Reproducibilidad: del código a la presentación"

PASOS = (
    ("Compilar", "cargo build --release --locked"),
    ("Regenerar los resultados", "bash scripts/regenerar.sh todo"),
    ("Construir y verificar la presentación", "bash scripts/presentacion.sh --sin-regenerar"),
    ("Correr las pruebas", "cargo test --release"),
)


def agregar(pres, salidas) -> None:
    d = pres.diapositiva(TITULO)
    filas = [["Paso", "Comando"]] + [list(p) for p in PASOS]
    d.tabla(MARGEN_IN, CONTENIDO_Y_IN + 0.6, ANCHO_UTIL, filas, anchos=[1, 1.3], tam=20)
    d.notas(
        [
            "Mensaje: todo lo que se mostró sale de estos cuatro comandos, desde un checkout limpio.",
            "La presentación usa solo lo que regenera el código: las cifras de las diapositivas y de estas "
            "notas se leen de los reportes y archivos de salida, y cada gráfico es el PNG regenerado.",
            "Hay dos comprobaciones independientes. El verificador de la presentación la compara con los "
            "archivos regenerados en la máquina, incluidos los bytes de cada figura. Una prueba del proyecto "
            "recalcula desde la biblioteca lo que la presentación cita y corre en cualquier máquina.",
            "El comando que regenera los resultados acepta un ejercicio en lugar de todos, para rehacer uno solo.",
            "No hace falta instalar nada además de Rust y Python: la presentación se arma con la biblioteca "
            "estándar.",
        ]
    )
