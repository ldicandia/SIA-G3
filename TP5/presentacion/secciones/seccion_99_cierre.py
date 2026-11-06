"""Seccion 99: diapositiva de cierre, la cadena de reproduccion.

Sin cifras: solo comandos y dos frases. Es la ultima seccion por el orden de
nombre de archivo y la unica diapositiva que cierra la presentacion.
"""

from __future__ import annotations

from pptx_minimo import ALTO_IN, ANCHO_IN, CONTENIDO_Y_IN, MARGEN_IN, Parrafo, altura_estimada

TITULO = "Reproducibilidad: de un checkout limpio a la presentación"

PASOS = (
    ("Clonar y compilar", "cargo build --release --locked"),
    ("Regenerar los resultados", "bash scripts/regenerar.sh todo"),
    ("Construir la presentación", "bash scripts/presentacion.sh"),
    ("Verificar la consistencia", "python3 -B presentacion/verificar.py"),
)

POR_EJERCICIO = (
    ("Ej 1a", "bash scripts/regenerar.sh 1a"),
    ("Ej 1b", "bash scripts/regenerar.sh 1b"),
    ("Ej 2", "bash scripts/regenerar.sh 2"),
)

FRASE_USO = (
    "La presentación usa solo lo que regenera el código: las cifras salen de los reportes y "
    "archivos de salida, y cada gráfico es el PNG regenerado."
)
FRASE_CONTROLES = (
    "Dos comprobaciones independientes lo exigen. presentacion/verificar.py compara la "
    "presentación con los archivos regenerados en esa máquina, incluidos los bytes de cada PNG, "
    "que dependen de las tipografías del sistema: solo vale en la máquina que los regeneró. "
    "tests/deck_consistency.rs recalcula desde la biblioteca todo lo que la presentación cita y "
    "corre en cualquier máquina."
)


def agregar(pres, salidas) -> None:
    d = pres.diapositiva(TITULO)
    ancho_util = ANCHO_IN - 2 * MARGEN_IN
    y_fin = ALTO_IN - MARGEN_IN

    # (a) Cadena de cuatro cajas, cada una con su comando real.
    hueco = 0.2
    ancho_caja = (ancho_util - (len(PASOS) - 1) * hueco) / len(PASOS)
    alto_caja = 1.2
    for i, (paso, comando) in enumerate(PASOS):
        x = MARGEN_IN + i * (ancho_caja + hueco)
        d.caja(
            x, CONTENIDO_Y_IN, ancho_caja, alto_caja,
            [Parrafo(paso, negrita=True, tam=14), Parrafo(comando, tam=12)],
            tam=14,
        )
    y = CONTENIDO_Y_IN + alto_caja + 0.3

    # (b) Un comando por ejercicio.
    filas = [["Ejercicio", "Comando que lo regenera"]] + [list(f) for f in POR_EJERCICIO]
    alto_tabla = d.tabla(MARGEN_IN, y, ancho_util * 0.6, filas, anchos=[1, 3], tam=13)
    y += alto_tabla + 0.3

    # (c) Dos frases.
    parrafos = [Parrafo(FRASE_USO), Parrafo(FRASE_CONTROLES)]
    alto = min(y_fin - y, altura_estimada(parrafos, ancho_util, tam=15) + 0.02)
    d.texto(MARGEN_IN, y, ancho_util, alto, parrafos, tam=15)
