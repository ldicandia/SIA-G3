"""Disposiciones compartidas por las secciones de la presentacion.

Tres reglas de diseno (ver README, seccion Presentacion):
- poco texto en la diapositiva (lo que explica el orador va a las notas);
- cada figura ocupa el ancho o el alto del area de contenido;
- letra de 18 pt o mas en el cuerpo y de 16 pt o mas en las tablas.

Solo biblioteca estandar.
"""

from __future__ import annotations

from pptx_minimo import (
    ALTO_IN,
    ANCHO_IN,
    COLOR_AMBAR,
    CONTENIDO_Y_IN,
    MARGEN_IN,
    TAM_MINIMO_TABLA,
    TAM_MINIMO_TEXTO,
    Parrafo,
    altura_estimada,
    dimensiones_png,
)

ANCHO_UTIL = ANCHO_IN - 2 * MARGEN_IN
ALTO_UTIL = ALTO_IN - MARGEN_IN - CONTENIDO_Y_IN
Y_FIN = ALTO_IN - MARGEN_IN
TAM = TAM_MINIMO_TEXTO
TAM_TABLA = TAM_MINIMO_TABLA
SEPARACION = 0.3
HOLGURA = 0.02


def px(n: str) -> str:
    return "píxel" if n == "1" else "píxeles"


def enumerar(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " y " + items[-1]


def vinetas(textos: list[str]) -> list[Parrafo]:
    return [Parrafo(t, vineta=True) for t in textos]


def _tamano(salidas, ej: str, nombre: str):
    ruta = salidas.png(ej, nombre)
    ancho, alto = dimensiones_png(ruta.read_bytes(), str(ruta))
    return ruta, ancho, alto


def texto_justo(d, x: float, y: float, w: float, parrafos, *, tam: float = TAM) -> float:
    """Caja de texto con el alto que estima el escritor; devuelve la y donde termina."""
    if isinstance(parrafos, (str, Parrafo)):
        parrafos = [parrafos]
    ps = [p if isinstance(p, Parrafo) else Parrafo(p) for p in parrafos]
    alto = altura_estimada(ps, w, tam=tam) + HOLGURA
    d.texto(x, y, w, alto, ps, tam=tam)
    return y + alto


def figura_ancha(d, salidas, ej: str, nombre: str, pie=None, *, fraccion: float = 1.0) -> None:
    """Figura a lo ancho del contenido (o a `fraccion` de el, centrada) y, debajo, el pie."""
    ruta, ancho_px, alto_px = _tamano(salidas, ej, nombre)
    ancho = ANCHO_UTIL * fraccion
    alto = ancho * alto_px / ancho_px
    d.imagen(ruta, MARGEN_IN + (ANCHO_UTIL - ancho) / 2, CONTENIDO_Y_IN, ancho, alto, alt=salidas.alt(ej, nombre))
    if pie is not None:
        texto_justo(d, MARGEN_IN, CONTENIDO_Y_IN + alto + 0.08, ANCHO_UTIL, pie)


def figura_alta(d, salidas, ej: str, nombre: str) -> tuple[float, float]:
    """Figura al alto del contenido, a la izquierda; devuelve (x, ancho) de la columna libre."""
    ruta, ancho_px, alto_px = _tamano(salidas, ej, nombre)
    ancho = ALTO_UTIL * ancho_px / alto_px
    d.imagen(ruta, MARGEN_IN, CONTENIDO_Y_IN, ancho, ALTO_UTIL, alt=salidas.alt(ej, nombre))
    x = MARGEN_IN + ancho + SEPARACION
    return x, ANCHO_IN - MARGEN_IN - x


def figura_alta_centrada(d, salidas, ej: str, nombre: str) -> None:
    """Figura al alto del contenido, centrada, sin texto al lado."""
    ruta, ancho_px, alto_px = _tamano(salidas, ej, nombre)
    ancho = ALTO_UTIL * ancho_px / alto_px
    d.imagen(ruta, (ANCHO_IN - ancho) / 2, CONTENIDO_Y_IN, ancho, ALTO_UTIL, alt=salidas.alt(ej, nombre))


def diagrama(d, cajas, *, y: float = CONTENIDO_Y_IN + 0.1, alto: float = 1.3, hueco: float = 0.5) -> float:
    """Cadena de cajas (titulo, detalle, relleno) unidas por flechas; devuelve la y donde termina."""
    n = len(cajas)
    ancho = (ANCHO_UTIL - (n - 1) * hueco) / n
    for i, (titulo, detalle, relleno) in enumerate(cajas):
        x = MARGEN_IN + i * (ancho + hueco)
        d.caja(x, y, ancho, alto, [Parrafo(titulo, negrita=True), Parrafo(detalle)], tam=TAM, relleno=relleno)
        if i < n - 1:
            d.texto(x + ancho, y + (alto - 0.6) / 2, hueco, 0.6, "→", tam=24)
    return y + alto


def cifra(d, x: float, y: float, w: float, valor: str, etiqueta: str) -> float:
    """Cifra destacada con el alto justo para su etiqueta; devuelve la y donde termina."""
    ps = [Parrafo(valor, tam=44), Parrafo(etiqueta, tam=TAM)]
    alto = altura_estimada(ps, w, tam=TAM) + 0.2
    d.cifra(x, y, w, alto, valor, etiqueta)
    return y + alto


def destacado(texto: str, tam: float = 32) -> Parrafo:
    return Parrafo(texto, tam=tam, negrita=True, color=COLOR_AMBAR)


def lineas_md(salidas, ej: str, archivo: str, variantes: list[str]) -> list[str]:
    """Lineas literales de una tabla Markdown cuyas filas empiezan con esas variantes."""
    todas = salidas.lineas(ej, archivo)
    halladas = []
    for v in variantes:
        fila = [l for l in todas if l.startswith(f"| {v} |")]
        if len(fila) != 1:
            raise ValueError(f"{ej}/{archivo}: se esperaba una fila de la variante {v!r} y hay {len(fila)}")
        halladas.append(fila[0])
    return halladas
