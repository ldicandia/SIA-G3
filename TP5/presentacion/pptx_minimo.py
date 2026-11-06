"""Escritor minimo de archivos PowerPoint (.pptx) con la biblioteca estandar.

Un .pptx es un ZIP de partes XML. Este modulo escribe el conjunto de partes que
emiten python-pptx y pptxgenjs (tipos de contenido, relaciones, docProps,
presentacion, patron, dos disenos, tema, diapositivas y medios) para que
PowerPoint y LibreOffice lo abran sin pedir reparacion.

Principios del modulo (ver el plan 04-02):
- Solo biblioteca estandar: un checkout limpio reconstruye la presentacion con
  python3 y nada que instalar.
- Nada se recorta en silencio: un texto que no entra en su caja, una forma que
  sale de la diapositiva o de los margenes, un caracter de control o un PNG
  ilegible levantan ErrorDeLayout al construir.
- Bytes deterministas: miembros sin comprimir, orden fijo, fechas fijas.

Unidades de la API: pulgadas (float). Texto en espanol.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

# --------------------------------------------------------------------------
# Constantes publicas
# --------------------------------------------------------------------------

ANCHO_IN = 13.333
ALTO_IN = 7.5
MARGEN_IN = 0.5
# Donde conviene empezar el contenido debajo del titulo (el titulo ocupa hasta 1.55).
CONTENIDO_Y_IN = 1.65

EMU_POR_PULGADA = 914400
ANCHO_EMU = 12192000
ALTO_EMU = 6858000

# Paleta: un verde azulado profundo para titulos y cajas, un ambar para las
# cifras destacadas y un gris claro para los bloques monoespaciados.
COLOR_TEAL = "0B5D66"
COLOR_AMBAR = "B45309"
COLOR_TEXTO = "1F2933"
COLOR_GRIS_CLARO = "F0F0F0"
COLOR_FONDO_CIFRA = "EAF2F3"
COLOR_BLANCO = "FFFFFF"
COLOR_BORDE_TABLA = "C9D3D6"
COLOR_FILA_TABLA = "F7FAFA"

FUENTE_TEXTO = "Calibri"
FUENTE_MONO = "Courier New"

TAM_TITULO = 28
TAM_PORTADA = 40
TAM_MINIMO_TEXTO = 10
TAM_MINIMO_TABLA = 11

# Estimacion de la altura del texto (ver ErrorDeLayout).
ANCHO_GLIFO_PROPORCIONAL = 0.52
ANCHO_GLIFO_MONO = 0.60
ALTO_LINEA = 1.2
RELLENO_LATERAL_IN = 0.1
RELLENO_VERTICAL_IN = 0.05
SANGRIA_VINETA_IN = 0.25

_EPS = 1e-6
_FECHA_FIJA = (2026, 1, 1, 0, 0, 0)
_FECHA_ISO = "2026-01-01T00:00:00Z"

TITULO_X, TITULO_Y, TITULO_W, TITULO_H = 0.5, 0.5, 12.333, 1.05
PORTADA_X, PORTADA_Y, PORTADA_W, PORTADA_H = 0.9, 2.3, 11.533, 1.6


class ErrorDeLayout(ValueError):
    """Un texto no entra en su caja, una forma sale de la diapositiva, un
    caracter de control en el texto o un PNG ilegible."""


# --------------------------------------------------------------------------
# Utilidades de validacion y medida
# --------------------------------------------------------------------------


def emu(pulgadas: float) -> int:
    """Pulgadas a EMU, entero (914400 por pulgada)."""
    return int(round(pulgadas * EMU_POR_PULGADA))


def validar_texto(texto: str, contexto: str = "texto") -> str:
    """Rechaza caracteres de control (salvo tabulacion y salto de linea) y los
    que XML 1.0 no admite. Devuelve el texto sin cambios."""
    if not isinstance(texto, str):
        raise ErrorDeLayout(f"{contexto}: se esperaba texto y llego {type(texto).__name__}")
    for car in texto:
        if car in "\t\n":
            continue
        o = ord(car)
        if (
            unicodedata.category(car) == "Cc"
            or 0xD800 <= o <= 0xDFFF
            or o in (0xFFFE, 0xFFFF)
        ):
            raise ErrorDeLayout(
                f"{contexto}: caracter no permitido U+{o:04X} (control o invalido en XML)"
            )
    return texto


def _atributo(valor: str, contexto: str) -> str:
    validar_texto(valor, contexto)
    if "\n" in valor or "\t" in valor:
        raise ErrorDeLayout(f"{contexto}: no admite saltos de linea ni tabulaciones")
    return escape(valor, {'"': "&quot;"})


def _color(valor: str | None, contexto: str) -> str | None:
    if valor is None:
        return None
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", valor):
        raise ErrorDeLayout(
            f"{contexto}: color '{valor}' invalido; se espera hexadecimal de 6 digitos sin '#'"
        )
    return valor.upper()


def _es_oscuro(color: str) -> bool:
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    return 0.299 * r + 0.587 * g + 0.114 * b < 140


@dataclass
class Parrafo:
    texto: str
    tam: float | None = None
    negrita: bool = False
    vineta: bool = False
    color: str | None = None


def _normalizar(parrafos) -> list[Parrafo]:
    """str o Parrafo a lista de Parrafo; un salto de linea parte el parrafo."""
    if isinstance(parrafos, (str, Parrafo)):
        parrafos = [parrafos]
    salida: list[Parrafo] = []
    for p in parrafos:
        if isinstance(p, str):
            p = Parrafo(p)
        validar_texto(p.texto, "parrafo")
        for parte in p.texto.split("\n"):
            salida.append(
                Parrafo(parte, tam=p.tam, negrita=p.negrita, vineta=p.vineta, color=p.color)
            )
    return salida


def _lineas_estimadas(texto: str, ancho_util: float, tam: float, mono: bool, sangria: float) -> int:
    ancho_car = (ANCHO_GLIFO_MONO if mono else ANCHO_GLIFO_PROPORCIONAL) * tam / 72.0
    disponible = ancho_util - sangria
    if disponible <= 0:
        raise ErrorDeLayout("la caja es mas angosta que su relleno")
    return max(1, math.ceil(len(texto) * ancho_car / disponible - 1e-9))


def altura_estimada(
    parrafos: list[Parrafo], ancho_in: float, *, tam: float, mono: bool = False
) -> float:
    """Altura estimada en pulgadas del texto dentro de una caja de ancho `ancho_in`.

    Ancho medio de glifo: 0.52 * tam (proporcional) o 0.60 * tam (monoespaciado);
    alto de linea 1.2 * tam; relleno lateral 0.1 y vertical 0.05 por lado;
    vinetas con 0.25 de sangria; el ajuste de linea es el techo del ancho del
    texto sobre el ancho util."""
    ancho_util = ancho_in - 2 * RELLENO_LATERAL_IN
    total = 0.0
    for p in parrafos:
        t = p.tam if p.tam is not None else tam
        sangria = SANGRIA_VINETA_IN if p.vineta else 0.0
        n = _lineas_estimadas(p.texto, ancho_util, t, mono, sangria)
        total += n * ALTO_LINEA * t / 72.0
    return total + 2 * RELLENO_VERTICAL_IN


def dimensiones_png(datos: bytes, contexto: str = "imagen") -> tuple[int, int]:
    """Ancho y alto de un PNG a partir de su cabecera (firma + IHDR)."""
    if len(datos) < 24 or datos[:8] != b"\x89PNG\r\n\x1a\n" or datos[12:16] != b"IHDR":
        raise ErrorDeLayout(f"{contexto}: no es un PNG legible")
    ancho, alto = struct.unpack(">II", datos[16:24])
    if ancho <= 0 or alto <= 0:
        raise ErrorDeLayout(f"{contexto}: PNG con dimensiones invalidas")
    return ancho, alto


def ajustar_imagen(
    ancho_px: int, alto_px: int, x: float, y: float, w: float, h: float
) -> tuple[float, float, float, float]:
    """Caja (x, y, w, h) de la imagen dentro de la caja dada: conserva la
    proporcion y queda centrada."""
    escala = min(w / ancho_px, h / alto_px)
    nw, nh = ancho_px * escala, alto_px * escala
    return x + (w - nw) / 2.0, y + (h - nh) / 2.0, nw, nh


# --------------------------------------------------------------------------
# Fragmentos XML
# --------------------------------------------------------------------------

_DECL = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
_NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
_NS_ALL = f'xmlns:a="{_NS_A}" xmlns:r="{_NS_R}" xmlns:p="{_NS_P}"'

_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_REL_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"

_CT = "application/vnd.openxmlformats-officedocument"
_CT_XML = "application/xml"


def _xfrm(x: float, y: float, w: float, h: float, tag: str = "a:xfrm") -> str:
    return (
        f'<{tag}><a:off x="{emu(x)}" y="{emu(y)}"/>'
        f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></{tag}>'
    )


def _relleno(color: str) -> str:
    return f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'


def _run(texto: str, tam: float, negrita: bool, color: str, mono: bool) -> str:
    sz = int(round(tam * 100))
    b = ' b="1"' if negrita else ""
    fuente = (
        f'<a:latin typeface="{FUENTE_MONO}"/><a:cs typeface="{FUENTE_MONO}"/>' if mono else ""
    )
    return (
        f'<a:r><a:rPr lang="es-AR" sz="{sz}"{b}>{_relleno(color)}{fuente}</a:rPr>'
        f"<a:t>{escape(texto)}</a:t></a:r>"
    )


def _parrafo_xml(
    p: Parrafo, tam: float, negrita: bool, color: str, mono: bool, alineacion: str | None
) -> str:
    t = p.tam if p.tam is not None else tam
    ppr_attrs = ""
    hijos = ""
    if p.vineta:
        ppr_attrs = f' marL="{emu(SANGRIA_VINETA_IN)}" indent="-{emu(SANGRIA_VINETA_IN)}"'
        hijos = '<a:buFont typeface="Arial"/><a:buChar char="&#8226;"/>'
    if alineacion:
        ppr_attrs += f' algn="{alineacion}"'
    ppr = f"<a:pPr{ppr_attrs}>{hijos}</a:pPr>" if (ppr_attrs or hijos) else ""
    run = _run(p.texto, t, negrita or p.negrita, p.color or color, mono)
    return f"<a:p>{ppr}{run}</a:p>"


def _cuerpo(
    parrafos: list[Parrafo],
    tam: float,
    negrita: bool,
    color: str,
    mono: bool,
    *,
    alineacion: str | None = None,
    ancla: str = "t",
) -> str:
    ps = "".join(_parrafo_xml(p, tam, negrita, color, mono, alineacion) for p in parrafos)
    return (
        '<p:txBody><a:bodyPr wrap="square" lIns="91440" tIns="45720" rIns="91440" '
        f'bIns="45720" rtlCol="0" anchor="{ancla}"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
        f"{ps}</p:txBody>"
    )


# --------------------------------------------------------------------------
# Diapositiva
# --------------------------------------------------------------------------


class Diapositiva:
    """Una diapositiva. Se crea con Presentacion.diapositiva()."""

    def __init__(self, presentacion: "Presentacion", titulo: str, portada: bool) -> None:
        validar_texto(titulo, "titulo de diapositiva")
        if "\n" in titulo:
            raise ErrorDeLayout(f"titulo '{titulo}': no admite saltos de linea")
        self._pres = presentacion
        self.titulo = titulo
        self.portada = portada
        self._formas: list[str] = []
        self._siguiente_id = 3  # 1 es el grupo raiz y 2 el titulo
        self._imagenes: list[tuple[str, str]] = []  # (id de relacion, nombre de medio)
        self._n_imagenes = 0
        self._verificar_titulo()

    # -- validacion -------------------------------------------------------

    def _error(self, forma: str, mensaje: str) -> ErrorDeLayout:
        return ErrorDeLayout(f"Diapositiva «{self.titulo}», forma «{forma}»: {mensaje}")

    def _verificar_caja(self, forma: str, x: float, y: float, w: float, h: float) -> None:
        if w <= 0 or h <= 0:
            raise self._error(forma, f"tamano invalido ({w} x {h})")
        if (
            x < MARGEN_IN - _EPS
            or y < MARGEN_IN - _EPS
            or x + w > ANCHO_IN - MARGEN_IN + _EPS
            or y + h > ALTO_IN - MARGEN_IN + _EPS
        ):
            raise self._error(
                forma,
                f"sale de los margenes de {MARGEN_IN} pulgadas "
                f"(x={x:.3f}, y={y:.3f}, w={w:.3f}, h={h:.3f}; "
                f"area util {MARGEN_IN}..{ANCHO_IN - MARGEN_IN:.3f} por {MARGEN_IN}..{ALTO_IN - MARGEN_IN:.3f})",
            )

    def _verificar_ajuste(
        self, forma: str, parrafos: list[Parrafo], w: float, h: float, tam: float, mono: bool
    ) -> None:
        try:
            necesaria = altura_estimada(parrafos, w, tam=tam, mono=mono)
        except ErrorDeLayout as e:
            raise self._error(forma, str(e)) from None
        if necesaria > h + _EPS:
            raise self._error(
                forma,
                f"el texto no entra: necesita {necesaria:.2f} pulgadas de alto y la caja tiene {h:.2f}",
            )

    def _verificar_titulo(self) -> None:
        if self.portada:
            x, y, w, h, tam = PORTADA_X, PORTADA_Y, PORTADA_W, PORTADA_H, TAM_PORTADA
        else:
            x, y, w, h, tam = TITULO_X, TITULO_Y, TITULO_W, TITULO_H, TAM_TITULO
        self._verificar_caja("titulo", x, y, w, h)
        self._verificar_ajuste("titulo", [Parrafo(self.titulo)], w, h, tam, False)

    def _id(self) -> int:
        i = self._siguiente_id
        self._siguiente_id += 1
        return i

    def _nombre(self, nombre: str | None, defecto: str) -> str:
        return _atributo(nombre if nombre is not None else defecto, "nombre de forma")

    def _verificar_tam(self, forma: str, tam: float, minimo: float) -> None:
        if tam < minimo:
            raise self._error(forma, f"tamano de letra {tam} menor que el minimo {minimo}")

    # -- formas -----------------------------------------------------------

    def texto(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        parrafos,
        *,
        tam: float = 16,
        negrita: bool = False,
        color: str | None = None,
        mono: bool = False,
        nombre: str | None = None,
    ) -> None:
        sid = self._id()
        forma = nombre if nombre is not None else f"Texto {sid}"
        ps = _normalizar(parrafos)
        self._verificar_caja(forma, x, y, w, h)
        self._verificar_tam(forma, tam, TAM_MINIMO_TEXTO)
        for p in ps:
            if p.tam is not None:
                self._verificar_tam(forma, p.tam, TAM_MINIMO_TEXTO)
        self._verificar_ajuste(forma, ps, w, h, tam, mono)
        col = _color(color, forma) or COLOR_TEXTO
        for p in ps:
            p.color = _color(p.color, forma)
        self._formas.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{self._nombre(nombre, forma)}"/>'
            '<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{_xfrm(x, y, w, h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>'
            f"{_cuerpo(ps, tam, negrita, col, mono)}</p:sp>"
        )

    def codigo(
        self, x: float, y: float, w: float, h: float, lineas, *, fuente: str, tam: float = 11
    ) -> None:
        sid = self._id()
        forma = f"salida: {fuente}"
        nombre = self._nombre(forma, forma)
        lineas = list(lineas)
        for linea in lineas:
            validar_texto(linea, forma)
            if "\n" in linea:
                raise self._error(forma, "cada linea del bloque debe ser un solo renglon")
        ps = [Parrafo(l) for l in lineas]
        self._verificar_caja(forma, x, y, w, h)
        self._verificar_tam(forma, tam, TAM_MINIMO_TEXTO)
        self._verificar_ajuste(forma, ps, w, h, tam, True)
        self._formas.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{nombre}"/>'
            '<p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{_xfrm(x, y, w, h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            f"{_relleno(COLOR_GRIS_CLARO)}<a:ln><a:noFill/></a:ln></p:spPr>"
            f"{_cuerpo(ps, tam, False, COLOR_TEXTO, True)}</p:sp>"
        )

    def imagen(self, ruta, x: float, y: float, w: float, h: float, *, alt: str) -> None:
        sid = self._id()
        ruta = Path(ruta)
        forma = f"Imagen {sid}"
        if not alt or not alt.strip():
            raise self._error(forma, "falta el texto alternativo de la imagen")
        descr = _atributo(alt, f"{forma}: texto alternativo")
        self._verificar_caja(forma, x, y, w, h)
        try:
            datos = ruta.read_bytes()
        except OSError as e:
            raise self._error(forma, f"no se pudo leer {ruta}: {e}") from None
        try:
            ancho_px, alto_px = dimensiones_png(datos, str(ruta))
        except ErrorDeLayout as e:
            raise self._error(forma, str(e)) from None
        ix, iy, iw, ih = ajustar_imagen(ancho_px, alto_px, x, y, w, h)
        medio = self._pres._registrar_medio(datos)
        rid = f"rId{2 + len(self._imagenes)}"
        self._imagenes.append((rid, medio))
        self._n_imagenes += 1
        self._formas.append(
            f'<p:pic><p:nvPicPr><p:cNvPr id="{sid}" name="{forma}" descr="{descr}"/>'
            '<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
            f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
            f'<p:spPr>{_xfrm(ix, iy, iw, ih)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
        )

    def tabla(
        self,
        x: float,
        y: float,
        w: float,
        filas,
        *,
        anchos=None,
        tam: float = 12,
        encabezado: bool = True,
    ) -> float:
        """Tabla nativa. Devuelve el alto total en pulgadas (para apilar formas)."""
        sid = self._id()
        forma = f"Tabla {sid}"
        filas = [[("" if c is None else str(c)) for c in fila] for fila in filas]
        if not filas or not filas[0]:
            raise self._error(forma, "la tabla no tiene filas")
        ncol = len(filas[0])
        for fila in filas:
            if len(fila) != ncol:
                raise self._error(forma, "todas las filas deben tener la misma cantidad de celdas")
            for c in fila:
                validar_texto(c, forma)
                if "\n" in c:
                    raise self._error(forma, "una celda no admite saltos de linea")
        self._verificar_tam(forma, tam, TAM_MINIMO_TABLA)
        if anchos is None:
            pesos = [1.0] * ncol
        else:
            pesos = [float(a) for a in anchos]
            if len(pesos) != ncol or any(a <= 0 for a in pesos):
                raise self._error(forma, "anchos debe tener un valor positivo por columna")
        total_peso = sum(pesos)
        anchos_in = [w * p / total_peso for p in pesos]
        # Alto de cada fila: el mayor entre sus celdas.
        alturas = []
        for i, fila in enumerate(filas):
            mayor = 0.0
            for c, ancho_col in zip(fila, anchos_in):
                negr = encabezado and i == 0
                est = altura_estimada([Parrafo(c, negrita=negr)], ancho_col, tam=tam)
                mayor = max(mayor, est)
            alturas.append(mayor)
        alto_total = sum(alturas)
        self._verificar_caja(forma, x, y, w, alto_total)
        # Anchos en EMU: la ultima columna absorbe el resto para sumar exactamente w.
        anchos_emu = [emu(a) for a in anchos_in]
        anchos_emu[-1] = emu(w) - sum(anchos_emu[:-1])
        grilla = "".join(f'<a:gridCol w="{a}"/>' for a in anchos_emu)
        filas_xml = []
        for i, fila in enumerate(filas):
            es_enc = encabezado and i == 0
            celdas = []
            for c in fila:
                color_txt = COLOR_BLANCO if es_enc else COLOR_TEXTO
                fondo = COLOR_TEAL if es_enc else (COLOR_FILA_TABLA if i % 2 == 0 else COLOR_BLANCO)
                borde = (
                    f'<a:lnB w="6350"><a:solidFill><a:srgbClr val="{COLOR_BORDE_TABLA}"/></a:solidFill></a:lnB>'
                )
                p = Parrafo(c, negrita=es_enc)
                celdas.append(
                    "<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>"
                    f"{_parrafo_xml(p, tam, es_enc, color_txt, False, None)}</a:txBody>"
                    '<a:tcPr marL="91440" marR="91440" marT="45720" marB="45720" anchor="ctr">'
                    f"{borde}{_relleno(fondo)}</a:tcPr></a:tc>"
                )
            filas_xml.append(f'<a:tr h="{emu(alturas[i])}">{"".join(celdas)}</a:tr>')
        enc_attr = ' firstRow="1"' if encabezado else ""
        self._formas.append(
            f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{sid}" name="{forma}"/>'
            '<p:cNvGraphicFramePr><a:graphicFrameLocks noGrp="1"/></p:cNvGraphicFramePr><p:nvPr/></p:nvGraphicFramePr>'
            f'{_xfrm(x, y, w, alto_total, "p:xfrm")}'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">'
            f'<a:tbl><a:tblPr{enc_attr}/><a:tblGrid>{grilla}</a:tblGrid>{"".join(filas_xml)}</a:tbl>'
            "</a:graphicData></a:graphic></p:graphicFrame>"
        )
        return alto_total

    def caja(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        texto,
        *,
        tam: float = 14,
        relleno: str | None = None,
        color: str | None = None,
    ) -> None:
        sid = self._id()
        forma = f"Caja {sid}"
        ps = _normalizar(texto)
        self._verificar_caja(forma, x, y, w, h)
        self._verificar_tam(forma, tam, TAM_MINIMO_TEXTO)
        self._verificar_ajuste(forma, ps, w, h, tam, False)
        fondo = _color(relleno, forma) or COLOR_TEAL
        col = _color(color, forma) or (COLOR_BLANCO if _es_oscuro(fondo) else COLOR_TEXTO)
        self._formas.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{forma}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{_xfrm(x, y, w, h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            f"{_relleno(fondo)}<a:ln><a:noFill/></a:ln></p:spPr>"
            f"{_cuerpo(ps, tam, False, col, False, alineacion='ctr', ancla='ctr')}</p:sp>"
        )

    def cifra(self, x: float, y: float, w: float, h: float, valor: str, etiqueta: str) -> None:
        sid = self._id()
        forma = f"Cifra {sid}"
        validar_texto(valor, forma)
        validar_texto(etiqueta, forma)
        ps = [
            Parrafo(valor, tam=44, negrita=True, color=COLOR_AMBAR),
            Parrafo(etiqueta, tam=14),
        ]
        self._verificar_caja(forma, x, y, w, h)
        self._verificar_ajuste(forma, ps, w, h, 14, False)
        self._formas.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{forma}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{_xfrm(x, y, w, h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            f"{_relleno(COLOR_FONDO_CIFRA)}<a:ln><a:noFill/></a:ln></p:spPr>"
            f"{_cuerpo(ps, 14, False, COLOR_TEXTO, False, alineacion='ctr', ancla='ctr')}</p:sp>"
        )

    # -- serializacion ------------------------------------------------------

    def _xml(self) -> str:
        if self.portada:
            ph = '<p:ph type="ctrTitle"/>'
            x, y, w, h, tam, alin, ancla = (
                PORTADA_X, PORTADA_Y, PORTADA_W, PORTADA_H, TAM_PORTADA, ' algn="ctr"', "ctr",
            )
        else:
            ph = '<p:ph type="title"/>'
            x, y, w, h, tam, alin, ancla = (
                TITULO_X, TITULO_Y, TITULO_W, TITULO_H, TAM_TITULO, ' algn="l"', "t",
            )
        titulo = (
            '<p:sp><p:nvSpPr><p:cNvPr id="2" name="Titulo"/>'
            f'<p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr>{ph}</p:nvPr></p:nvSpPr>'
            f"<p:spPr>{_xfrm(x, y, w, h)}</p:spPr>"
            f'<p:txBody><a:bodyPr anchor="{ancla}"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
            f"<a:p><a:pPr{alin}/>{_run(self.titulo, tam, True, COLOR_TEAL, False)}</a:p></p:txBody></p:sp>"
        )
        return (
            f"{_DECL}<p:sld {_NS_ALL}><p:cSld><p:spTree>"
            '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
            '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
            f'{titulo}{"".join(self._formas)}'
            "</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
        )

    def _rels(self) -> str:
        capa = 1 if self.portada else 2
        rels = [
            f'<Relationship Id="rId1" Type="{_REL}/slideLayout" Target="../slideLayouts/slideLayout{capa}.xml"/>'
        ]
        for rid, medio in self._imagenes:
            rels.append(f'<Relationship Id="{rid}" Type="{_REL}/image" Target="../media/{medio}"/>')
        return f'{_DECL}<Relationships xmlns="{_REL_PKG}">{"".join(rels)}</Relationships>'


# --------------------------------------------------------------------------
# Partes fijas (patron, disenos, tema, propiedades)
# --------------------------------------------------------------------------


def _estilo_nivel(tag: str, tam: int, negrita: bool, color_esquema: str, extra: str = "") -> str:
    b = ' b="1"' if negrita else ""
    return (
        f'<a:{tag} marL="0" algn="l" defTabSz="914400" rtl="0" eaLnBrk="1" latinLnBrk="0" hangingPunct="1">'
        f'{extra}<a:defRPr sz="{tam}"{b} kern="1200"><a:solidFill><a:schemeClr val="{color_esquema}"/></a:solidFill>'
        '<a:latin typeface="+mn-lt"/><a:ea typeface="+mn-ea"/><a:cs typeface="+mn-cs"/></a:defRPr>'
        f"</a:{tag}>"
    )


def _texto_por_defecto() -> str:
    return (
        '<a:defPPr><a:defRPr lang="es-AR"/></a:defPPr>'
        + _estilo_nivel("lvl1pPr", 1800, False, "tx1")
    )


def _ph_sp(
    sid: int, nombre: str, ph: str, x: float, y: float, w: float, h: float, aviso: str, ancla: str = "t"
) -> str:
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{nombre}"/>'
        f'<p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr>{ph}</p:nvPr></p:nvSpPr>'
        f"<p:spPr>{_xfrm(x, y, w, h)}</p:spPr>"
        f'<p:txBody><a:bodyPr vert="horz" lIns="91440" tIns="45720" rIns="91440" bIns="45720" '
        f'rtlCol="0" anchor="{ancla}"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
        f'<a:p><a:r><a:rPr lang="es-AR"/><a:t>{escape(aviso)}</a:t></a:r></a:p></p:txBody></p:sp>'
    )


_GRUPO_RAIZ = (
    '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
    '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
    '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
)


def _parte_patron() -> str:
    titulo = _ph_sp(
        2, "Marcador de titulo", '<p:ph type="title"/>',
        TITULO_X, TITULO_Y, TITULO_W, TITULO_H, "Haga clic para modificar el estilo de titulo",
    )
    cuerpo = _ph_sp(
        3, "Marcador de texto", '<p:ph type="body" idx="1"/>',
        MARGEN_IN, CONTENIDO_Y_IN, TITULO_W, ALTO_IN - MARGEN_IN - CONTENIDO_Y_IN,
        "Haga clic para modificar el estilo de texto",
    )
    estilo_titulo = (
        "<p:titleStyle>"
        '<a:lvl1pPr algn="l" defTabSz="914400" rtl="0" eaLnBrk="1" latinLnBrk="0" hangingPunct="1">'
        f'<a:defRPr sz="{TAM_TITULO * 100}" b="1" kern="1200"><a:solidFill><a:schemeClr val="tx2"/></a:solidFill>'
        '<a:latin typeface="+mj-lt"/><a:ea typeface="+mj-ea"/><a:cs typeface="+mj-cs"/></a:defRPr></a:lvl1pPr>'
        "</p:titleStyle>"
    )
    estilo_cuerpo = (
        "<p:bodyStyle>"
        '<a:lvl1pPr marL="228600" indent="-228600" algn="l" defTabSz="914400" rtl="0" eaLnBrk="1" '
        'latinLnBrk="0" hangingPunct="1"><a:buFont typeface="Arial"/><a:buChar char="&#8226;"/>'
        '<a:defRPr sz="1800" kern="1200"><a:solidFill><a:schemeClr val="tx1"/></a:solidFill>'
        '<a:latin typeface="+mn-lt"/><a:ea typeface="+mn-ea"/><a:cs typeface="+mn-cs"/></a:defRPr></a:lvl1pPr>'
        "</p:bodyStyle>"
    )
    otro = f"<p:otherStyle>{_texto_por_defecto()}</p:otherStyle>"
    return (
        f"{_DECL}<p:sldMaster {_NS_ALL}><p:cSld>"
        '<p:bg><p:bgRef idx="1001"><a:schemeClr val="bg1"/></p:bgRef></p:bg>'
        f"<p:spTree>{_GRUPO_RAIZ}{titulo}{cuerpo}</p:spTree></p:cSld>"
        '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" '
        'accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
        '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/>'
        '<p:sldLayoutId id="2147483650" r:id="rId2"/></p:sldLayoutIdLst>'
        f"<p:txStyles>{estilo_titulo}{estilo_cuerpo}{otro}</p:txStyles></p:sldMaster>"
    )


def _parte_diseno_portada() -> str:
    titulo = _ph_sp(
        2, "Titulo", '<p:ph type="ctrTitle"/>',
        PORTADA_X, PORTADA_Y, PORTADA_W, PORTADA_H, "Haga clic para modificar el estilo de titulo", "ctr",
    )
    sub = _ph_sp(
        3, "Subtitulo", '<p:ph type="subTitle" idx="1"/>',
        PORTADA_X, 4.2, PORTADA_W, 1.2, "Haga clic para modificar el subtitulo",
    )
    return (
        f'{_DECL}<p:sldLayout {_NS_ALL} type="title" preserve="1"><p:cSld name="Diapositiva de titulo">'
        f"<p:spTree>{_GRUPO_RAIZ}{titulo}{sub}</p:spTree></p:cSld>"
        "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>"
    )


def _parte_diseno_titulo() -> str:
    titulo = _ph_sp(
        2, "Titulo", '<p:ph type="title"/>',
        TITULO_X, TITULO_Y, TITULO_W, TITULO_H, "Haga clic para modificar el estilo de titulo",
    )
    return (
        f'{_DECL}<p:sldLayout {_NS_ALL} type="titleOnly" preserve="1"><p:cSld name="Solo el titulo">'
        f"<p:spTree>{_GRUPO_RAIZ}{titulo}</p:spTree></p:cSld>"
        "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>"
    )


def _parte_tema() -> str:
    def color(nombre: str, hexa: str) -> str:
        return f'<a:{nombre}><a:srgbClr val="{hexa}"/></a:{nombre}>'

    colores = (
        color("dk1", COLOR_TEXTO) + color("lt1", COLOR_BLANCO) + color("dk2", COLOR_TEAL)
        + color("lt2", COLOR_GRIS_CLARO) + color("accent1", COLOR_TEAL) + color("accent2", COLOR_AMBAR)
        + color("accent3", "5B7C99") + color("accent4", "7A9E7E") + color("accent5", "8E6C8A")
        + color("accent6", "9AA5AB") + color("hlink", "0B5D66") + color("folHlink", "5B7C99")
    )
    relleno = '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    lineas = "".join(
        f'<a:ln w="{w}" cap="flat" cmpd="sng" algn="ctr">{relleno}<a:prstDash val="solid"/></a:ln>'
        for w in (6350, 12700, 19050)
    )
    efectos = "<a:effectStyle><a:effectLst/></a:effectStyle>" * 3
    return (
        f'{_DECL}<a:theme xmlns:a="{_NS_A}" name="TP5"><a:themeElements>'
        f'<a:clrScheme name="TP5">{colores}</a:clrScheme>'
        f'<a:fontScheme name="TP5"><a:majorFont><a:latin typeface="{FUENTE_TEXTO}"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>'
        f'<a:minorFont><a:latin typeface="{FUENTE_TEXTO}"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme>'
        f'<a:fmtScheme name="TP5"><a:fillStyleLst>{relleno * 3}</a:fillStyleLst>'
        f"<a:lnStyleLst>{lineas}</a:lnStyleLst><a:effectStyleLst>{efectos}</a:effectStyleLst>"
        f"<a:bgFillStyleLst>{relleno * 3}</a:bgFillStyleLst></a:fmtScheme>"
        "</a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/></a:theme>"
    )


def _parte_pres_props() -> str:
    return f"{_DECL}<p:presentationPr {_NS_ALL}/>"


def _parte_vista_props() -> str:
    return f'{_DECL}<p:viewPr {_NS_ALL}><p:gridSpacing cx="76200" cy="76200"/></p:viewPr>'


def _parte_estilos_tabla() -> str:
    return (
        f'{_DECL}<a:tblStyleLst xmlns:a="{_NS_A}" def="{{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}}"/>'
    )


def _rels_xml(lista: list[tuple[str, str, str]]) -> str:
    cuerpo = "".join(f'<Relationship Id="{i}" Type="{t}" Target="{d}"/>' for i, t, d in lista)
    return f'{_DECL}<Relationships xmlns="{_REL_PKG}">{cuerpo}</Relationships>'


# --------------------------------------------------------------------------
# Presentacion
# --------------------------------------------------------------------------


class Presentacion:
    """Un archivo .pptx en construccion."""

    def __init__(self, titulo: str) -> None:
        validar_texto(titulo, "titulo de la presentacion")
        self.titulo = titulo
        self._diapositivas: list[Diapositiva] = []
        self._medios: dict[str, tuple[str, bytes]] = {}  # sha256 -> (nombre, bytes)

    @property
    def n_diapositivas(self) -> int:
        return len(self._diapositivas)

    @property
    def n_imagenes(self) -> int:
        return sum(d._n_imagenes for d in self._diapositivas)

    def diapositiva(self, titulo: str, *, portada: bool = False) -> Diapositiva:
        d = Diapositiva(self, titulo, portada)
        self._diapositivas.append(d)
        return d

    def _registrar_medio(self, datos: bytes) -> str:
        clave = hashlib.sha256(datos).hexdigest()
        if clave not in self._medios:
            self._medios[clave] = (f"image{len(self._medios) + 1}.png", datos)
        return self._medios[clave][0]

    # -- paquete -------------------------------------------------------------

    def _partes(self) -> dict[str, bytes]:
        n = len(self._diapositivas)
        partes: dict[str, str | bytes] = {}

        overrides = [
            ("/ppt/presentation.xml", f"{_CT}.presentationml.presentation.main+xml"),
            ("/ppt/slideMasters/slideMaster1.xml", f"{_CT}.presentationml.slideMaster+xml"),
            ("/ppt/slideLayouts/slideLayout1.xml", f"{_CT}.presentationml.slideLayout+xml"),
            ("/ppt/slideLayouts/slideLayout2.xml", f"{_CT}.presentationml.slideLayout+xml"),
            ("/ppt/theme/theme1.xml", f"{_CT}.theme+xml"),
            ("/ppt/presProps.xml", f"{_CT}.presentationml.presProps+xml"),
            ("/ppt/viewProps.xml", f"{_CT}.presentationml.viewProps+xml"),
            ("/ppt/tableStyles.xml", f"{_CT}.presentationml.tableStyles+xml"),
            ("/docProps/core.xml", "application/vnd.openxmlformats-package.core-properties+xml"),
            ("/docProps/app.xml", f"{_CT}.extended-properties+xml"),
        ]
        for i in range(1, n + 1):
            overrides.append((f"/ppt/slides/slide{i}.xml", f"{_CT}.presentationml.slide+xml"))
        ct = "".join(f'<Override PartName="{p}" ContentType="{t}"/>' for p, t in overrides)
        partes["[Content_Types].xml"] = (
            f'{_DECL}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            f'<Default Extension="xml" ContentType="{_CT_XML}"/>'
            '<Default Extension="png" ContentType="image/png"/>'
            f"{ct}</Types>"
        )
        partes["_rels/.rels"] = _rels_xml(
            [
                ("rId1", f"{_REL}/officeDocument", "ppt/presentation.xml"),
                ("rId2", f"{_REL_PKG}/metadata/core-properties", "docProps/core.xml"),
                ("rId3", f"{_REL}/extended-properties", "docProps/app.xml"),
            ]
        )
        titulo = escape(self.titulo)
        partes["docProps/core.xml"] = (
            f'{_DECL}<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f"<dc:title>{titulo}</dc:title><dc:creator>TP5</dc:creator><cp:lastModifiedBy>TP5</cp:lastModifiedBy>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{_FECHA_ISO}</dcterms:created>'
            f'<dcterms:modified xsi:type="dcterms:W3CDTF">{_FECHA_ISO}</dcterms:modified></cp:coreProperties>'
        )
        partes["docProps/app.xml"] = (
            f'{_DECL}<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
            "<TotalTime>0</TotalTime><Application>pptx_minimo (TP5)</Application>"
            f"<PresentationFormat>Panoramica</PresentationFormat><Slides>{n}</Slides><Notes>0</Notes>"
            "<HiddenSlides>0</HiddenSlides></Properties>"
        )

        ids = "".join(f'<p:sldId id="{256 + i}" r:id="rId{2 + i}"/>' for i in range(n))
        lista_ids = f"<p:sldIdLst>{ids}</p:sldIdLst>" if n else ""
        partes["ppt/presentation.xml"] = (
            f'{_DECL}<p:presentation {_NS_ALL} saveSubsetFonts="1">'
            '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
            f'{lista_ids}<p:sldSz cx="{ANCHO_EMU}" cy="{ALTO_EMU}"/><p:notesSz cx="6858000" cy="9144000"/>'
            f"<p:defaultTextStyle>{_texto_por_defecto()}</p:defaultTextStyle></p:presentation>"
        )
        rels_pres = [("rId1", f"{_REL}/slideMaster", "slideMasters/slideMaster1.xml")]
        for i in range(n):
            rels_pres.append((f"rId{2 + i}", f"{_REL}/slide", f"slides/slide{i + 1}.xml"))
        base = 2 + n
        rels_pres += [
            (f"rId{base}", f"{_REL}/presProps", "presProps.xml"),
            (f"rId{base + 1}", f"{_REL}/viewProps", "viewProps.xml"),
            (f"rId{base + 2}", f"{_REL}/theme", "theme/theme1.xml"),
            (f"rId{base + 3}", f"{_REL}/tableStyles", "tableStyles.xml"),
        ]
        partes["ppt/_rels/presentation.xml.rels"] = _rels_xml(rels_pres)
        partes["ppt/presProps.xml"] = _parte_pres_props()
        partes["ppt/viewProps.xml"] = _parte_vista_props()
        partes["ppt/tableStyles.xml"] = _parte_estilos_tabla()
        partes["ppt/theme/theme1.xml"] = _parte_tema()
        partes["ppt/slideMasters/slideMaster1.xml"] = _parte_patron()
        partes["ppt/slideMasters/_rels/slideMaster1.xml.rels"] = _rels_xml(
            [
                ("rId1", f"{_REL}/slideLayout", "../slideLayouts/slideLayout1.xml"),
                ("rId2", f"{_REL}/slideLayout", "../slideLayouts/slideLayout2.xml"),
                ("rId3", f"{_REL}/theme", "../theme/theme1.xml"),
            ]
        )
        for k, xml in ((1, _parte_diseno_portada()), (2, _parte_diseno_titulo())):
            partes[f"ppt/slideLayouts/slideLayout{k}.xml"] = xml
            partes[f"ppt/slideLayouts/_rels/slideLayout{k}.xml.rels"] = _rels_xml(
                [("rId1", f"{_REL}/slideMaster", "../slideMasters/slideMaster1.xml")]
            )
        for i, d in enumerate(self._diapositivas, start=1):
            partes[f"ppt/slides/slide{i}.xml"] = d._xml()
            partes[f"ppt/slides/_rels/slide{i}.xml.rels"] = d._rels()
        for nombre, datos in self._medios.values():
            partes[f"ppt/media/{nombre}"] = datos
        return {k: (v.encode("utf-8") if isinstance(v, str) else v) for k, v in partes.items()}

    def guardar(self, ruta) -> None:
        """Escribe el paquete: miembros sin comprimir, [Content_Types].xml
        primero y el resto en orden alfabetico, fechas y atributos fijos, de
        modo que dos guardados de la misma presentacion son identicos byte a byte."""
        ruta = Path(ruta)
        partes = self._partes()
        orden = ["[Content_Types].xml"] + sorted(k for k in partes if k != "[Content_Types].xml")
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ruta, "w", compression=zipfile.ZIP_STORED) as z:
            for nombre in orden:
                info = zipfile.ZipInfo(nombre, date_time=_FECHA_FIJA)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o644 << 16
                z.writestr(info, partes[nombre])
