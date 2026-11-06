"""Verificador estructural de un .pptx, listado de titulos y volcado de texto.

Solo biblioteca estandar. El paquete se lee siempre en memoria: nunca se
extrae a disco (un ZIP hostil no puede escribir fuera de ningun directorio).

Uso:
    python3 -B presentacion/verificar.py --pptx RUTA [--listar] [--texto]
    python3 -B presentacion/verificar.py --pptx RUTA --salida RAIZ

Con --salida se verifica ademas la consistencia de la presentacion con las
salidas que regenero scripts/regenerar.sh en esa raiz (secciones, numeros,
imagenes por SHA-256 y texto alternativo, lineas citadas, filas de tablas,
afirmaciones retiradas). Las imagenes se comparan byte a byte: esa comprobacion
solo vale en la maquina que regenero los PNG (dependen de las tipografias del
sistema).
"""

from __future__ import annotations

import argparse
import hashlib
import posixpath
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

AQUI = Path(__file__).resolve().parent
if str(AQUI) not in sys.path:
    sys.path.insert(0, str(AQUI))

from lector import EJERCICIOS, FaltaEntrada, Salidas  # noqa: E402

LIMITE_BYTES = 100 * 1024 * 1024

_NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
_REL_BASE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_R_ID = "{%s}id" % _NS["r"]
_R_EMBED = "{%s}embed" % _NS["r"]


def _es_medio(nombre: str) -> bool:
    return nombre.startswith("ppt/media/")


def _ruta_insegura(nombre: str) -> bool:
    return (
        nombre.startswith("/")
        or "\\" in nombre
        or ".." in nombre.split("/")
        or re.match(r"^[A-Za-z]:", nombre) is not None
    )


def _resolver(base_dir: str, destino: str) -> str:
    if destino.startswith("/"):
        return posixpath.normpath(destino.lstrip("/"))
    return posixpath.normpath(posixpath.join(base_dir, destino))


def _dir_origen(nombre_rels: str) -> str:
    """Directorio de la parte a la que pertenece un archivo .rels."""
    if nombre_rels == "_rels/.rels":
        return ""
    carpeta, archivo = posixpath.split(nombre_rels)  # 'ppt/_rels', 'presentation.xml.rels'
    return posixpath.dirname(carpeta)


def verificar_estructura(ruta) -> list[str]:
    """Lista de errores del paquete; vacia cuando es valido."""
    errores: list[str] = []
    try:
        z = zipfile.ZipFile(ruta)
    except (zipfile.BadZipFile, OSError) as e:
        return [f"no se pudo abrir el paquete: {e}"]
    with z:
        infos = z.infolist()
        nombres = [i.filename for i in infos]
        conjunto = set(nombres)

        vistos = set()
        for n in nombres:
            if n in vistos:
                errores.append(f"miembro duplicado: {n}")
            vistos.add(n)
            if _ruta_insegura(n):
                errores.append(f"nombre de miembro inseguro (absoluto o con '..'): {n}")
        if sum(i.file_size for i in infos) >= LIMITE_BYTES:
            errores.append("el tamano total de los miembros supera 100 MB")
        for i in infos:
            if i.compress_type != zipfile.ZIP_STORED:
                errores.append(f"miembro comprimido (debe estar almacenado): {i.filename}")
        if not nombres or nombres[0] != "[Content_Types].xml":
            errores.append("[Content_Types].xml debe ser el primer miembro del paquete")

        # Todas las partes que no son medios deben ser XML bien formado.
        raices: dict[str, ET.Element] = {}
        for i in infos:
            n = i.filename
            if n.endswith("/") or _es_medio(n) or n in raices:
                continue
            try:
                raices[n] = ET.fromstring(z.read(n))
            except ET.ParseError as e:
                errores.append(f"parte mal formada ({n}): {e}")
            except (zipfile.BadZipFile, NotImplementedError, RuntimeError) as e:
                errores.append(f"parte ilegible ({n}): {e}")

        # Tipos de contenido.
        ct = raices.get("[Content_Types].xml")
        if ct is None:
            errores.append("falta [Content_Types].xml")
        else:
            defaults = {
                e.get("Extension", "").lower() for e in ct.findall("ct:Default", _NS)
            }
            overrides = {e.get("PartName", "") for e in ct.findall("ct:Override", _NS)}
            for n in conjunto:
                if n == "[Content_Types].xml" or n.endswith("/"):
                    continue
                ext = n.rsplit(".", 1)[-1].lower() if "." in n.rsplit("/", 1)[-1] else ""
                if "/" + n not in overrides and ext not in defaults:
                    errores.append(f"parte sin tipo de contenido: {n}")
            for o in overrides:
                if o.lstrip("/") not in conjunto:
                    errores.append(f"tipo de contenido para una parte inexistente: {o}")

        # Relaciones.
        for n, raiz in raices.items():
            if not n.endswith(".rels"):
                continue
            base = _dir_origen(n)
            for rel in raiz.findall("rel:Relationship", _NS):
                if rel.get("TargetMode") == "External":
                    continue
                destino = rel.get("Target", "")
                resuelto = _resolver(base, destino)
                if resuelto not in conjunto:
                    errores.append(f"relacion con destino inexistente en {n}: {destino} -> {resuelto}")

        # Lista de diapositivas.
        errores.extend(_verificar_diapositivas(raices, conjunto))
    return errores


def _verificar_diapositivas(raices: dict[str, ET.Element], conjunto: set[str]) -> list[str]:
    errores: list[str] = []
    paquete = raices.get("_rels/.rels")
    if paquete is None:
        return ["falta _rels/.rels"]
    pres_nombre = None
    for rel in paquete.findall("rel:Relationship", _NS):
        if rel.get("Type") == f"{_REL_BASE}/officeDocument":
            pres_nombre = _resolver("", rel.get("Target", ""))
    if pres_nombre is None or pres_nombre not in raices:
        return ["el paquete no declara una parte de presentacion legible"]
    pres = raices[pres_nombre]
    rels_nombre = posixpath.join(posixpath.dirname(pres_nombre), "_rels", posixpath.basename(pres_nombre) + ".rels")
    rels = raices.get(rels_nombre)
    if rels is None:
        return [f"falta {rels_nombre}"]
    base = posixpath.dirname(pres_nombre)
    por_id = {r.get("Id"): r for r in rels.findall("rel:Relationship", _NS)}
    partes_en_rels = set()
    for r in rels.findall("rel:Relationship", _NS):
        if r.get("Type") == f"{_REL_BASE}/slide":
            partes_en_rels.add(_resolver(base, r.get("Target", "")))
    en_lista = []
    ids_vistos = set()
    lista = pres.find("p:sldIdLst", _NS)
    for s in (lista.findall("p:sldId", _NS) if lista is not None else []):
        rid = s.get(_R_ID)
        sid = s.get("id")
        if sid in ids_vistos:
            errores.append(f"identificador de diapositiva repetido: {sid}")
        ids_vistos.add(sid)
        if sid is None or not sid.isdigit() or int(sid) < 256:
            errores.append(f"identificador de diapositiva invalido: {sid}")
        rel = por_id.get(rid)
        if rel is None or rel.get("Type") != f"{_REL_BASE}/slide":
            errores.append(f"la lista de diapositivas apunta a una relacion que no es de diapositiva: {rid}")
            continue
        en_lista.append(_resolver(base, rel.get("Target", "")))
    partes_slide = {n for n in conjunto if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)}
    if set(en_lista) != partes_slide or len(en_lista) != len(set(en_lista)):
        errores.append(
            "la lista de diapositivas no coincide con las partes de diapositiva: "
            f"lista={sorted(set(en_lista))} partes={sorted(partes_slide)}"
        )
    if partes_en_rels != partes_slide:
        errores.append("las relaciones de la presentacion no coinciden con las partes de diapositiva")
    return errores


def _diapositivas(ruta) -> list[tuple[int, ET.Element]]:
    """(numero desde 1, raiz XML) de cada diapositiva en el orden de la lista."""
    with zipfile.ZipFile(ruta) as z:
        paquete = ET.fromstring(z.read("_rels/.rels"))
        pres_nombre = next(
            _resolver("", r.get("Target", ""))
            for r in paquete.findall("rel:Relationship", _NS)
            if r.get("Type") == f"{_REL_BASE}/officeDocument"
        )
        pres = ET.fromstring(z.read(pres_nombre))
        rels_nombre = posixpath.join(
            posixpath.dirname(pres_nombre), "_rels", posixpath.basename(pres_nombre) + ".rels"
        )
        rels = ET.fromstring(z.read(rels_nombre))
        base = posixpath.dirname(pres_nombre)
        por_id = {r.get("Id"): r.get("Target", "") for r in rels.findall("rel:Relationship", _NS)}
        salida = []
        lista = pres.find("p:sldIdLst", _NS)
        for n, s in enumerate(lista.findall("p:sldId", _NS) if lista is not None else [], start=1):
            nombre = _resolver(base, por_id[s.get(_R_ID)])
            salida.append((n, ET.fromstring(z.read(nombre))))
        return salida


def _texto_de(elemento: ET.Element) -> str:
    return "".join((t.text or "") for t in elemento.iter(f"{{{_NS['a']}}}t"))


def titulos(ruta) -> list[str]:
    """Titulo (texto del marcador de titulo) de cada diapositiva, en orden."""
    salida = []
    for _n, raiz in _diapositivas(ruta):
        titulo = ""
        for sp in raiz.iter(f"{{{_NS['p']}}}sp"):
            ph = sp.find("p:nvSpPr/p:nvPr/p:ph", _NS)
            if ph is not None and ph.get("type") in ("title", "ctrTitle"):
                titulo = _texto_de(sp)
                break
        salida.append(titulo)
    return salida


def textos(ruta) -> list[tuple[int, str]]:
    """Cada tramo de texto de cada diapositiva como (numero de diapositiva, texto),
    en el orden del documento; incluye las celdas de las tablas."""
    salida = []
    for n, raiz in _diapositivas(ruta):
        for r in raiz.iter(f"{{{_NS['a']}}}r"):
            t = r.find("a:t", _NS)
            salida.append((n, (t.text or "") if t is not None else ""))
    return salida


# --------------------------------------------------------------------------
# Consistencia de la presentacion con las salidas regeneradas
# --------------------------------------------------------------------------

TITULO_PORTADA = "TP5: autoencoders (básico, denoising y variacional)"
PREFIJO_CIERRE = "Reproducibilidad"
PREFIJO_RESUMEN = "Resumen"

_ETIQUETAS_COMUNES = ("Arquitectura", "Técnicas", "Resultados", "Conclusiones")
SECCIONES = (
    ("Ej 1a", _ETIQUETAS_COMUNES),
    ("Ej 1b", _ETIQUETAS_COMUNES),
    ("Ej 2", ("Dataset",) + _ETIQUETAS_COMUNES),
)

# Unica constante de teoria que se admite sin fuente en las salidas: el 0.5 de las
# formulas de la arquitectura del VAE (sigma = exp(0.5 * logvar) y el factor de la KL).
CONSTANTES_DE_TEORIA = frozenset({"0.5"})

AFIRMACIONES_RETIRADAS = (
    "no es una copia de ninguno",
    "no es una copia",
    "no es copia",
    "no se trata de una copia",
    "está a al menos",
)

# Estudios de las tablas comparativas: (ejercicio, archivo, estudios).
TABLAS = (
    ("ej1a", "comparison.md", ("arch", "optim", "act", "loss")),
    ("ej1b", "dae_comparison.md", ("lat", "hid", "noise")),
)

_NUMERO = re.compile(r"(?<!\w)(\d+\.\d+|\d{3,})(?!\w)")
_ALT = re.compile(r"^output/(%s)/([^/\\]+)$" % "|".join(EJERCICIOS))
_SUFIJOS_CORPUS = (".txt", ".csv", ".md")


def numeros_a_verificar(texto: str) -> list[str]:
    """Numeros que se comprueban: decimales (cifras, punto, cifras) y enteros de tres
    o mas cifras, sin pegarse a una palabra, y sin signo. Los enteros de una y dos
    cifras son indices, cuentas de semillas o de diapositivas y no se comprueban."""
    return _NUMERO.findall(texto)


class Diapositiva:
    """Lo que la verificacion lee de una diapositiva."""

    def __init__(self, n: int, titulo: str) -> None:
        self.n = n
        self.titulo = titulo
        self.tramos: list[str] = []
        self.bloques: list[tuple[str, list[str]]] = []  # (fuente, parrafos)
        self.imagenes: list[tuple[str, bytes]] = []  # (texto alternativo, bytes)

    @property
    def texto(self) -> str:
        return "\n".join(self.tramos)


def _parrafos_de(sp: ET.Element) -> list[str]:
    return [_texto_de(p) for p in sp.iter(f"{{{_NS['a']}}}p")]


def _cargar_diapositivas(ruta) -> list[Diapositiva]:
    """Diapositivas en el orden de la lista, leidas en memoria (nunca se extrae a disco)."""
    salida: list[Diapositiva] = []
    with zipfile.ZipFile(ruta) as z:
        nombres = set(z.namelist())
        paquete = ET.fromstring(z.read("_rels/.rels"))
        pres_nombre = next(
            _resolver("", r.get("Target", ""))
            for r in paquete.findall("rel:Relationship", _NS)
            if r.get("Type") == f"{_REL_BASE}/officeDocument"
        )
        pres = ET.fromstring(z.read(pres_nombre))
        base = posixpath.dirname(pres_nombre)
        rels_nombre = posixpath.join(base, "_rels", posixpath.basename(pres_nombre) + ".rels")
        rels = ET.fromstring(z.read(rels_nombre))
        por_id = {r.get("Id"): r.get("Target", "") for r in rels.findall("rel:Relationship", _NS)}
        lista = pres.find("p:sldIdLst", _NS)
        for n, s in enumerate(lista.findall("p:sldId", _NS) if lista is not None else [], start=1):
            nombre = _resolver(base, por_id[s.get(_R_ID)])
            raiz = ET.fromstring(z.read(nombre))
            rels_slide = posixpath.join(
                posixpath.dirname(nombre), "_rels", posixpath.basename(nombre) + ".rels"
            )
            destinos = {}
            if rels_slide in nombres:
                for r in ET.fromstring(z.read(rels_slide)).findall("rel:Relationship", _NS):
                    destinos[r.get("Id")] = _resolver(posixpath.dirname(nombre), r.get("Target", ""))

            titulo = ""
            for sp in raiz.iter(f"{{{_NS['p']}}}sp"):
                ph = sp.find("p:nvSpPr/p:nvPr/p:ph", _NS)
                if ph is not None and ph.get("type") in ("title", "ctrTitle"):
                    titulo = _texto_de(sp)
                    break
            d = Diapositiva(n, titulo)
            for r in raiz.iter(f"{{{_NS['a']}}}r"):
                t = r.find("a:t", _NS)
                d.tramos.append((t.text or "") if t is not None else "")
            for sp in raiz.iter(f"{{{_NS['p']}}}sp"):
                c = sp.find("p:nvSpPr/p:cNvPr", _NS)
                forma = c.get("name", "") if c is not None else ""
                if forma.startswith("salida: "):
                    d.bloques.append((forma[len("salida: "):], _parrafos_de(sp)))
            for pic in raiz.iter(f"{{{_NS['p']}}}pic"):
                c = pic.find("p:nvPicPr/p:cNvPr", _NS)
                alt = c.get("descr", "") if c is not None else ""
                blip = pic.find("p:blipFill/a:blip", _NS)
                rid = blip.get(_R_EMBED) if blip is not None else None
                datos = b""
                if rid in destinos and destinos[rid] in nombres:
                    datos = z.read(destinos[rid])
                d.imagenes.append((alt, datos))
            salida.append(d)
    return salida


def _comprobar_secciones(diaps: list[Diapositiva]) -> list[str]:
    errores: list[str] = []
    if not diaps:
        return ["presentacion: no tiene diapositivas"]
    if diaps[0].titulo != TITULO_PORTADA:
        errores.append(
            f"diapositiva 1: la primera diapositiva debe ser la portada «{TITULO_PORTADA}» "
            f"y es «{diaps[0].titulo}»"
        )
    if len(diaps) < 2 or not diaps[1].titulo.startswith(PREFIJO_RESUMEN):
        titulo = diaps[1].titulo if len(diaps) > 1 else ""
        errores.append(
            f"diapositiva 2: la segunda diapositiva debe empezar con «{PREFIJO_RESUMEN}» y es «{titulo}»"
        )
    ultima = diaps[-1]
    if not ultima.titulo.startswith(PREFIJO_CIERRE):
        errores.append(
            f"diapositiva {ultima.n}: la ultima diapositiva debe empezar con «{PREFIJO_CIERRE}» "
            f"y es «{ultima.titulo}»"
        )
    grupo_anterior = -1
    etiquetas: list[set[str]] = [set() for _ in SECCIONES]
    for d in (diaps[2:-1] if len(diaps) >= 3 else []):
        grupo = next(
            (i for i, (ej, _tags) in enumerate(SECCIONES) if d.titulo.startswith(f"{ej}: ")), None
        )
        if grupo is None:
            errores.append(
                f"diapositiva {d.n}: el titulo «{d.titulo}» no empieza con «Ej 1a: », «Ej 1b: » ni «Ej 2: »"
            )
            continue
        if grupo < grupo_anterior:
            errores.append(
                f"diapositiva {d.n}: «{d.titulo}» esta fuera de orden (las secciones van Ej 1a, Ej 1b, Ej 2)"
            )
        grupo_anterior = max(grupo_anterior, grupo)
        resto = d.titulo[len(SECCIONES[grupo][0]) + 2:]
        for tag in SECCIONES[grupo][1]:
            if resto.startswith(tag):
                etiquetas[grupo].add(tag)
    for i, (ej, tags) in enumerate(SECCIONES):
        for tag in tags:
            if tag not in etiquetas[i]:
                errores.append(f"presentacion: falta una diapositiva «{ej}: {tag}...» en la seccion {ej}")
    return errores


def _corpus_de_numeros(raiz: Path, raiz_repo: Path) -> set[str]:
    corpus: set[str] = set()
    archivos: list[Path] = []
    for ej in EJERCICIOS:
        carpeta = raiz / ej
        if carpeta.is_dir():
            archivos += [
                p for p in sorted(carpeta.rglob("*")) if p.is_file() and p.suffix in _SUFIJOS_CORPUS
            ]
    docs = raiz_repo / "docs"
    if docs.is_dir():
        archivos += sorted(docs.glob("ej*-analisis.md"))
    for p in archivos:
        corpus.update(numeros_a_verificar(p.read_text(encoding="utf-8", errors="replace")))
    return corpus


def _comprobar_numeros(diaps, corpus: set[str]) -> tuple[list[str], int]:
    errores: list[str] = []
    verificados = 0
    for d in diaps:
        vistos: set[str] = set()
        for t in d.tramos:
            for tok in numeros_a_verificar(t):
                verificados += 1
                if tok in corpus or tok in CONSTANTES_DE_TEORIA or tok in vistos:
                    continue
                vistos.add(tok)
                errores.append(
                    f"diapositiva {d.n}: el numero {tok} no aparece en ninguna salida regenerada "
                    "ni en los documentos de analisis"
                )
    return errores, verificados


def _comprobar_imagenes(diaps, raiz: Path) -> tuple[list[str], int]:
    errores: list[str] = []
    verificadas = 0
    for d in diaps:
        for alt, datos in d.imagenes:
            m = _ALT.match(alt)
            if m is None:
                errores.append(
                    f"diapositiva {d.n}: el texto alternativo «{alt}» no tiene la forma "
                    "output/EJERCICIO/ARCHIVO con un ejercicio ej1a, ej1b o ej2"
                )
                continue
            archivo = raiz / m.group(1) / m.group(2)
            if not archivo.is_file():
                errores.append(
                    f"diapositiva {d.n}: la imagen «{alt}» no existe en {raiz} "
                    f"(se crea con: bash scripts/regenerar.sh {m.group(1)[2:]})"
                )
                continue
            if hashlib.sha256(datos).digest() != hashlib.sha256(archivo.read_bytes()).digest():
                errores.append(f"diapositiva {d.n}: la imagen «{alt}» difiere del PNG regenerado {archivo}")
                continue
            verificadas += 1
    return errores, verificadas


def _comprobar_citas(diaps, raiz: Path) -> tuple[list[str], int]:
    errores: list[str] = []
    citadas = 0
    transcripciones: dict[str, set[str] | None] = {}
    for d in diaps:
        for fuente, parrafos in d.bloques:
            if fuente not in transcripciones:
                partes = fuente.split("/")
                lineas = None
                if len(partes) == 2 and partes[0] in EJERCICIOS and partes[1] not in ("", ".", ".."):
                    ruta = raiz / partes[0] / partes[1]
                    if ruta.is_file():
                        lineas = set(ruta.read_text(encoding="utf-8").splitlines())
                transcripciones[fuente] = lineas
            lineas = transcripciones[fuente]
            if lineas is None:
                errores.append(
                    f"diapositiva {d.n}: el bloque citado «{fuente}» nombra una transcripcion "
                    f"que no existe en {raiz}"
                )
                continue
            for p in parrafos:
                if p in lineas:
                    citadas += 1
                else:
                    errores.append(
                        f"diapositiva {d.n}: la linea {p!r} del bloque «{fuente}» "
                        "no es una linea de la transcripcion"
                    )
    return errores, citadas


def _contiene_secuencia(tramos: list[str], fila: list[str]) -> bool:
    n = len(fila)
    return any(tramos[i : i + n] == fila for i in range(len(tramos) - n + 1))


def _comprobar_tablas(diaps, salidas: Salidas) -> tuple[list[str], int]:
    errores: list[str] = []
    filas_ok = 0
    for ej, archivo, estudios in TABLAS:
        for estudio in estudios:
            try:
                _enc, filas = salidas.tabla_md(ej, archivo, estudio)
            except (FaltaEntrada, ValueError) as e:
                errores.append(f"presentacion: no se pudo leer la tabla {estudio} de {ej}/{archivo}: {e}")
                continue
            for fila in filas:
                if any(_contiene_secuencia(d.tramos, fila) for d in diaps):
                    filas_ok += 1
                else:
                    errores.append(
                        f"presentacion: la fila {fila!r} de la tabla {estudio} de {ej}/{archivo} "
                        "no esta completa y en orden en ninguna diapositiva"
                    )
    return errores, filas_ok


def _comprobar_retiradas(diaps) -> list[str]:
    errores: list[str] = []
    for d in diaps:
        bajo = d.texto.lower()
        for frase in AFIRMACIONES_RETIRADAS:
            if frase in bajo:
                errores.append(f"diapositiva {d.n}: contiene la afirmacion retirada «{frase}»")
    return errores


def verificar_consistencia(ruta_pptx, raiz_salidas, raiz_repo) -> tuple[list[str], dict]:
    """Errores (cada uno nombra su diapositiva y el elemento) y contadores."""
    raiz = Path(raiz_salidas)
    repo = Path(raiz_repo)
    diaps = _cargar_diapositivas(ruta_pptx)
    salidas = Salidas(raiz)
    errores: list[str] = []
    errores += _comprobar_secciones(diaps)
    e, n_num = _comprobar_numeros(diaps, _corpus_de_numeros(raiz, repo))
    errores += e
    e, n_img = _comprobar_imagenes(diaps, raiz)
    errores += e
    e, n_lin = _comprobar_citas(diaps, raiz)
    errores += e
    e, n_fil = _comprobar_tablas(diaps, salidas)
    errores += e
    errores += _comprobar_retiradas(diaps)
    contadores = {
        "diapositivas": len(diaps),
        "imagenes": n_img,
        "numeros": n_num,
        "lineas": n_lin,
        "filas": n_fil,
    }
    return errores, contadores


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verifica la estructura de un .pptx")
    ap.add_argument("--pptx", required=True, help="ruta del archivo .pptx")
    ap.add_argument("--listar", action="store_true", help="imprime 'N: titulo' por diapositiva")
    ap.add_argument("--texto", action="store_true", help="imprime 'N: texto' por tramo de texto")
    ap.add_argument(
        "--salida",
        default=None,
        help="raiz de las salidas regeneradas: verifica ademas la consistencia de la presentacion con ellas",
    )
    args = ap.parse_args(argv)

    errores = verificar_estructura(args.pptx)
    if errores:
        for e in errores:
            print(e)
        print("ESTRUCTURA FALLA")
        return 1
    with zipfile.ZipFile(args.pptx) as z:
        partes = len(z.infolist())
    lista = titulos(args.pptx)
    print(f"ESTRUCTURA OK: {partes} partes, {len(lista)} diapositivas")
    if args.listar:
        for n, t in enumerate(lista, start=1):
            print(f"{n}: {t}")
    if args.texto:
        for n, t in textos(args.pptx):
            print(f"{n}: {t}")
    if args.salida is not None:
        try:
            errores, c = verificar_consistencia(args.pptx, args.salida, AQUI.parent)
        except FaltaEntrada as e:
            print(f"CONSISTENCIA FALLA: {e}")
            return 1
        for e in errores:
            print(f"- {e}")
        if errores:
            print(f"CONSISTENCIA FALLA: {len(errores)} problemas")
            return 1
        print(
            f"CONSISTENCIA OK: {c['diapositivas']} diapositivas, {c['imagenes']} imagenes verificadas, "
            f"{c['numeros']} numeros verificados, {c['lineas']} lineas citadas, {c['filas']} filas de tablas"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
