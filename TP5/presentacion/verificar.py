"""Verificador estructural de un .pptx, listado de titulos y volcado de texto.

Solo biblioteca estandar. El paquete se lee siempre en memoria: nunca se
extrae a disco (un ZIP hostil no puede escribir fuera de ningun directorio).

Uso:
    python3 -B presentacion/verificar.py --pptx RUTA [--listar] [--texto]
"""

from __future__ import annotations

import argparse
import posixpath
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verifica la estructura de un .pptx")
    ap.add_argument("--pptx", required=True, help="ruta del archivo .pptx")
    ap.add_argument("--listar", action="store_true", help="imprime 'N: titulo' por diapositiva")
    ap.add_argument("--texto", action="store_true", help="imprime 'N: texto' por tramo de texto")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
