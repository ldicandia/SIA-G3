"""Pruebas de la verificacion de consistencia (presentacion/verificar.py).

Sin dependencia de output/ ni de los documentos reales: los decks se construyen con
pptx_minimo en un directorio temporal, con una raiz de salidas y una raiz de repositorio
sinteticas. El deck sintetico sigue la tabla CONSIGNA real del verificador: una diapositiva
por entrada, con sus imagenes y las tablas de sus estudios.
"""

import re
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pptx_minimo as pm  # noqa: E402
import verificar as v  # noqa: E402

TRANSCRIPCION = """== Ej 2: ejemplo ==
configuracion: encoder 4096-512-4 | decoder 2-512-4096
    linea con sangria 12.3456
RESULTADO EJ2: OK (11/11 comprobaciones) 987
"""

TABLA_1A = """## Estudio: {e}

| variante | semillas | epocas | mediana |
|---|---|---|---|
| {e}-a | 3/3 | 275 | 0.1234 |
| {e}-b | 2/3 | 300 | 0.5678 |
"""

TABLA_1B = """## Estudio: {e}

| variante | semillas | puntaje |
|---|---|---|
| {e}-b | 2/3 | 1.2345 |
"""

DOC = "Documento de analisis. El valor 0.4321 se verifica contra el codigo.\n"

NOTAS = (
    "Estas son las notas del orador de la diapositiva de prueba y explican con sus propias palabras "
    "lo que la diapositiva muestra, como se lee la figura y cuales son las preguntas probables."
)

# Titulo concreto de cada entrada de la consigna (su expresion regular sin las clases).
TITULOS = {r"Ej 1a: Resultados: \d+/\d+ patrones con ≤ \d+ píxel": "Ej 1a: Resultados: 32/32 patrones con ≤ 1 píxel"}


def png(ancho: int, alto: int, gris: int) -> bytes:
    def trozo(tipo: bytes, datos: bytes) -> bytes:
        crc = zlib.crc32(tipo + datos) & 0xFFFFFFFF
        return struct.pack(">I", len(datos)) + tipo + datos + struct.pack(">I", crc)

    fila = b"\x00" + bytes([gris]) * ancho
    crudo = zlib.compress(fila * alto)
    return (
        b"\x89PNG\r\n\x1a\n"
        + trozo(b"IHDR", struct.pack(">IIBBBBB", ancho, alto, 8, 0, 0, 0, 0))
        + trozo(b"IDAT", crudo)
        + trozo(b"IEND", b"")
    )


def archivo_de(patron: str) -> str:
    """Nombre de archivo concreto para una expresion regular de imagen de la consigna."""
    return patron.replace("\\d", "0").replace("\\.", ".")


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.salidas = self.dir / "salidas"
        self.repo = self.dir / "repo"
        for ej in ("ej1a", "ej1b", "ej2"):
            (self.salidas / ej).mkdir(parents=True)
        (self.salidas / "ej2" / "salida-ej2.txt").write_text(TRANSCRIPCION, encoding="utf-8")
        (self.salidas / "ej1a" / "comparison.md").write_text(
            "\n".join(TABLA_1A.format(e=e) for e in ("arch", "optim", "act", "loss")), encoding="utf-8"
        )
        (self.salidas / "ej1b" / "dae_comparison.md").write_text(
            "\n".join(TABLA_1B.format(e=e) for e in ("lat", "hid", "noise")), encoding="utf-8"
        )
        # Una imagen apaisada por cada figura de la consigna; las de a pares, cuadradas.
        for _item, _titulo, imagenes, _estudios in v.CONSIGNA:
            for k, patron in enumerate(imagenes):
                alt = archivo_de(patron)
                lado = (100, 100) if len(imagenes) > 1 else (230, 100)
                (self.salidas / alt.split("/")[1] / alt.split("/")[2]).write_bytes(png(*lado, 40 + 7 * k + len(alt)))
        (self.dir / "otro.png").write_bytes(png(230, 100, 200))
        (self.repo / "docs").mkdir(parents=True)
        for ej in ("ej1a", "ej1b", "ej2"):
            (self.repo / "docs" / f"{ej}-analisis.md").write_text(DOC, encoding="utf-8")

    def _tabla(self, d, estudio: str, y: float, columnas=None, recorte=None) -> None:
        ej, archivo = ("ej1a", "comparison.md") if estudio in ("arch", "optim", "act", "loss") else ("ej1b", "dae_comparison.md")
        enc, filas = v.Salidas(self.salidas).tabla_md(ej, archivo, estudio)
        idx = columnas or list(range(len(enc)))
        cuerpo = [[f[i] for i in idx] for f in filas]
        if recorte:
            cuerpo = [[recorte(c) for c in f] for f in cuerpo]
        d.tabla(0.5, y, 12, [[enc[i] for i in idx]] + cuerpo)

    def deck(
        self,
        *,
        texto_extra=None,
        notas_extra=None,
        citas=None,
        alt_latente=None,
        imagen_latente=None,
        caja_latente=(0.5, 1.65, 12.333, 5.35),
        lineas_bloque=None,
        fuente_bloque="ej2/salida-ej2.txt",
        omitir=None,
        sin_estudio=None,
        sin_imagen=None,
        columnas_arch=None,
        recorte_arch=None,
        sin_notas=None,
        notas_cortas=None,
        titulo_largo=False,
        tercera_imagen=False,
        primero=None,
        cierre="Reproducibilidad: del código a la presentación",
        orden=("Ej 1a", "Ej 1b", "Ej 2"),
    ) -> Path:
        pres = pm.Presentacion(v.TITULO_PORTADA)
        d = pres.diapositiva(primero or v.TITULO_PORTADA, portada=True)
        d.notas("Portada.")
        d = pres.diapositiva("Resumen: tres ejercicios")
        d.notas(NOTAS)
        hechas = set()
        for ej in orden:
            for item, patron, imagenes, estudios in v.CONSIGNA:
                titulo = TITULOS.get(patron, patron)
                if not titulo.startswith(f"{ej}: ") or omitir == titulo:
                    continue
                hechas.add(titulo)
                if titulo_largo and titulo == "Ej 2: Dataset":
                    d = pres.diapositiva("Ej 2: Dataset " + "x" * 47)
                else:
                    d = pres.diapositiva(titulo)
                y = 1.65
                for estudio in estudios:
                    if estudio == sin_estudio:
                        continue
                    if estudio == "arch":
                        self._tabla(d, estudio, y, columnas_arch, recorte_arch)
                    else:
                        self._tabla(d, estudio, y)
                    y += 1.6
                es_latente = titulo == "Ej 1a: Resultados: espacio latente"
                for k, img in enumerate(imagenes):
                    alt = archivo_de(img)
                    if alt == sin_imagen:
                        continue
                    ruta = self.salidas / alt.split("/")[1] / alt.split("/")[2]
                    if len(imagenes) > 1:
                        caja = (0.5 + k * 6.2, 1.65, 6.0, 5.35)
                    elif es_latente:
                        caja = caja_latente
                        ruta = imagen_latente or ruta
                        alt = alt_latente or alt
                    else:
                        caja = (0.5, 1.65, 12.333, 5.35)
                    d.imagen(ruta, *caja, alt=alt)
                if es_latente and tercera_imagen:
                    for k in range(2):
                        d.imagen(self.salidas / "ej1a" / "latent.png", 0.5, 1.65, 12.333, 5.35, alt="output/ej1a/latent.png")
                if titulo == "Ej 2: Arquitectura":
                    lineas = lineas_bloque or [
                        "RESULTADO EJ2: OK (11/11 comprobaciones) 987",
                        "    linea con sangria 12.3456",
                    ]
                    d.codigo(0.5, 1.7, 12, 1.2, lineas, fuente=fuente_bloque)
                    if texto_extra:
                        d.texto(0.5, 3.2, 12, 3.5, texto_extra)
                if sin_notas == titulo:
                    continue
                if notas_cortas == titulo:
                    d.notas(" ".join(["palabra"] * 29))
                elif titulo == "Ej 2: Arquitectura":
                    d.notas([NOTAS] + ([notas_extra] if notas_extra else []), citas=citas)
                else:
                    d.notas(NOTAS)
            for tag in dict(v.SECCIONES)[ej]:
                if not any(t.startswith(f"{ej}: {tag}") for t in hechas) and omitir != f"{ej}: {tag}":
                    d = pres.diapositiva(f"{ej}: {tag}")
                    d.notas(NOTAS)
        d = pres.diapositiva(cierre)
        d.notas(NOTAS)
        ruta = self.dir / "deck.pptx"
        pres.guardar(ruta)
        return ruta

    def verificar(self, ruta):
        return v.verificar_consistencia(ruta, self.salidas, self.repo)

    def assertProblema(self, errores, *fragmentos):
        for e in errores:
            if all(f in e for f in fragmentos):
                return
        self.fail(f"ningun problema contiene {fragmentos}; problemas: {errores}")


class PruebasNumeros(unittest.TestCase):
    def test_numeric_token_rule(self):
        texto = "dae_examples_p0.10.png 12 123 4.5 -0.25 1234abc x2 7 1e5 12345"
        self.assertEqual(v.numeros_a_verificar(texto), ["123", "4.5", "0.25", "12345"])

    def test_word_count_ignores_symbols(self):
        self.assertEqual(v.contar_palabras(["→", "uno dos → 3", "≤ 1 píxel (BCE) + KL"]), 7)

    def test_every_assignment_item_is_in_the_checklist(self):
        items = {item for item, *_ in v.CONSIGNA}
        self.assertEqual(items, {"1.a.1", "1.a.2", "1.a.3", "1.a.4", "1.b.1", "1.b.2", "2.a", "2.b", "2.c"})
        for _item, patron, _imagenes, _estudios in v.CONSIGNA:
            self.assertLessEqual(len(TITULOS.get(patron, patron)), v.MAX_TITULO)


class PruebasVerificar(Base):
    def test_consistent_deck_passes(self):
        errores, c = self.verificar(self.deck())
        self.assertEqual(errores, [])
        self.assertEqual(c["imagenes"], sum(len(i) for _a, _b, i, _d in v.CONSIGNA))
        self.assertEqual(c["lineas"], 2)
        self.assertEqual(c["filas"], 9)
        self.assertGreater(c["numeros"], 5)
        self.assertEqual(c["notas"], c["diapositivas"])
        self.assertLessEqual(c["palabras"], v.MAX_PALABRAS)

    def test_number_not_in_outputs_is_reported(self):
        errores, _ = self.verificar(self.deck(texto_extra="El valor 7.7777 y el entero 4567 no existen."))
        self.assertProblema(errores, "7.7777", "diapositiva")
        self.assertProblema(errores, "4567")

    def test_number_in_notes_is_reported_like_one_on_a_slide(self):
        errores, _ = self.verificar(self.deck(notas_extra="En las notas, el valor 8.8888 tampoco existe."))
        self.assertProblema(errores, "8.8888", "(notas)")
        errores, _ = self.verificar(self.deck(notas_extra="En las notas, 12.3456 y 0.4321 si existen."))
        self.assertEqual(errores, [])

    def test_theory_constant_is_allowed(self):
        ok = self.deck(texto_extra="sigma = exp(0.5 * logvar), indice 7 y 99, archivo dae_examples_p0.10.png")
        errores, _ = self.verificar(ok)
        self.assertEqual(errores, [])
        errores, _ = self.verificar(self.deck(texto_extra="factor 0.55"))
        self.assertProblema(errores, "0.55")

    def test_number_from_documents_and_outputs_is_accepted(self):
        errores, _ = self.verificar(self.deck(texto_extra="doc 0.4321, salida 12.3456 y 987"))
        self.assertEqual(errores, [])

    def test_image_hash_mismatch_is_reported(self):
        errores, _ = self.verificar(self.deck(imagen_latente=self.dir / "otro.png"))
        self.assertProblema(errores, "output/ej1a/latent.png", "difiere")

    def test_image_alt_must_name_a_regenerated_file(self):
        errores, _ = self.verificar(self.deck(alt_latente="latent.png"))
        self.assertProblema(errores, "texto alternativo", "latent.png")
        errores, _ = self.verificar(self.deck(alt_latente="output/ej1a/no-existe.png"))
        self.assertProblema(errores, "no existe", "no-existe.png")
        errores, _ = self.verificar(self.deck(alt_latente="output/otro/latent.png"))
        self.assertProblema(errores, "texto alternativo")

    def test_small_picture_is_reported(self):
        # 89 % del ancho y, por la proporcion, mucho menos del 90 % del alto.
        errores, _ = self.verificar(self.deck(caja_latente=(0.5, 1.65, 0.89 * 12.333, 5.35)))
        self.assertProblema(errores, "output/ej1a/latent.png", "90%")
        errores, _ = self.verificar(self.deck(caja_latente=(0.5, 1.65, 0.91 * 12.333, 5.35)))
        self.assertEqual(errores, [])

    def test_third_picture_is_reported(self):
        errores, _ = self.verificar(self.deck(tercera_imagen=True))
        self.assertProblema(errores, "3 imagenes", "maximo es 2")

    def test_quoted_line_not_in_transcript_is_reported(self):
        errores, _ = self.verificar(
            self.deck(lineas_bloque=["RESULTADO EJ2: OK (11/11 comprobaciones) 987", "linea inventada"])
        )
        self.assertProblema(errores, "linea inventada", "ej2/salida-ej2.txt")
        errores, _ = self.verificar(self.deck(fuente_bloque="ej2/falta.txt"))
        self.assertProblema(errores, "ej2/falta.txt", "no existe")
        errores, _ = self.verificar(self.deck(fuente_bloque="../repo/docs/ej2-analisis.md"))
        self.assertProblema(errores, "no existe")

    def test_leading_spaces_of_quoted_lines_matter(self):
        errores, _ = self.verificar(self.deck(lineas_bloque=["linea con sangria 12.3456"]))
        self.assertProblema(errores, "linea con sangria 12.3456", "no es una linea")

    def test_notes_citation_must_be_a_line_of_its_file(self):
        buena = [("ej2/salida-ej2.txt", ["    linea con sangria 12.3456", "RESULTADO EJ2: OK (11/11 comprobaciones) 987"])]
        errores, c = self.verificar(self.deck(citas=buena))
        self.assertEqual(errores, [])
        self.assertEqual(c["lineas"], 4)
        errores, _ = self.verificar(self.deck(citas=[("ej2/salida-ej2.txt", ["linea con sangria 12.3456"])]))
        self.assertProblema(errores, "linea con sangria 12.3456", "notas", "no es una linea")
        errores, _ = self.verificar(self.deck(citas=[("ej2/falta.txt", ["x"])]))
        self.assertProblema(errores, "ej2/falta.txt", "notas", "no existe")

    def test_table_row_must_be_an_in_order_subsequence_of_its_source(self):
        errores, c = self.verificar(self.deck(columnas_arch=[0, 1, 3]))
        self.assertEqual(errores, [])
        self.assertEqual(c["filas"], 9)
        errores, _ = self.verificar(self.deck(columnas_arch=[0, 3, 1]))
        self.assertProblema(errores, "arch-a", "subsecuencia")
        errores, _ = self.verificar(self.deck(recorte_arch=lambda c: c.replace("0.1234", "0.123")))
        self.assertProblema(errores, "arch-a", "subsecuencia")

    def test_study_needs_all_its_variants_or_its_curves_figure(self):
        # arch: sin tabla y sin figura de curvas en el deck.
        errores, _ = self.verificar(self.deck(sin_estudio="arch"))
        self.assertProblema(errores, "estudio arch", "arch-a, arch-b")
        self.assertProblema(errores, "consigna 1.a.2", "arch")
        # optim: ninguna variante en tablas, pero su figura de curvas esta incrustada.
        errores, _ = self.verificar(self.deck())
        self.assertFalse([e for e in errores if "optim" in e])
        errores, _ = self.verificar(self.deck(sin_imagen="output/ej1a/curves_optim_loss.png"))
        self.assertProblema(errores, "estudio optim", "curves_optim_loss.png")
        # Una variante nueva en la fuente que la diapositiva no muestra.
        ruta = self.deck()
        md = self.salidas / "ej1b" / "dae_comparison.md"
        md.write_text(
            md.read_text(encoding="utf-8").replace("| lat-b | 2/3 | 1.2345 |", "| lat-b | 2/3 | 1.2345 |\n| lat-c | 1/3 | 1.2345 |"),
            encoding="utf-8",
        )
        errores, _ = self.verificar(ruta)
        self.assertProblema(errores, "estudio lat", "lat-c")

    def test_each_retracted_phrase_is_reported(self):
        frases = {
            "no es una copia de ninguno": "El bloque No es una copia de ninguno del dataset.",
            "no es una copia": "Se dice que NO ES UNA COPIA del original.",
            "no es copia": "Entonces no es copia de nada.",
            "no se trata de una copia": "Aqui no se trata de una copia.",
            "está a al menos": "El bloque Está a al menos tantos pixeles.",
        }
        self.assertEqual(set(frases), set(v.AFIRMACIONES_RETIRADAS))
        for frase, texto in frases.items():
            with self.subTest(frase=frase):
                errores, _ = self.verificar(self.deck(texto_extra=texto))
                self.assertProblema(errores, "afirmacion retirada", frase)
                errores, _ = self.verificar(self.deck(notas_extra=texto))
                self.assertProblema(errores, "(notas)", "afirmacion retirada", frase)

    def test_section_order_and_bookends_are_checked(self):
        errores, _ = self.verificar(self.deck(omitir="Ej 1b: Conclusiones"))
        self.assertProblema(errores, "Ej 1b: Conclusiones")
        errores, _ = self.verificar(self.deck(omitir="Ej 2: Dataset"))
        self.assertProblema(errores, "Ej 2: Dataset")
        errores, _ = self.verificar(self.deck(orden=("Ej 1b", "Ej 1a", "Ej 2")))
        self.assertProblema(errores, "fuera de orden")
        errores, _ = self.verificar(self.deck(primero="Otro titulo"))
        self.assertProblema(errores, "diapositiva 1", "portada")
        errores, _ = self.verificar(self.deck(cierre="Ej 2: Cierre"))
        self.assertProblema(errores, "ultima diapositiva", "Reproducibilidad")

    def test_word_budget(self):
        errores, c = self.verificar(self.deck(texto_extra=" ".join(["palabra"] * 30)))
        self.assertEqual(errores, [])
        self.assertEqual(c["palabras"], 40)
        errores, _ = self.verificar(self.deck(texto_extra=" ".join(["palabra"] * 31)))
        self.assertProblema(errores, "Ej 2: Arquitectura", "41 palabras", "maximo es 40")

    def test_long_title_is_reported(self):
        errores, _ = self.verificar(self.deck(titulo_largo=True))
        self.assertProblema(errores, "61 caracteres", "maximo")

    def test_slide_without_notes_or_with_short_notes_is_reported(self):
        errores, _ = self.verificar(self.deck(sin_notas="Ej 1b: Arquitectura"))
        self.assertProblema(errores, "Ej 1b: Arquitectura", "no tiene notas")
        errores, _ = self.verificar(self.deck(notas_cortas="Ej 1b: Arquitectura"))
        self.assertProblema(errores, "Ej 1b: Arquitectura", "29 palabras", "minimo es 30")

    def test_missing_assignment_item_is_reported_by_name(self):
        errores, _ = self.verificar(self.deck(omitir="Ej 1a: Resultados: espacio latente"))
        self.assertProblema(errores, "consigna 1.a.3", "falta la diapositiva")
        errores, _ = self.verificar(self.deck(sin_imagen="output/ej1a/latent.png"))
        self.assertProblema(errores, "consigna 1.a.3", "no incrusta", "latent")
        errores, _ = self.verificar(self.deck(sin_imagen="output/ej2/vae_new_ore.png"))
        self.assertProblema(errores, "consigna 2.c", "vae_new_ore")
        errores, _ = self.verificar(self.deck(sin_estudio="noise"))
        self.assertProblema(errores, "consigna 1.b.1", "noise-b", "noise")

    def test_small_fonts_in_a_foreign_deck_are_reported(self):
        ruta = self.deck()
        import zipfile

        partes = []
        with zipfile.ZipFile(ruta) as z:
            for info in z.infolist():
                datos = z.read(info.filename)
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", info.filename):
                    datos = datos.replace(b'sz="1800"', b'sz="1200"').replace(b'sz="1600"', b'sz="1100"')
                partes.append((info.filename, datos))
        otra = self.dir / "chica.pptx"
        with zipfile.ZipFile(otra, "w", compression=zipfile.ZIP_STORED) as z:
            for nombre, datos in partes:
                z.writestr(nombre, datos)
        errores, _ = self.verificar(otra)
        self.assertProblema(errores, "12 pt", "minimo es 18")
        self.assertProblema(errores, "11 pt", "minimo es 16")


if __name__ == "__main__":
    unittest.main()
