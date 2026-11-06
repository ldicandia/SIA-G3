"""Pruebas de la verificacion de consistencia (presentacion/verificar.py).

Sin dependencia de output/ ni de los documentos reales: los decks se construyen con
pptx_minimo en un directorio temporal, con una raiz de salidas y una raiz de repositorio
sinteticas.
"""

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

| variante | semillas | mediana |
|---|---|---|
| {e}-a | 3/3 | 0.1234 |
"""

TABLA_1B = """## Estudio: {e}

| variante | semillas | puntaje |
|---|---|---|
| {e}-b | 2/3 | 1.2345 |
"""

DOC = "Documento de analisis. El valor 0.4321 se verifica contra el codigo.\n"


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
        (self.salidas / "ej1a" / "g.png").write_bytes(png(40, 20, 100))
        (self.dir / "otro.png").write_bytes(png(40, 20, 200))
        (self.repo / "docs").mkdir(parents=True)
        for ej in ("ej1a", "ej1b", "ej2"):
            (self.repo / "docs" / f"{ej}-analisis.md").write_text(DOC, encoding="utf-8")

    def deck(
        self,
        *,
        texto_extra=None,
        alt="output/ej1a/g.png",
        imagen=None,
        lineas_bloque=None,
        fuente_bloque="ej2/salida-ej2.txt",
        omitir=None,
        orden=("Ej 1a", "Ej 1b", "Ej 2"),
        primero=None,
        cierre="Reproducibilidad: de un checkout limpio a la presentación",
    ) -> Path:
        pres = pm.Presentacion(v.TITULO_PORTADA)
        pres.diapositiva(primero or v.TITULO_PORTADA, portada=True)
        pres.diapositiva("Resumen: tres ejercicios")
        etiquetas = dict(v.SECCIONES)
        for ej in orden:
            for tag in etiquetas[ej]:
                if omitir == (ej, tag):
                    continue
                d = pres.diapositiva(f"{ej}: {tag}")
                if (ej, tag) == ("Ej 1a", "Resultados"):
                    d.imagen(imagen or self.salidas / "ej1a" / "g.png", 7, 1.7, 5, 3, alt=alt)
                    d.tabla(0.5, 1.7, 6, [["variante", "semillas", "mediana"], ["arch-a", "3/3", "0.1234"]])
                    d.tabla(0.5, 3.2, 6, [["v", "s", "m"], ["optim-a", "3/3", "0.1234"]])
                    d.tabla(0.5, 4.2, 6, [["v", "s", "m"], ["act-a", "3/3", "0.1234"]])
                    d.tabla(0.5, 5.2, 6, [["v", "s", "m"], ["loss-a", "3/3", "0.1234"]])
                    if texto_extra:
                        d.texto(7, 5.0, 5.5, 1.5, texto_extra)
                if (ej, tag) == ("Ej 1b", "Resultados"):
                    d.tabla(0.5, 1.7, 6, [["v", "s", "p"], ["lat-b", "2/3", "1.2345"]])
                    d.tabla(0.5, 3.2, 6, [["v", "s", "p"], ["hid-b", "2/3", "1.2345"]])
                    d.tabla(0.5, 4.2, 6, [["v", "s", "p"], ["noise-b", "2/3", "1.2345"]])
                if (ej, tag) == ("Ej 2", "Resultados"):
                    lineas = lineas_bloque or [
                        "RESULTADO EJ2: OK (11/11 comprobaciones) 987",
                        "    linea con sangria 12.3456",
                    ]
                    d.codigo(0.5, 1.7, 9, 1.2, lineas, fuente=fuente_bloque)
        pres.diapositiva(cierre)
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


class PruebasVerificar(Base):
    def test_consistent_deck_passes(self):
        errores, c = self.verificar(self.deck())
        self.assertEqual(errores, [])
        self.assertEqual(c["imagenes"], 1)
        self.assertEqual(c["lineas"], 2)
        self.assertEqual(c["filas"], 7)
        self.assertGreater(c["numeros"], 5)
        self.assertEqual(c["diapositivas"], 2 + 4 + 4 + 5 + 1)

    def test_number_not_in_outputs_is_reported(self):
        errores, _ = self.verificar(self.deck(texto_extra="El valor 7.7777 y el entero 4567 no existen."))
        self.assertProblema(errores, "7.7777", "diapositiva")
        self.assertProblema(errores, "4567")

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
        errores, _ = self.verificar(self.deck(imagen=self.dir / "otro.png"))
        self.assertProblema(errores, "output/ej1a/g.png", "difiere")

    def test_image_alt_must_name_a_regenerated_file(self):
        errores, _ = self.verificar(self.deck(alt="g.png"))
        self.assertProblema(errores, "texto alternativo", "g.png")
        errores, _ = self.verificar(self.deck(alt="output/ej1a/no-existe.png"))
        self.assertProblema(errores, "no existe", "no-existe.png")
        errores, _ = self.verificar(self.deck(alt="output/otro/g.png"))
        self.assertProblema(errores, "texto alternativo")

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

    def test_missing_table_row_is_reported(self):
        ruta = self.deck()
        md = self.salidas / "ej1b" / "dae_comparison.md"
        md.write_text(md.read_text(encoding="utf-8").replace("| lat-b | 2/3 | 1.2345 |", "| lat-b | 2/3 | 1.2345 |\n| lat-c | 1/3 | 9.8765 |"), encoding="utf-8")
        errores, _ = self.verificar(ruta)
        self.assertProblema(errores, "lat-c", "lat")

    def test_shortened_data_cell_is_reported(self):
        ruta = self.deck()
        md = self.salidas / "ej1a" / "comparison.md"
        md.write_text(md.read_text(encoding="utf-8").replace("0.1234", "0.12345"), encoding="utf-8")
        errores, _ = self.verificar(ruta)
        self.assertProblema(errores, "arch-a")

    def test_each_retracted_phrase_is_reported(self):
        frases = {
            "no es una copia de ninguno": "El emoji No es una copia de ninguno del dataset.",
            "no es una copia": "Se dice que NO ES UNA COPIA del original.",
            "no es copia": "Entonces no es copia de nada.",
            "no se trata de una copia": "Aqui no se trata de una copia.",
            "está a al menos": "El emoji Está a al menos tantos pixeles.",
        }
        self.assertEqual(set(frases), set(v.AFIRMACIONES_RETIRADAS))
        for frase, texto in frases.items():
            with self.subTest(frase=frase):
                errores, _ = self.verificar(self.deck(texto_extra=texto))
                self.assertProblema(errores, "afirmacion retirada", frase)

    def test_section_order_and_bookends_are_checked(self):
        errores, _ = self.verificar(self.deck(omitir=("Ej 1b", "Resultados")))
        self.assertProblema(errores, "Ej 1b: Resultados")
        errores, _ = self.verificar(self.deck(omitir=("Ej 2", "Dataset")))
        self.assertProblema(errores, "Ej 2: Dataset")
        errores, _ = self.verificar(self.deck(orden=("Ej 1b", "Ej 1a", "Ej 2")))
        self.assertProblema(errores, "fuera de orden")
        errores, _ = self.verificar(self.deck(primero="Otro titulo"))
        self.assertProblema(errores, "diapositiva 1", "portada")
        errores, _ = self.verificar(self.deck(cierre="Ej 2: Cierre"))
        self.assertProblema(errores, "ultima diapositiva", "Reproducibilidad")


if __name__ == "__main__":
    unittest.main()
