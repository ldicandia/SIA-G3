"""Pruebas del lector de salidas. Textos de muestra en linea y un directorio temporal."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lector  # noqa: E402
from lector import FaltaEntrada, Salidas, campo, numero  # noqa: E402

REPORTE = """== Ej 1a: ejemplo ==
configuracion elegida: arch-h64x32-s0 | arquitectura: 35-64-32-2-32-64-35 | epocas: 10000
idx  caracter  pixeles_incorrectos
  0  `       0
    sangrado con espacios
RESULTADO: 32/32 patrones con a lo sumo 1 pixel incorrecto (maximo 0)
final
"""

COMPARACION = """## Estudio: arch

| variante | semillas | mediana |
|---|---|---|
| arch-h24 | 3/3 | 1100 |
| arch-h32x16 | 2/3 | 575 |

## Estudio: optim

| variante | semillas |
|:---|---:|
| optim-sgd | 0/3 |
"""


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.raiz = Path(self._tmp.name)
        (self.raiz / "ej1a").mkdir()
        (self.raiz / "ej1a" / "salida-ej1a.txt").write_text(REPORTE, encoding="utf-8")
        (self.raiz / "ej1a" / "comparison.md").write_text(COMPARACION, encoding="utf-8")
        (self.raiz / "ej1a" / "r.csv").write_text("nivel,valor\n0.10,3\n0.20,4\n", encoding="utf-8")
        (self.raiz / "ej1a" / "g.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 20)
        (self.raiz / "ej1a" / "falso.png").write_bytes(b"no es png")
        self.s = Salidas(self.raiz)


class PruebasLector(Base):
    def test_missing_file_names_the_command(self):
        with self.assertRaises(FaltaEntrada) as ctx:
            self.s.ruta("ej1b", "salida-ej1b.txt")
        msg = str(ctx.exception)
        self.assertIn("salida-ej1b.txt", msg)
        self.assertIn("bash scripts/regenerar.sh 1b", msg)
        with self.assertRaises(FaltaEntrada) as ctx:
            Salidas(self.raiz / "no-existe").lineas("ej2", "salida-ej2.txt")
        self.assertIn("bash scripts/regenerar.sh 2", str(ctx.exception))

    def test_unknown_exercise_is_a_value_error(self):
        with self.assertRaises(ValueError):
            self.s.ruta("ej3", "x")
        with self.assertRaises(ValueError):
            self.s.alt("ej9", "x.png")

    def test_lineas_keep_leading_spaces(self):
        lineas = self.s.lineas("ej1a", "salida-ej1a.txt")
        self.assertIn("    sangrado con espacios", lineas)
        self.assertIn("  0  `       0", lineas)
        self.assertTrue(all(not l.endswith("\n") for l in lineas))
        self.assertEqual(lineas[-1], "final")

    def test_linea_requires_exactly_one_match(self):
        self.assertTrue(self.s.linea("ej1a", "salida-ej1a.txt", "RESULTADO:").startswith("RESULTADO: 32/32"))
        with self.assertRaises(ValueError):
            self.s.linea("ej1a", "salida-ej1a.txt", "NO EXISTE")
        (self.raiz / "ej1a" / "dos.txt").write_text("A: 1\nA: 2\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.s.linea("ej1a", "dos.txt", "A:")
        self.assertEqual(self.s.lineas_con_prefijo("ej1a", "dos.txt", "A:"), ["A: 1", "A: 2"])

    def test_bloque_returns_n_lines(self):
        b = self.s.bloque("ej1a", "salida-ej1a.txt", "idx", 3)
        self.assertEqual(len(b), 3)
        self.assertTrue(b[0].startswith("idx"))
        self.assertEqual(b[2], "    sangrado con espacios")
        with self.assertRaises(ValueError):
            self.s.bloque("ej1a", "salida-ej1a.txt", "final", 2)
        with self.assertRaises(ValueError):
            self.s.bloque("ej1a", "salida-ej1a.txt", "nada", 1)

    def test_tabla_md_returns_header_and_rows(self):
        enc, filas = self.s.tabla_md("ej1a", "comparison.md", "arch")
        self.assertEqual(enc, ["variante", "semillas", "mediana"])
        self.assertEqual(filas, [["arch-h24", "3/3", "1100"], ["arch-h32x16", "2/3", "575"]])
        enc, filas = self.s.tabla_md("ej1a", "comparison.md", "optim")
        self.assertEqual(enc, ["variante", "semillas"])
        self.assertEqual(filas, [["optim-sgd", "0/3"]])
        with self.assertRaises(ValueError):
            self.s.tabla_md("ej1a", "comparison.md", "otro")

    def test_csv_rows_are_dicts(self):
        self.assertEqual(
            self.s.csv("ej1a", "r.csv"),
            [{"nivel": "0.10", "valor": "3"}, {"nivel": "0.20", "valor": "4"}],
        )

    def test_png_checks_the_signature(self):
        self.assertEqual(self.s.png("ej1a", "g.png"), self.raiz / "ej1a" / "g.png")
        with self.assertRaises(ValueError):
            self.s.png("ej1a", "falso.png")

    def test_alt_is_independent_of_the_root(self):
        self.assertEqual(self.s.alt("ej1a", "latent.png"), "output/ej1a/latent.png")
        self.assertEqual(Salidas("otra/raiz").alt("ej1a", "latent.png"), "output/ej1a/latent.png")

    def test_campo_reads_a_pipe_separated_field(self):
        linea = self.s.lineas("ej1a", "salida-ej1a.txt")[1]
        self.assertEqual(campo(linea, "configuracion elegida"), "arch-h64x32-s0")
        self.assertEqual(campo(linea, "arquitectura"), "35-64-32-2-32-64-35")
        self.assertEqual(campo(linea, "epocas"), "10000")
        with self.assertRaises(ValueError):
            campo(linea, "optimizador")

    def test_numero_returns_first_group(self):
        linea = "RESULTADO: 32/32 patrones (maximo 0)"
        self.assertEqual(numero(linea, r"(\d+)/\d+ patrones"), "32")
        self.assertEqual(numero(linea, r"\d+/(\d+) patrones"), "32")
        self.assertEqual(numero(linea, r"maximo (\d+)"), "0")

    def test_numero_raises_without_match(self):
        with self.assertRaises(ValueError):
            numero("sin cifras", r"(\d+)")
        with self.assertRaises(ValueError):
            numero("hay 12", r"\d+")  # sin grupo de captura

    def test_module_exports(self):
        for nombre in ("Salidas", "FaltaEntrada", "campo", "numero"):
            self.assertTrue(hasattr(lector, nombre))


if __name__ == "__main__":
    unittest.main()
