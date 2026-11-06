"""Pruebas del escritor minimo de .pptx y del verificador estructural.

Sin dependencia de output/: los PNG se fabrican en la prueba.
"""

import io
import struct
import sys
import tempfile
import unittest
import warnings
import zlib
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pptx_minimo as pm  # noqa: E402
import verificar as vf  # noqa: E402

NS_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def png_sintetico(ancho: int, alto: int) -> bytes:
    """PNG valido de un solo color, ancho x alto."""

    def trozo(tipo: bytes, datos: bytes) -> bytes:
        crc = zlib.crc32(tipo + datos) & 0xFFFFFFFF
        return struct.pack(">I", len(datos)) + tipo + datos + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", ancho, alto, 8, 2, 0, 0, 0)
    fila = b"\x00" + b"\x20\x80\xa0" * ancho
    return (
        b"\x89PNG\r\n\x1a\n"
        + trozo(b"IHDR", ihdr)
        + trozo(b"IDAT", zlib.compress(fila * alto))
        + trozo(b"IEND", b"")
    )


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.png = self.dir / "g.png"
        self.png.write_bytes(png_sintetico(200, 100))

    def mazo_ejemplo(self) -> Path:
        p = pm.Presentacion("Ejemplo")
        portada = p.diapositiva("Portada de ejemplo", portada=True)
        portada.texto(0.9, 4.2, 11.5, 0.6, "Subtitulo", tam=20)
        d = p.diapositiva("Ej 1a: Resultados")
        d.texto(0.5, 1.7, 6, 1.5, [pm.Parrafo("Uno", vineta=True), "Dos & tres < cuatro > cinco"])
        d.codigo(0.5, 3.4, 6, 1.0, ["RESULTADO: 32/32", "   con sangria", ""], fuente="ej1a/salida-ej1a.txt")
        d.imagen(self.png, 7, 1.7, 5, 3, alt="output/ej1a/g.png")
        d.tabla(0.5, 4.6, 8, [["a", "b"], ["1", "2"], ["3", "4"]])
        d.caja(9, 4.6, 3.5, 1.0, "Caja")
        d.cifra(9, 5.8, 3.5, 1.1, "32/32", "patrones")
        ruta = self.dir / "ejemplo.pptx"
        p.guardar(ruta)
        return ruta

    def partes(self, ruta) -> list[tuple[str, bytes]]:
        with zipfile.ZipFile(ruta) as z:
            return [(i.filename, z.read(i.filename)) for i in z.infolist()]

    def reescribir(self, partes, nombre="mod.pptx", comprimir=False) -> Path:
        ruta = self.dir / nombre
        modo = zipfile.ZIP_DEFLATED if comprimir else zipfile.ZIP_STORED
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with zipfile.ZipFile(ruta, "w", compression=modo) as z:
                for n, datos in partes:
                    z.writestr(n, datos)
        return ruta


class PruebasEscritor(Base):
    def test_inches_to_emu_is_exact_integer(self):
        self.assertEqual(pm.emu(1), 914400)
        self.assertEqual(pm.emu(0.5), 457200)
        self.assertIsInstance(pm.emu(2.25), int)
        self.assertEqual((pm.ANCHO_EMU, pm.ALTO_EMU), (12192000, 6858000))
        ruta = self.mazo_ejemplo()
        pres = dict(self.partes(ruta))["ppt/presentation.xml"].decode()
        self.assertIn('<p:sldSz cx="12192000" cy="6858000"/>', pres)

    def test_text_is_xml_escaped(self):
        ruta = self.mazo_ejemplo()
        xml = dict(self.partes(ruta))["ppt/slides/slide2.xml"].decode()
        self.assertIn("Dos &amp; tres &lt; cuatro &gt; cinco", xml)
        self.assertNotIn("Dos & tres", xml)
        self.assertIn((2, "Dos & tres < cuatro > cinco"), vf.textos(ruta))

    def test_control_characters_are_rejected(self):
        p = pm.Presentacion("t")
        d = p.diapositiva("Titulo")
        for malo in ("a\x00b", "a\x1bb", "a\x7fb", "a\x85b"):
            with self.assertRaises(pm.ErrorDeLayout):
                d.texto(0.5, 1.7, 6, 1, malo)
        d.texto(0.5, 1.7, 6, 1, "con\ttab\ny salto")  # tab y salto de linea se aceptan
        with self.assertRaises(pm.ErrorDeLayout):
            p.diapositiva("Tit\x01ulo")

    def test_png_header_gives_size(self):
        self.assertEqual(pm.dimensiones_png(png_sintetico(321, 123)), (321, 123))

    def test_non_png_is_rejected(self):
        with self.assertRaises(pm.ErrorDeLayout):
            pm.dimensiones_png(b"GIF89a" + b"\0" * 40)
        malo = self.dir / "no.png"
        malo.write_bytes(b"esto no es un png")
        d = pm.Presentacion("t").diapositiva("Titulo")
        with self.assertRaises(pm.ErrorDeLayout):
            d.imagen(malo, 1, 2, 3, 3, alt="x")
        with self.assertRaises(pm.ErrorDeLayout):
            d.imagen(self.dir / "inexistente.png", 1, 2, 3, 3, alt="x")

    def test_image_is_fitted_inside_box_keeping_aspect_and_centred(self):
        x, y, w, h = pm.ajustar_imagen(200, 100, 1.0, 2.0, 4.0, 4.0)
        self.assertAlmostEqual(w / h, 2.0)
        self.assertAlmostEqual(w, 4.0)
        self.assertAlmostEqual(h, 2.0)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 3.0)  # centrada en vertical
        x, y, w, h = pm.ajustar_imagen(100, 200, 1.0, 2.0, 4.0, 4.0)
        self.assertAlmostEqual(h, 4.0)
        self.assertAlmostEqual(w, 2.0)
        self.assertAlmostEqual(x, 2.0)  # centrada en horizontal

    def test_text_that_does_not_fit_raises(self):
        d = pm.Presentacion("t").diapositiva("Titulo")
        largo = "palabra " * 200
        with self.assertRaises(pm.ErrorDeLayout) as ctx:
            d.texto(0.5, 1.7, 4, 0.5, largo, tam=16)
        self.assertIn("Titulo", str(ctx.exception))
        with self.assertRaises(pm.ErrorDeLayout):
            d.codigo(0.5, 1.7, 3, 0.4, ["x" * 200], fuente="a/b.txt")

    def test_text_that_fits_does_not_raise(self):
        d = pm.Presentacion("t").diapositiva("Titulo")
        d.texto(0.5, 1.7, 6, 1.0, "Una linea corta", tam=16)
        # 0.52 * 16 / 72 = 0.1156 in por caracter; 5.8 in utiles -> 50 caracteres por renglon
        d.texto(0.5, 3.0, 6, 0.5, "x" * 50, tam=16)
        self.assertEqual(pm.altura_estimada([pm.Parrafo("x" * 50)], 6, tam=16), 16 * 1.2 / 72 + 0.1)
        self.assertGreater(pm.altura_estimada([pm.Parrafo("x" * 51)], 6, tam=16), 2 * 16 * 1.2 / 72)

    def test_height_estimate_uses_monospace_and_bullet_rules(self):
        mono = pm.altura_estimada([pm.Parrafo("x" * 30)], 4, tam=10, mono=True)
        prop = pm.altura_estimada([pm.Parrafo("x" * 30)], 4, tam=10, mono=False)
        self.assertEqual(prop, 10 * 1.2 / 72 + 0.1)
        # 30 * 0.6 * 10 / 72 = 2.5 in; utiles 3.8 -> una linea tambien en mono
        self.assertEqual(mono, prop)
        # con vineta (0.25 de sangria) y ancho justo, pasa a dos renglones
        sin = pm.altura_estimada([pm.Parrafo("x" * 31)], 4, tam=10, mono=True)
        con = pm.altura_estimada([pm.Parrafo("x" * 31, vineta=True)], 4, tam=10, mono=True)
        self.assertEqual(sin, prop)
        self.assertEqual(con, prop)  # 31 * 0.0833 = 2.58 in < 3.55
        largo = pm.altura_estimada([pm.Parrafo("x" * 60, vineta=True)], 4, tam=10, mono=True)
        self.assertGreater(largo, prop)

    def test_shape_outside_slide_or_margins_raises(self):
        d = pm.Presentacion("t").diapositiva("Titulo")
        casos = [
            (0.2, 1.7, 3, 1),  # margen izquierdo
            (0.5, 0.2, 3, 1),  # margen superior
            (10.5, 1.7, 3, 1),  # sale por la derecha (10.5 + 3 > 12.833)
            (0.5, 6.5, 3, 1),  # sale por abajo (6.5 + 1 > 7.0)
            (-1, 1.7, 3, 1),  # fuera de la diapositiva
            (0.5, 1.7, 15, 1),  # mas ancha que la diapositiva
        ]
        for x, y, w, h in casos:
            with self.assertRaises(pm.ErrorDeLayout, msg=str((x, y, w, h))):
                d.texto(x, y, w, h, "x")
            with self.assertRaises(pm.ErrorDeLayout):
                d.caja(x, y, w, h, "x")
        d.texto(0.5, 1.7, 12.333, 5.3, "justo en el borde")

    def test_table_taller_than_the_slide_raises(self):
        d = pm.Presentacion("t").diapositiva("Titulo")
        filas = [["fila", "x"]] * 40
        with self.assertRaises(pm.ErrorDeLayout):
            d.tabla(0.5, 1.7, 8, filas)

    def test_package_is_stored_not_compressed(self):
        ruta = self.mazo_ejemplo()
        with zipfile.ZipFile(ruta) as z:
            self.assertTrue(z.infolist())
            for info in z.infolist():
                self.assertEqual(info.compress_type, zipfile.ZIP_STORED, info.filename)
            self.assertEqual(z.infolist()[0].filename, "[Content_Types].xml")
            self.assertEqual(
                [i.filename for i in z.infolist()[1:]], sorted(i.filename for i in z.infolist()[1:])
            )

    def test_package_bytes_are_deterministic(self):
        a = self.mazo_ejemplo().read_bytes()
        b = self.mazo_ejemplo().read_bytes()
        self.assertEqual(a, b)
        p1 = self.dir / "otro" / "x.pptx"
        p = pm.Presentacion("Ejemplo")
        p.diapositiva("T")
        p.guardar(p1)  # crea los directorios padre
        self.assertTrue(p1.is_file())

    def test_code_block_lines_are_one_run_each(self):
        ruta = self.mazo_ejemplo()
        raiz = ET.fromstring(dict(self.partes(ruta))["ppt/slides/slide2.xml"])
        bloque = [
            sp
            for sp in raiz.iter("{http://schemas.openxmlformats.org/presentationml/2006/main}sp")
            if sp.find(".//{http://schemas.openxmlformats.org/presentationml/2006/main}cNvPr").get("name")
            == "salida: ej1a/salida-ej1a.txt"
        ]
        self.assertEqual(len(bloque), 1)
        parrafos = bloque[0].findall(f".//{NS_A}p")
        self.assertEqual(len(parrafos), 3)
        for par in parrafos:
            self.assertEqual(len(par.findall(f"{NS_A}r")), 1)
        self.assertEqual(
            ["".join(t.text or "" for t in par.iter(f"{NS_A}t")) for par in parrafos],
            ["RESULTADO: 32/32", "   con sangria", ""],
        )
        self.assertIn((2, "   con sangria"), vf.textos(ruta))

    def test_table_cells_are_one_run_each(self):
        ruta = self.mazo_ejemplo()
        raiz = ET.fromstring(dict(self.partes(ruta))["ppt/slides/slide2.xml"])
        celdas = raiz.findall(f".//{NS_A}tc")
        self.assertEqual(len(celdas), 6)
        for c in celdas:
            self.assertEqual(len(c.findall(f".//{NS_A}r")), 1)
        self.assertEqual(len(raiz.findall(f".//{NS_A}gridCol")), 2)

    def test_picture_description_equals_alt(self):
        ruta = self.mazo_ejemplo()
        raiz = ET.fromstring(dict(self.partes(ruta))["ppt/slides/slide2.xml"])
        pic = raiz.find(".//{http://schemas.openxmlformats.org/presentationml/2006/main}pic")
        cnv = pic.find(".//{http://schemas.openxmlformats.org/presentationml/2006/main}cNvPr")
        self.assertEqual(cnv.get("descr"), "output/ej1a/g.png")
        with self.assertRaises(pm.ErrorDeLayout):
            pm.Presentacion("t").diapositiva("T").imagen(self.png, 1, 2, 3, 3, alt="  ")

    def test_counts_and_media_are_shared(self):
        p = pm.Presentacion("t")
        d = p.diapositiva("Uno")
        d.imagen(self.png, 0.5, 1.7, 3, 2, alt="a")
        d.imagen(self.png, 4.5, 1.7, 3, 2, alt="b")
        self.assertEqual((p.n_diapositivas, p.n_imagenes), (1, 2))
        ruta = self.dir / "m.pptx"
        p.guardar(ruta)
        medios = [n for n, _ in self.partes(ruta) if n.startswith("ppt/media/")]
        self.assertEqual(medios, ["ppt/media/image1.png"])
        self.assertEqual(vf.verificar_estructura(ruta), [])

    def test_special_characters_in_attributes_keep_package_well_formed(self):
        p = pm.Presentacion('Titulo & "comillas" <x>')
        d = p.diapositiva("A & B")
        d.imagen(self.png, 0.5, 1.7, 3, 2, alt='alt "con" <marcas> & mas')
        d.codigo(0.5, 4, 6, 1, ["x"], fuente="a&b/c<d>.txt")
        ruta = self.dir / "e.pptx"
        p.guardar(ruta)
        self.assertEqual(vf.verificar_estructura(ruta), [])
        self.assertEqual(vf.titulos(ruta), ["A & B"])

    def test_font_size_floors(self):
        d = pm.Presentacion("t").diapositiva("Titulo")
        with self.assertRaises(pm.ErrorDeLayout):
            d.texto(0.5, 1.7, 6, 1, "x", tam=9)
        with self.assertRaises(pm.ErrorDeLayout):
            d.tabla(0.5, 1.7, 6, [["a"]], tam=10)
        with self.assertRaises(pm.ErrorDeLayout):
            d.codigo(0.5, 1.7, 6, 1, ["x"], fuente="f", tam=9)


class PruebasVerificador(Base):
    def test_verifier_accepts_a_valid_deck(self):
        ruta = self.mazo_ejemplo()
        self.assertEqual(vf.verificar_estructura(ruta), [])

    def test_verifier_detects_missing_relationship_target(self):
        partes = [
            (n, d.replace(b"slideLayout1.xml", b"slideLayoutX.xml") if n == "ppt/slides/_rels/slide1.xml.rels" else d)
            for n, d in self.partes(self.mazo_ejemplo())
        ]
        errores = vf.verificar_estructura(self.reescribir(partes))
        self.assertTrue(any("destino inexistente" in e for e in errores), errores)

    def test_verifier_detects_part_without_content_type(self):
        partes = self.partes(self.mazo_ejemplo()) + [("ppt/extra.bin", b"datos")]
        errores = vf.verificar_estructura(self.reescribir(partes))
        self.assertTrue(any("sin tipo de contenido" in e and "extra.bin" in e for e in errores), errores)

    def test_verifier_detects_compressed_members(self):
        errores = vf.verificar_estructura(self.reescribir(self.partes(self.mazo_ejemplo()), comprimir=True))
        self.assertTrue(any("comprimido" in e for e in errores), errores)

    def test_verifier_detects_malformed_xml(self):
        partes = [
            (n, d[: len(d) // 2] if n == "ppt/slides/slide1.xml" else d)
            for n, d in self.partes(self.mazo_ejemplo())
        ]
        errores = vf.verificar_estructura(self.reescribir(partes))
        self.assertTrue(any("mal formada" in e and "slide1.xml" in e for e in errores), errores)

    def test_verifier_detects_duplicate_member_names(self):
        partes = self.partes(self.mazo_ejemplo())
        partes.append(partes[3])
        errores = vf.verificar_estructura(self.reescribir(partes))
        self.assertTrue(any("duplicado" in e for e in errores), errores)

    def test_verifier_detects_slide_list_mismatch(self):
        partes = [
            (n, d.replace(b'<p:sldId id="257" r:id="rId3"/>', b"") if n == "ppt/presentation.xml" else d)
            for n, d in self.partes(self.mazo_ejemplo())
        ]
        errores = vf.verificar_estructura(self.reescribir(partes))
        self.assertTrue(any("diapositiva" in e for e in errores), errores)

    def test_verifier_rejects_unsafe_member_names(self):
        partes = self.partes(self.mazo_ejemplo()) + [("../fuera.xml", b"<a/>"), ("/abs.xml", b"<a/>")]
        errores = vf.verificar_estructura(self.reescribir(partes))
        self.assertEqual(sum("inseguro" in e for e in errores), 2, errores)

    def test_verifier_rejects_a_file_that_is_not_a_zip(self):
        ruta = self.dir / "no.pptx"
        ruta.write_bytes(b"no soy un zip")
        self.assertTrue(vf.verificar_estructura(ruta))

    def test_titulos_and_textos_in_order(self):
        ruta = self.mazo_ejemplo()
        self.assertEqual(vf.titulos(ruta), ["Portada de ejemplo", "Ej 1a: Resultados"])
        t = vf.textos(ruta)
        self.assertEqual(t[0], (1, "Portada de ejemplo"))
        slide2 = [x for n, x in t if n == 2]
        self.assertEqual(slide2[0], "Ej 1a: Resultados")
        for esperado in ("RESULTADO: 32/32", "   con sangria", "a", "b", "1", "2", "3", "4", "Caja", "32/32", "patrones"):
            self.assertIn(esperado, slide2)

    def test_cli_prints_ok_and_listing(self):
        ruta = self.mazo_ejemplo()
        salida = io.StringIO()
        antes = sys.stdout
        sys.stdout = salida
        try:
            codigo = vf.main(["--pptx", str(ruta), "--listar", "--texto"])
        finally:
            sys.stdout = antes
        lineas = salida.getvalue().splitlines()
        self.assertEqual(codigo, 0)
        self.assertTrue(lineas[0].startswith("ESTRUCTURA OK: "))
        self.assertIn("1: Portada de ejemplo", lineas)
        self.assertIn("2: Ej 1a: Resultados", lineas)
        self.assertIn("2:    con sangria", lineas)


if __name__ == "__main__":
    unittest.main()
