"""Comprobaciones estructurales automáticas de TP4-presentacion.pptx.

Falla (salida distinta de cero) cuando: la cantidad de diapositivas no coincide con
--expect-slides, alguna diapositiva no tiene notas, una forma sale de la página, hay
guiones largos (U+2014, U+2013) en texto o notas, un título supera 56 caracteres, una
figura de figures/ no está nombrada en build_deck.py (huérfana) o la altura estimada de un cuadro
de texto lo empuja más allá de 8,5 pulgadas. Solo avisa cuando una diapositiva supera
45 palabras de texto nativo o cuando un cuadro de texto se superpone con una figura.

Además recomputa la matriz U desde runs/som/weights.csv y la compara con neurons.csv.

Uso, desde TP4/ con el intérprete de WSL:
    .venv/bin/python docs/presentacion/check_deck.py --expect-slides 36
"""

from pathlib import Path
import argparse
import csv
import hashlib
import math
import sys

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Emu

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DECK = HERE / "TP4-presentacion.pptx"
FIG = HERE / "figures"
SLIDE_W, SLIDE_H = 16.0, 9.0
TOLERANCE = .02
MAX_TITLE = 56
MAX_WORDS = 45
BOTTOM_LIMIT = 8.5
FOOTER_TOP = 8.4
LONG_DASHES = ("—", "–")


def inches(value):
    return Emu(value).inches


def font_size(shape):
    for paragraph in shape.text_frame.paragraphs:
        if paragraph.font.size:
            return paragraph.font.size.pt
        for run in paragraph.runs:
            if run.font.size:
                return run.font.size.pt
    return 18.0


def is_bold(shape):
    for paragraph in shape.text_frame.paragraphs:
        if paragraph.font.bold:
            return True
        for run in paragraph.runs:
            if run.font.bold:
                return True
    return False


def estimated_extent(shape):
    """Ancho y alto estimados (pulgadas) del texto, con Arial a 0,5 em por carácter."""
    size = font_size(shape)
    width = inches(shape.width)
    chars_per_line = max(1, int(width * 72 / (size * .5)))
    lines, longest = 0, 0
    for paragraph in shape.text_frame.paragraphs:
        text = "".join(run.text for run in paragraph.runs)
        longest = max(longest, len(text))
        lines += max(1, math.ceil(len(text) / chars_per_line))
    height = lines * size * 1.2 / 72 + len(shape.text_frame.paragraphs) * 5 / 72
    text_width = min(width, longest * size * .5 / 72)
    return text_width, height


def boxes_overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def check_deck(expect):
    failures, warnings = [], []
    prs = Presentation(str(DECK))
    slides = list(prs.slides)
    if expect is not None and len(slides) != expect:
        failures.append(f"se esperaban {expect} diapositivas y hay {len(slides)}")
    used = set()
    print(f"{'#':>3}  {'figs':>4} {'pal.':>4}  título")
    for number, slide in enumerate(slides, 1):
        title, words, pictures = "", 0, []
        texts = []
        for shape in slide.shapes:
            left, top = inches(shape.left), inches(shape.top)
            width, height = inches(shape.width), inches(shape.height)
            if (left < -TOLERANCE or top < -TOLERANCE or left + width > SLIDE_W + TOLERANCE
                    or top + height > SLIDE_H + TOLERANCE):
                failures.append(f"diapositiva {number}: una forma sale de la página "
                                f"({left:.2f}, {top:.2f}, {width:.2f}, {height:.2f})")
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                used.add(hashlib.sha1(shape.image.blob).hexdigest())
                pictures.append((left, top, width, height))
                continue
            if not shape.has_text_frame or not shape.text_frame.text.strip():
                continue
            value = shape.text_frame.text
            if any(dash in value for dash in LONG_DASHES):
                failures.append(f"diapositiva {number}: guion largo en el texto")
            size = font_size(shape)
            if size >= 32 and is_bold(shape) and top < 1.0 and not title:
                title = value.strip()
                if len(title) > MAX_TITLE:
                    failures.append(f"diapositiva {number}: título de {len(title)} caracteres "
                                    f"(máximo {MAX_TITLE}): {title}")
            if top >= FOOTER_TOP or size == 15 and top < .3:
                continue
            text_w, text_h = estimated_extent(shape)
            if top + text_h > BOTTOM_LIMIT + TOLERANCE:
                failures.append(f"diapositiva {number}: texto estimado hasta {top + text_h:.2f} in "
                                f"(límite {BOTTOM_LIMIT}): {value[:40]!r}")
            words += len(value.split())
            texts.append((value, (left, top, text_w, text_h)))
        notes = slide.notes_slide.notes_text_frame.text if slide.has_notes_slide else ""
        if not notes.strip():
            failures.append(f"diapositiva {number}: sin notas")
        if any(dash in notes for dash in LONG_DASHES):
            failures.append(f"diapositiva {number}: guion largo en las notas")
        last_line = notes.strip().splitlines()[-1] if notes.strip() else ""
        if "Fuente:" not in last_line:
            warnings.append(f"diapositiva {number}: las notas no terminan con una línea Fuente:")
        if words > MAX_WORDS:
            warnings.append(f"diapositiva {number}: {words} palabras de texto nativo (más de {MAX_WORDS})")
        for value, box in texts:
            if any(boxes_overlap(box, picture) for picture in pictures):
                warnings.append(f"diapositiva {number}: el texto {value[:30]!r} se superpone con una figura")
        print(f"{number:>3}  {len(pictures):>4} {words:>4}  {title}")
    source = (HERE / "build_deck.py").read_text(encoding="utf-8")
    for path in sorted(FIG.glob("*.png")):
        if f'"{path.name}"' not in source and f'"{path.stem}"' not in source:
            failures.append(f"figura huérfana, build_deck.py no la nombra: {path.name}")
    for path in sorted(FIG.glob("*.png")):
        if hashlib.sha1(path.read_bytes()).hexdigest() not in used and f'"{path.name}"' in source:
            warnings.append(f"{path.name} se nombra en build_deck.py pero no coincide byte a byte con la imagen de ninguna diapositiva")
    return failures, warnings


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_u_matrix():
    """Recomputa U desde los pesos y la compara con neurons.csv."""
    weights = read_rows(ROOT / "runs/som/weights.csv")
    neurons = read_rows(ROOT / "runs/som/neurons.csv")
    features = [name for name in weights[0] if name not in ("neuron", "row", "col")]
    vectors = {int(r["neuron"]): [float(r[name]) for name in features] for r in weights}
    position = {int(r["neuron"]): (float(r["x"]), float(r["y"])) for r in neurons}
    worst = 0.0
    for r in neurons:
        i = int(r["neuron"])
        distances = []
        for j, (x, y) in position.items():
            if j != i and math.hypot(x - position[i][0], y - position[i][1]) <= 1.00001:
                distances.append(math.dist(vectors[i], vectors[j]))
        recomputed = sum(distances) / len(distances)
        worst = max(worst, abs(recomputed - float(r["u_distance"])))
    print(f"\nMatriz U: {len(features)} variables, {len(neurons)} neuronas; "
          f"desvío máximo recomputado contra neurons.csv = {worst:.2e}")
    assert worst < 1e-3, f"la matriz U recomputada difiere de neurons.csv en {worst}"
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--expect-slides", type=int, default=None)
    parser.add_argument("--skip-u", action="store_true", help="omite la recomputación de la matriz U")
    args = parser.parse_args()
    failures, warnings = check_deck(args.expect_slides)
    if not args.skip_u:
        try:
            check_u_matrix()
        except (AssertionError, OSError, KeyError, ValueError) as error:
            failures.append(f"matriz U: {error}")
    for item in warnings:
        print(f"AVISO {item}")
    for item in failures:
        print(f"FALLA {item}")
    print(f"\n{len(failures)} fallas, {len(warnings)} avisos")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
