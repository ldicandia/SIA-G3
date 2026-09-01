"""Generate the shipped targets entirely from deterministic Pillow primitives."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ASSET_SIZE = (256, 256)
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

# Filenames in the same order main()/generate_all() write them.
ASSET_FILES = ("flag_ar.png", "silhouette.png", "pictogram.png")

# Explicit PNG encoder parameters, identical to tp2/io/images.py:save_png, so
# output bytes stop depending on Pillow's per-release encoder defaults
# (01-VERIFICATION.md gap #1 / IO-08).
PNG_SAVE_KWARGS = {"format": "PNG", "optimize": False, "compress_level": 6}


def make_flag() -> Image.Image:
    image = Image.new("RGB", ASSET_SIZE, "white")
    draw = ImageDraw.Draw(image)
    blue = (116, 172, 223)
    height = ASSET_SIZE[1] // 3
    draw.rectangle((0, 0, 255, height - 1), fill=blue)
    draw.rectangle((0, 2 * height, 255, 255), fill=blue)
    return image


def make_silhouette() -> Image.Image:
    image = Image.new("RGB", ASSET_SIZE, "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((88, 34, 168, 114), fill="black")
    draw.rounded_rectangle((48, 112, 208, 234), radius=50, fill="black")
    return image


def make_pictogram() -> Image.Image:
    image = Image.new("RGB", ASSET_SIZE, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((35, 35, 220, 220), outline="black", width=14)
    draw.polygon(((128, 63), (62, 171), (194, 171)), fill="black")
    return image


_MAKERS = {"flag_ar.png": make_flag, "silhouette.png": make_silhouette, "pictogram.png": make_pictogram}


def generate_all(out_dir: Path) -> None:
    """Write all three assets into out_dir under the pinned encoder params.

    The single write-path used by both the default (no-flag) mode and by
    check()'s temp-directory regeneration.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename in ASSET_FILES:
        _MAKERS[filename]().save(out_dir / filename, **PNG_SAVE_KWARGS)


def check(against: Path = ASSETS_DIR) -> bool:
    """Regenerate all three assets into a fresh temp dir and byte-compare.

    Never writes into `against` itself, so a failing check cannot dirty a
    tracked binary. Returns True only if all three files match byte-for-byte.
    """
    all_match = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        generate_all(tmp_dir)
        for filename in ASSET_FILES:
            committed = (against / filename).read_bytes()
            regenerated = (tmp_dir / filename).read_bytes()
            if committed == regenerated:
                print(f"OK: {filename}")
            else:
                print(f"MISMATCH: {filename}", file=sys.stderr)
                all_match = False
    return all_match


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate into a temp dir and byte-compare against assets/ without writing to it",
    )
    args = parser.parse_args()

    if args.check:
        sys.exit(0 if check() else 1)

    generate_all(ASSETS_DIR)


if __name__ == "__main__":
    main()
