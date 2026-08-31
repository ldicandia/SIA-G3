"""Generate the shipped targets entirely from deterministic Pillow primitives."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ASSET_SIZE = (256, 256)
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


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


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, maker in (("flag_ar.png", make_flag), ("silhouette.png", make_silhouette), ("pictogram.png", make_pictogram)):
        maker().save(ASSETS_DIR / filename, format="PNG")


if __name__ == "__main__":
    main()
