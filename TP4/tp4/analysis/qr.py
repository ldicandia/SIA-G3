"""QR input and output ends for the Hopfield experiment (QR-01, QR-04).

Loading contract (QR-01): a menu PNG is read with Pillow only, converted to grayscale
(mode "L") and never resized or cropped. A pixel below `GRAY_THRESHOLD` (128) is black
and maps to +1; 128 or above is white and maps to -1. An image that is not exactly
`side` x `side` is rejected with `DataError` (resampling would invent modules). The
stored matrix is the code itself: no quiet zone. The quiet zone and the upscaling exist
only in the image handed to the decoder (`matrix_to_image`).

Decoder (QR-04): `zxing-cpp` is an OPTIONAL extra (`requirements-qr.txt`). It is never
imported at module level: `decoder_status()` imports it lazily, so every run and every
test works without it and `decode_matrix` raises `DecoderUnavailableError` instead of
crashing a caller that did not check.

CLI:

    python -m tp4.analysis.qr IMAGE [IMAGE ...] [--side N] [--scale S] [--quiet-zone Q]

Exit code 0 when every image loads, round-trips and decodes; 1 when any image does not
decode or the decoder is unavailable; 2 with `error: ...` on stderr for an unreadable image.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from tp4.errors import DataError, Tp4Error
from tp4.runs import resolve_project_path

QR_SIDE = 37
GRAY_THRESHOLD = 128
DEFAULT_SCALE = 8
DEFAULT_QUIET_ZONE = 4
MAX_SCALE = 16
MAX_QUIET_ZONE = 10
DECODER_PACKAGE = "zxing-cpp"
DECODER_MODULE = "zxingcpp"

_MAX_REASON = 200
_SUFFIX = "_menu_qr"


@dataclass(frozen=True)
class QrImage:
    name: str
    path: Path
    pixels: np.ndarray  # side x side uint8, as stored in the file (mode L)
    matrix: np.ndarray  # side x side int8, black = +1, white = -1
    n_gray: int  # pixels that are neither 0 nor 255


@dataclass(frozen=True)
class DecoderStatus:
    available: bool
    name: str
    version: str | None
    reason: str | None


class DecoderUnavailableError(RuntimeError):
    """The optional QR decoder (zxing-cpp) cannot be imported."""


def qr_name(path: str | Path) -> str:
    """`food` for `food_menu_qr.png`: the stem without a trailing `_menu_qr`."""
    stem = Path(path).stem
    return stem[: -len(_SUFFIX)] if stem.endswith(_SUFFIX) else stem


def load_qr_image(path: str | Path, side: int = QR_SIDE) -> QrImage:
    """Load a QR PNG as an exact `side` x `side` matrix of +-1 (black = +1)."""
    resolved = resolve_project_path(path)
    label = resolved.name
    try:
        with Image.open(resolved) as handle:
            gray = handle.convert("L")
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise DataError(f"cannot read QR image {label}: {exc}") from exc
    width, height = gray.size
    if (width, height) != (side, side):
        raise DataError(f"QR image {label} must be {side}x{side}, got {width}x{height}")
    pixels = np.asarray(gray, dtype=np.uint8).copy()
    matrix = np.where(pixels < GRAY_THRESHOLD, 1, -1).astype(np.int8)
    n_gray = int(((pixels != 0) & (pixels != 255)).sum())
    return QrImage(qr_name(resolved), resolved, pixels, matrix, n_gray)


def _check_int(value: Any, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an int in [{lo}, {hi}], got {value!r}")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {int(value)}")
    return int(value)


def matrix_to_image(matrix: np.ndarray, scale: int = 1, quiet_zone: int = 0) -> np.ndarray:
    """Render a +-1 matrix as a uint8 image (+1 -> 0 black, -1 -> 255 white).

    Each module becomes `scale` x `scale` pixels, then `quiet_zone` modules of white are
    added on every side.
    """
    m = np.asarray(matrix)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"matrix must be square and 2-D, got shape {m.shape}")
    if not np.isin(m, (-1, 1)).all():
        raise ValueError("matrix values must be -1 or +1")
    scale = _check_int(scale, "scale", 1, MAX_SCALE)
    quiet_zone = _check_int(quiet_zone, "quiet_zone", 0, MAX_QUIET_ZONE)
    img = np.where(m == 1, 0, 255).astype(np.uint8)
    img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    pad = quiet_zone * scale
    return np.pad(img, pad, mode="constant", constant_values=255) if pad else img


def decoder_status() -> DecoderStatus:
    """Import the decoder lazily and report whether it is usable (re-checked on every call)."""
    try:
        importlib.import_module(DECODER_MODULE)
    except Exception as exc:  # any import failure degrades to "unavailable", never raises
        reason = f"{type(exc).__name__}: {exc}"[:_MAX_REASON]
        return DecoderStatus(False, DECODER_PACKAGE, None, reason)
    try:
        version: str | None = importlib.metadata.version(DECODER_PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return DecoderStatus(True, DECODER_PACKAGE, version, None)


def decode_matrix(
    matrix: np.ndarray,
    *,
    scale: int = DEFAULT_SCALE,
    quiet_zone: int = DEFAULT_QUIET_ZONE,
) -> str | None:
    """Decode a +-1 matrix with the optional decoder; None when no QR symbol is found."""
    status = decoder_status()
    if not status.available:
        raise DecoderUnavailableError(status.reason)
    image = matrix_to_image(matrix, scale=scale, quiet_zone=quiet_zone)
    module = importlib.import_module(DECODER_MODULE)
    for result in module.read_barcodes(image):
        if "QR" in str(result.format):
            return str(result.text)
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tp4.analysis.qr",
        description="Load menu QR PNGs as +-1 matrices, check the roundtrip and decode them.",
    )
    parser.add_argument("images", nargs="+", help="QR PNG files")
    parser.add_argument("--side", type=int, default=QR_SIDE)
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE)
    parser.add_argument("--quiet-zone", type=int, default=DEFAULT_QUIET_ZONE)
    args = parser.parse_args(argv)

    status = decoder_status()
    if not status.available:
        print(f"decoder unavailable: {status.reason}")

    ok = status.available
    try:
        for image_path in args.images:
            img = load_qr_image(image_path, side=args.side)
            roundtrip = np.array_equal(matrix_to_image(img.matrix), img.pixels)
            black = float((img.matrix == 1).mean())
            tag = "roundtrip ok" if roundtrip else "roundtrip MISMATCH"
            print(f"{image_path}: {args.side}x{args.side}, black {black:.4f}, {tag}")
            ok = ok and roundtrip
            if status.available:
                text = decode_matrix(img.matrix, scale=args.scale, quiet_zone=args.quiet_zone)
                print(f"  decoded: {text if text is not None else 'NONE'}")
                ok = ok and text is not None
    except (Tp4Error, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
