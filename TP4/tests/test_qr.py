"""QR module conversion and decoding."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tp4.analysis import qr
from tp4.analysis.qr import (
    DEFAULT_QUIET_ZONE,
    DEFAULT_SCALE,
    GRAY_THRESHOLD,
    MAX_QUIET_ZONE,
    MAX_SCALE,
    QR_SIDE,
    DecoderUnavailableError,
    decode_matrix,
    decoder_status,
    load_qr_image,
    matrix_to_image,
    qr_name,
)
from tp4.errors import DataError
from tp4.runs import PROJECT_ROOT

NAMES = ("food", "wine", "beverages", "dessert", "specials")
PATHS = [PROJECT_ROOT / "datos" / "hopfield" / f"{n}_menu_qr.png" for n in NAMES]


@pytest.fixture
def no_decoder(monkeypatch):
    """Make `import zxingcpp` fail inside tp4.analysis.qr, whether or not it is installed."""
    real = importlib.import_module

    def fake(name, *args, **kwargs):
        if name == "zxingcpp":
            raise ModuleNotFoundError("No module named 'zxingcpp'")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(qr.importlib, "import_module", fake)


def _save_gray(tmp_path: Path, name: str, pixels: np.ndarray) -> Path:
    path = tmp_path / name
    Image.fromarray(pixels.astype(np.uint8), mode="L").save(path)
    return path


def test_constants():
    assert QR_SIDE == 37
    assert GRAY_THRESHOLD == 128
    assert (DEFAULT_SCALE, DEFAULT_QUIET_ZONE) == (8, 4)
    assert (MAX_SCALE, MAX_QUIET_ZONE) == (16, 10)


def test_qr_name():
    assert qr_name("datos/hopfield/food_menu_qr.png") == "food"
    assert qr_name(Path("x/specials_menu_qr.png")) == "specials"
    assert qr_name("other.png") == "other"


@pytest.mark.parametrize("path", PATHS, ids=NAMES)
def test_original_loads_as_pm1_matrix(path):
    img = load_qr_image(path)
    assert img.matrix.shape == (37, 37)
    assert img.matrix.dtype == np.int8
    assert set(np.unique(img.matrix)) == {-1, 1}
    assert img.n_gray == 0
    assert img.pixels.shape == (37, 37)
    assert img.pixels.dtype == np.uint8
    assert (img.matrix == 1).mean() == pytest.approx((img.pixels < 128).mean())
    assert img.name == qr_name(path)


def _finder_ok(block: np.ndarray) -> bool:
    ring = np.ones((7, 7), dtype=bool)
    ring[1:-1, 1:-1] = False
    mid = np.zeros((7, 7), dtype=bool)
    mid[1:-1, 1:-1] = True
    mid[2:-2, 2:-2] = False
    centre = np.zeros((7, 7), dtype=bool)
    centre[2:-2, 2:-2] = True
    return bool(
        (block[ring] == 1).all() and (block[mid] == -1).all() and (block[centre] == 1).all()
    )


@pytest.mark.parametrize("path", PATHS, ids=NAMES)
def test_finder_patterns_in_three_corners(path):
    m = load_qr_image(path).matrix
    assert _finder_ok(m[:7, :7])
    assert _finder_ok(m[:7, -7:])
    assert _finder_ok(m[-7:, :7])


@pytest.mark.parametrize("path", PATHS, ids=NAMES)
def test_roundtrip_is_pixel_identical(path):
    img = load_qr_image(path)
    assert np.array_equal(matrix_to_image(img.matrix), img.pixels)
    original = np.asarray(Image.open(path).convert("L"))
    assert np.array_equal(matrix_to_image(img.matrix), original)


def test_render_shape_and_quiet_zone():
    m = load_qr_image(PATHS[0]).matrix
    out = matrix_to_image(m, scale=8, quiet_zone=4)
    assert out.shape == (360, 360)
    assert out.dtype == np.uint8
    assert (out[:32, :] == 255).all() and (out[-32:, :] == 255).all()
    assert (out[:, :32] == 255).all() and (out[:, -32:] == 255).all()
    assert np.array_equal(out[32:-32:8, 32:-32:8], matrix_to_image(m))


def test_threshold_127_black_128_white(tmp_path):
    px = np.full((37, 37), 255, dtype=np.uint8)
    px[0, 0], px[0, 1] = 127, 128
    img = load_qr_image(_save_gray(tmp_path, "t.png", px))
    assert img.matrix[0, 0] == 1
    assert img.matrix[0, 1] == -1
    assert img.n_gray == 2


def test_rgb_png_is_converted_to_grayscale(tmp_path):
    rgb = np.full((37, 37, 3), 255, dtype=np.uint8)
    rgb[3, 4] = (0, 0, 0)
    path = tmp_path / "rgb.png"
    Image.fromarray(rgb, mode="RGB").save(path)
    img = load_qr_image(path)
    assert img.matrix[3, 4] == 1
    assert img.matrix.sum() == -(37 * 37) + 2


def test_exact_side_passes_and_other_size_is_rejected(tmp_path):
    ok = _save_gray(tmp_path, "ok.png", np.full((37, 37), 255))
    assert load_qr_image(ok).matrix.shape == (37, 37)
    bad = _save_gray(tmp_path, "bad.png", np.full((37, 38), 255))  # 38 wide, 37 high
    with pytest.raises(DataError, match="bad.png"):
        load_qr_image(bad)
    small = _save_gray(tmp_path, "small.png", np.full((21, 21), 255))
    assert load_qr_image(small, side=21).matrix.shape == (21, 21)


def test_missing_file_is_data_error(tmp_path):
    with pytest.raises(DataError, match="nope.png"):
        load_qr_image(tmp_path / "nope.png")


def test_non_image_file_is_data_error(tmp_path):
    path = tmp_path / "text.png"
    path.write_text("this is not a PNG")
    with pytest.raises(DataError, match="text.png"):
        load_qr_image(path)


@pytest.mark.parametrize(
    "matrix, kwargs",
    [
        (np.zeros((3, 3), dtype=np.int8), {}),
        (np.ones((3, 4), dtype=np.int8), {}),
        (np.ones(9, dtype=np.int8), {}),
        (np.ones((3, 3), dtype=np.int8), {"scale": 0}),
        (np.ones((3, 3), dtype=np.int8), {"scale": 17}),
        (np.ones((3, 3), dtype=np.int8), {"scale": 2.0}),
        (np.ones((3, 3), dtype=np.int8), {"quiet_zone": -1}),
        (np.ones((3, 3), dtype=np.int8), {"quiet_zone": 11}),
    ],
)
def test_matrix_to_image_validation(matrix, kwargs):
    with pytest.raises(ValueError):
        matrix_to_image(matrix, **kwargs)


def test_scale_and_quiet_zone_limits_are_inclusive():
    m = np.array([[1, -1], [-1, 1]], dtype=np.int8)
    assert matrix_to_image(m, scale=16, quiet_zone=10).shape == ((2 + 20) * 16,) * 2
    assert matrix_to_image(m, scale=1, quiet_zone=0).tolist() == [[0, 255], [255, 0]]


def test_decoder_unavailable_status(no_decoder):
    status = decoder_status()
    assert status.available is False
    assert status.name == "zxing-cpp"
    assert status.reason.startswith("ModuleNotFoundError")


def test_decode_matrix_raises_when_unavailable(no_decoder):
    with pytest.raises(DecoderUnavailableError):
        decode_matrix(load_qr_image(PATHS[0]).matrix)


def test_no_module_level_decoder_import():
    tree = ast.parse(Path(qr.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "zxingcpp" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "zxingcpp"


def test_originals_decode_to_menu_urls(capsys):
    pytest.importorskip("zxingcpp")
    texts = []
    for name, path in zip(NAMES, PATHS):
        text = decode_matrix(load_qr_image(path).matrix)
        assert text is not None, name
        assert f"{name}_menu" in text
        texts.append(text)
    assert len(set(texts)) == 5
