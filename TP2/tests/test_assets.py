"""Byte-determinism contracts for scripts/make_assets.py (01-VERIFICATION.md gap #1 / IO-08).

`scripts/` has no `__init__.py` and is imported here as a Python 3 namespace
package under pytest.ini's `pythonpath = .` — the same mechanism `tests/`
already uses to reach into `tp2/`.
"""

from __future__ import annotations

from pathlib import Path

from scripts.make_assets import ASSET_FILES, check, generate_all


def test_committed_assets_regenerate_byte_identically_via_check_mode() -> None:
    assert check() is True


def test_make_assets_generation_is_deterministic_across_two_runs(tmp_path: Path) -> None:
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    generate_all(out1)
    generate_all(out2)
    for name in ASSET_FILES:
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes()
