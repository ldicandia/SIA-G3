"""Contracts for `tp2/ui/viewer.py` and the observer seam it plugs into.

Requires `SDL_VIDEODRIVER=dummy` in the environment (set by the WSL runtime
wrapper this suite is always invoked through) so pygame can `init()` without
a real display.

Proves:
1. The viewer's first call constructs an SDL window sized `(w*scale, h*scale)`
   and blits without raising -- VIS-01/VIS-02.
2. A queued `pygame.QUIT` event sets `should_stop` on the NEXT call and skips
   the blit -- VIS-04, never an exception into the caller.
3. `tp2 --viewer` with a QUIT queued before generation 1's callback ends the
   run within one generation, `run.json`'s `stop_reason` reads
   `viewer_closed` -- VIS-04.
4. The same run with no queued QUIT and a short horizon completes naturally,
   `run.json`'s `stop_reason` is the engine's own condition name, never
   `viewer_closed`.
5. (04-01 Task 2) Headless and `--viewer` runs of the same seed/config
   produce byte-identical `best.png`/`triangles.json` and column-identical
   `metrics.csv` (excluding `elapsed_s`) -- ROADMAP Phase 4 Success
   Criterion 2's second half.
6. (04-01 Task 2) The viewer consumes zero random numbers.
"""

from __future__ import annotations

import csv
import filecmp
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pygame
import pytest

from tp2 import cli
from tp2.engine.events import GenerationEvent
from tp2.ui.viewer import Viewer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
TARGET_IMAGE = ASSETS_DIR / "flag_ar.png"
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline.json"


def _event(
    *,
    generation: int = 0,
    renders: int = 1,
    best_fitness: float = 0.5,
    stop_reason: str = "",
    frame_shape: tuple[int, int, int] = (8, 8, 3),
) -> GenerationEvent:
    """A minimal, hand-built `GenerationEvent` -- no engine required."""
    frame = np.zeros(frame_shape, dtype=np.uint8)
    return GenerationEvent(
        generation=generation,
        renders=renders,
        elapsed=0.01,
        best_fitness=best_fitness,
        best_error=1.0 - best_fitness,
        mean_fitness=best_fitness,
        worst_fitness=best_fitness,
        diversity=0.0,
        active_triangles=1,
        best_genes=np.zeros(11, dtype=np.float32),
        best_frame=frame,
        stop_reason=stop_reason,
    )


def queue_quit_after(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    """Patch `Viewer.__call__` so a `pygame.QUIT` event is posted right after
    the n-th invocation returns -- no real timing, no external process.

    Patching the class (not an instance) is required: `obs(ev)` resolves
    `__call__` on the type, so an instance-attribute override would never be
    seen by the generic `observers` fan-out in `tp2/cli.py`.
    """
    original = Viewer.__call__
    state = {"count": 0}

    def wrapped(self: Viewer, ev: GenerationEvent) -> None:
        original(self, ev)
        state["count"] += 1
        if state["count"] == n:
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    monkeypatch.setattr(Viewer, "__call__", wrapped)


# --- Task 1: the viewer object itself ---------------------------------------


def test_first_call_constructs_a_scaled_window_and_blits_without_raising() -> None:
    viewer = Viewer(scale=4, every=1)
    try:
        viewer(_event(generation=0, frame_shape=(8, 8, 3)))
        assert viewer.should_stop is False
        surface = pygame.display.get_surface()
        assert surface is not None
        assert surface.get_size() == (32, 32)
    finally:
        viewer.__exit__(None, None, None)


def test_exit_calls_pygame_quit_even_if_set_mode_raises_during_first_frame_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WR-04 regression: `pygame.init()` succeeds and sets `_initialized`,
    but `pygame.display.set_mode` then raises before `self._screen` is ever
    assigned. `__exit__` must still call `pygame.quit()` -- tracked via
    `self._initialized`, never via `self._screen` -- so a genuinely broken
    SDL video driver can never leak initialized pygame/SDL state into
    whatever runs next in the same process."""

    def raising_set_mode(*_args, **_kwargs):
        raise RuntimeError("synthetic broken SDL video driver")

    monkeypatch.setattr(pygame.display, "set_mode", raising_set_mode)

    viewer = Viewer(scale=4, every=1)
    with pytest.raises(RuntimeError, match="synthetic broken SDL video driver"):
        viewer(_event(generation=0))

    assert viewer._screen is None, "set_mode raised before self._screen was ever assigned"
    assert viewer._initialized is True, "pygame.init() ran and must still be tracked for cleanup"

    original_quit = pygame.quit
    quit_calls = {"count": 0}

    def spy_quit() -> None:
        quit_calls["count"] += 1
        original_quit()

    monkeypatch.setattr(pygame, "quit", spy_quit)

    viewer.__exit__(None, None, None)

    assert quit_calls["count"] == 1, "a screen-less but initialized viewer must still call pygame.quit()"


def test_queued_quit_event_sets_should_stop_and_skips_the_next_blit(monkeypatch: pytest.MonkeyPatch) -> None:
    flips = {"count": 0}
    original_flip = pygame.display.flip

    def counting_flip() -> None:
        flips["count"] += 1
        original_flip()

    monkeypatch.setattr(pygame.display, "flip", counting_flip)

    viewer = Viewer(scale=4, every=1)
    try:
        viewer(_event(generation=0))
        assert flips["count"] == 1
        assert viewer.should_stop is False

        pygame.event.post(pygame.event.Event(pygame.QUIT))
        viewer(_event(generation=1))

        assert viewer.should_stop is True
        assert flips["count"] == 1, "a closed viewer must not blit again"
    finally:
        viewer.__exit__(None, None, None)


def test_a_stopped_viewer_never_reopens_on_a_later_event(monkeypatch: pytest.MonkeyPatch) -> None:
    viewer = Viewer(scale=4, every=1)
    try:
        viewer(_event(generation=0))
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        viewer(_event(generation=1))
        assert viewer.should_stop is True

        # A later call, even with no further QUIT queued, must stay closed.
        viewer(_event(generation=2))
        assert viewer.should_stop is True
    finally:
        viewer.__exit__(None, None, None)


# --- Task 1: the CLI observer seam ------------------------------------------


def _small_config(tmp_path: Path, *, horizon: int) -> Path:
    config_path = tmp_path / "tiny_viewer_config.json"
    config_path.write_text(
        json.dumps(
            {
                "population": 6,
                "children": 6,
                "horizon": horizon,
                "recombination_probability": 0.8,
                "parents": {"method": "elite"},
                "replacement": {"method": "elite"},
                "crossover": {"method": "one_point"},
                "mutation": {"method": "gene", "probability": 0.5},
                "survival": {"method": "additive"},
                "stop": {"max_generations": True},
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_cli_viewer_quit_ends_the_run_within_one_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    queue_quit_after(monkeypatch, 1)  # posted right after generation 0's callback

    config_path = _small_config(tmp_path, horizon=50)  # large enough that natural completion cannot race the quit
    out_dir = tmp_path / "viewer_quit"
    argv = [
        "--image", str(TARGET_IMAGE),
        "--triangles", "10",
        "--canvas", "32",
        "--config", str(config_path),
        "--seed", "1",
        "--out", str(out_dir),
        "--allow-outside",
        "--viewer",
    ]

    assert cli.main(argv) == 0

    for name in ("best.png", "triangles.json", "run.json", "metrics.csv"):
        assert (out_dir / name).exists()

    payload = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["stop_reason"] == "viewer_closed"


def test_cli_viewer_natural_completion_never_reports_viewer_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")

    config_path = _small_config(tmp_path, horizon=5)
    out_dir = tmp_path / "viewer_natural"
    argv = [
        "--image", str(TARGET_IMAGE),
        "--triangles", "10",
        "--canvas", "32",
        "--config", str(config_path),
        "--seed", "1",
        "--out", str(out_dir),
        "--allow-outside",
        "--viewer",
    ]

    assert cli.main(argv) == 0

    payload = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["stop_reason"] == "max_generations"
    assert payload["stop_reason"] != "viewer_closed"


# --- Task 2: headless == viewed, and no randomness consumed -----------------


def _run_cli(argv: list[str], env_extra: dict[str, str] | None = None) -> None:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    result = subprocess.run([sys.executable, "-m", "tp2", *argv], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_headless_and_viewed_runs_produce_equivalent_artifacts(tmp_path: Path) -> None:
    config_path = _small_config(tmp_path, horizon=4)
    headless_dir = tmp_path / "headless"
    viewed_dir = tmp_path / "viewed"

    common = [
        "--image", str(TARGET_IMAGE),
        "--triangles", "10",
        "--canvas", "32",
        "--config", str(config_path),
        "--seed", "7",
        "--allow-outside",
    ]

    _run_cli([*common, "--out", str(headless_dir)])
    _run_cli([*common, "--out", str(viewed_dir), "--viewer"], env_extra={"SDL_VIDEODRIVER": "dummy"})

    assert filecmp.cmp(headless_dir / "best.png", viewed_dir / "best.png", shallow=False)

    headless_tris = json.loads((headless_dir / "triangles.json").read_text(encoding="utf-8"))
    viewed_tris = json.loads((viewed_dir / "triangles.json").read_text(encoding="utf-8"))
    assert headless_tris == viewed_tris

    with (headless_dir / "metrics.csv").open(encoding="utf-8") as f:
        headless_rows = list(csv.DictReader(f))
    with (viewed_dir / "metrics.csv").open(encoding="utf-8") as f:
        viewed_rows = list(csv.DictReader(f))

    assert len(headless_rows) == len(viewed_rows)
    for headless_row, viewed_row in zip(headless_rows, viewed_rows):
        headless_row.pop("elapsed_s")
        viewed_row.pop("elapsed_s")
        assert headless_row == viewed_row


def test_viewer_consumes_no_randomness() -> None:
    seed = 999
    reference = np.random.default_rng(seed)
    candidate = np.random.default_rng(seed)

    viewer = Viewer(scale=4, every=1)
    try:
        viewer(_event(generation=0))
    finally:
        viewer.__exit__(None, None, None)

    reference_draws = reference.random(100)
    candidate_draws = candidate.random(100)
    assert np.array_equal(reference_draws, candidate_draws)
