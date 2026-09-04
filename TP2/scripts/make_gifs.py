"""Generate a small, representative set of animated GIFs from real GA runs.

Drives the same public engine composition `tp2/cli.py`'s `--config` path
already uses (`load_config`, `build_run_config`, `Evaluator`, `Run`,
`load_target`), sampling `GenerationEvent.best_frame` at a fixed interval
instead of one frame per generation, then assembles the sampled frames into
an animated GIF with Pillow. Never touches `tp2/engine/*` or `tp2/cli.py`.

Runnable directly (`python scripts/make_gifs.py`), not only via `python -m`:
put the project root on sys.path before importing tp2, the same convention
`scripts/generate_plots.py` uses.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tp2.engine.config import build_run_config, load_config  # noqa: E402
from tp2.engine.fitness import Evaluator  # noqa: E402
from tp2.engine.loop import Run  # noqa: E402
from tp2.io.images import load_target  # noqa: E402

CANVAS = (128, 128)
TRIANGLES = 20
TARGET_FRAME_COUNT = 60
GIF_SCALE = 2
GIF_FRAME_DURATION_MS = 80
GIFS_DIR = PROJECT_ROOT / "plots" / "gifs"


@dataclass(frozen=True, slots=True)
class GifSpec:
    name: str
    image: str
    config: str
    seed: int
    caption: str


RUN_SPECS: list[GifSpec] = [
    GifSpec(
        "flag_ar_elite.gif",
        "assets/flag_ar.png",
        "configs/baseline.json",
        1,
        "Bandera Argentina bajo selección elite (configs/baseline.json)",
    ),
    GifSpec(
        "pictogram_elite.gif",
        "assets/pictogram.png",
        "configs/baseline.json",
        1,
        "Pictograma bajo el mismo operador elite (configs/baseline.json)",
    ),
    GifSpec(
        "flag_ar_roulette.gif",
        "assets/flag_ar.png",
        "configs/roulette_demo.json",
        1,
        "Misma bandera bajo selección ruleta (configs/roulette_demo.json), "
        "para contrastar presión de selección contra flag_ar_elite.gif",
    ),
    GifSpec(
        "mona_lisa_elite.gif",
        "assets/gif_extra/mona_lisa.jpg",
        "configs/baseline.json",
        1,
        "Mona Lisa (Leonardo da Vinci, dominio público, vía Wikimedia Commons) bajo selección "
        "elite (configs/baseline.json) — un guiño a 'EvoLisa', el nombre del hill climber (1+1); "
        "objetivo fuera de la matriz formal de 75 corridas, presupuesto de 20 triángulos igual "
        "al resto de los GIFs, sin pretensión de fidelidad.",
    ),
    GifSpec(
        "girl_pearl_earring_elite.gif",
        "assets/gif_extra/girl_pearl_earring.jpg",
        "configs/baseline.json",
        1,
        "La joven de la perla (Johannes Vermeer, dominio público, vía Wikimedia Commons) bajo "
        "selección elite (configs/baseline.json) — segundo retrato, para variar la complejidad "
        "del objetivo; también fuera de la matriz formal de 75 corridas.",
    ),
]


def sampled_generations(horizon: int, target_frame_count: int) -> set[int]:
    """Pick which generations to capture so a GIF has at most ~target_frame_count frames.

    `step` floors to 1 when `horizon < target_frame_count`, so no generation
    is skipped for a short run; `horizon` is explicitly added in case it is
    not already a multiple of `step`.
    """
    step = max(1, horizon // target_frame_count)
    return set(range(0, horizon + 1, step)) | {horizon}


def capture_run(spec: GifSpec) -> list[np.ndarray]:
    """Drive a real GA run through the public engine API, sampling frames."""
    image_path = PROJECT_ROOT / spec.image
    config_path = PROJECT_ROOT / spec.config
    config_data = load_config(config_path)
    config = build_run_config(config_data)
    target = load_target(image_path, CANVAS)
    evaluator = Evaluator(target, CANVAS)
    rng = np.random.default_rng(spec.seed)
    run = Run(config, evaluator, TRIANGLES, rng)
    keep = sampled_generations(config.horizon, TARGET_FRAME_COUNT)

    started = time.perf_counter()
    frames: list[np.ndarray] = []
    generations_run = 0
    for event in run:
        generations_run = event.generation
        if event.generation in keep or event.stop_reason:
            frames.append(event.best_frame)
        if event.stop_reason:
            break
    elapsed = time.perf_counter() - started
    print(
        f"{spec.name}: captured {len(frames)} frames over {generations_run} generations "
        f"in {elapsed:.1f}s"
    )
    return frames


def frames_to_gif(frames: list[np.ndarray], path: Path) -> None:
    """Assemble sampled RGB uint8 frames into a nearest-neighbour-upscaled GIF."""
    images = [
        Image.fromarray(frame, "RGB").resize(
            (frame.shape[1] * GIF_SCALE, frame.shape[0] * GIF_SCALE), Image.Resampling.NEAREST
        )
        for frame in frames
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=GIF_FRAME_DURATION_MS,
        loop=0,
        optimize=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="*", default=None, help="only build the named GifSpec.name entries"
    )
    args = parser.parse_args(argv)

    specs = RUN_SPECS
    if args.only is not None:
        wanted = set(args.only)
        specs = [spec for spec in RUN_SPECS if spec.name in wanted]
        if not specs:
            parser.error(f"--only matched no GifSpec.name in {sorted(wanted)}")

    for spec in specs:
        frames = capture_run(spec)
        out_path = GIFS_DIR / spec.name
        frames_to_gif(frames, out_path)
        size_kb = out_path.stat().st_size / 1024
        print(f"{out_path} ({size_kb:.1f} KB, {len(frames)} frames)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
