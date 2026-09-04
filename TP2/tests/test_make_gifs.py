from __future__ import annotations

import json

import numpy as np
from PIL import Image

from scripts.make_gifs import GIF_SCALE, GifSpec, capture_run, frames_to_gif, sampled_generations


def test_sampled_generations_lands_exactly_on_horizon() -> None:
    result = sampled_generations(horizon=3000, target_frame_count=60)
    assert 0 in result
    assert 3000 in result
    assert len(result) <= 61


def test_sampled_generations_short_horizon_skips_nothing() -> None:
    result = sampled_generations(horizon=5, target_frame_count=60)
    assert result == {0, 1, 2, 3, 4, 5}


def test_frames_to_gif_writes_readable_upscaled_gif(tmp_path) -> None:
    # Distinct per-frame colors: Pillow's GIF encoder collapses consecutive
    # pixel-identical frames even with optimize=False, so identical frames
    # would silently produce n_frames == 1 instead of exercising the
    # multi-frame write path this test is actually checking.
    frames = [np.full((8, 8, 3), fill_value, dtype=np.uint8) for fill_value in (0, 128, 255)]
    out_path = tmp_path / "test.gif"
    frames_to_gif(frames, out_path)

    with Image.open(out_path) as gif:
        assert gif.n_frames == 3
        assert gif.size == (8 * GIF_SCALE, 8 * GIF_SCALE)


def test_capture_run_drives_real_engine_end_to_end(tmp_path) -> None:
    config_path = tmp_path / "tiny.json"
    config_path.write_text(
        json.dumps(
            {
                "population": 4,
                "children": 4,
                "horizon": 5,
                "recombination_probability": 0.8,
                "parents": {"method": "elite"},
                "replacement": {"method": "elite"},
                "crossover": {"method": "one_point", "boundary": "triangle"},
                "mutation": {"method": "gene", "probability": 0.9},
                "survival": {"method": "additive"},
                "stop": {"max_generations": True},
            }
        ),
        encoding="utf-8",
    )
    spec = GifSpec(
        name="tiny.gif",
        image="assets/flag_ar.png",
        config=str(config_path),
        seed=1,
        caption="tiny smoke run",
    )

    frames = capture_run(spec)

    assert len(frames) >= 2
    for frame in frames:
        assert frame.shape == (128, 128, 3)
        assert frame.dtype == np.uint8
