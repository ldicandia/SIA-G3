"""Long-running high-fidelity Mona Lisa approximation.

Runs an optimization loop using the project's Evaluator, target image loader,
and rasterizer, designed to run for up to ~20 minutes (or target fitness 0.996)
to achieve portrait-level detail. Captures periodic checkpoints and builds
an animated GIF upon completion.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tp2.engine.fitness import Evaluator  # noqa: E402
from tp2.engine.genome import (  # noqa: E402
    A,
    ACTIVE,
    B,
    G,
    GENES_PER_TRIANGLE,
    R,
    bounds_for,
    random_population,
    reflect,
)
from tp2.io.images import load_target, save_png  # noqa: E402

CANVAS = (128, 128)
IMAGE_PATH = PROJECT_ROOT / "assets" / "gif_extra" / "mona_lisa.jpg"
OUTPUT_DIR = PROJECT_ROOT / "plots" / "long_run_mona_lisa"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=1200.0, help="maximum duration in seconds (default 1200s = 20m)")
    parser.add_argument("--triangles", type=int, default=100, help="number of triangles (default 100)")
    parser.add_argument("--target-fitness", type=float, default=0.9965, help="target fitness to stop early")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="output directory")
    args = parser.parse_args(argv)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    target = load_target(IMAGE_PATH, CANVAS)
    evaluator = Evaluator(target, CANVAS)

    budget = args.triangles
    bounds = bounds_for(budget)
    genes = random_population(rng, 1, budget)[0]
    best_fitness, best_frame = evaluator.evaluate(genes)

    print(f"=== Iniciando corrida extendida de Mona Lisa ===", flush=True)
    print(f"Presupuesto: {budget} triángulos | Duración máx: {args.seconds:.0f}s ({args.seconds/60:.1f} min)", flush=True)
    print(f"Fitness inicial: {best_fitness:.6f}", flush=True)
    print(f"Directorio de salida: {out_dir}", flush=True)

    # Save initial checkpoint
    initial_img = Image.fromarray(best_frame, "RGB").resize((256, 256), Image.Resampling.NEAREST)
    initial_img.save(checkpoints_dir / "cp_00_inicio.png")

    gif_frames: list[np.ndarray] = [best_frame]
    metrics_rows: list[tuple[float, int, int, float]] = [(0.0, 0, 1, best_fitness)]

    # Milestones for checkpoints: 25%, 50%, 75%, 90%
    milestones = {
        0.25: ("cp_25_pct.png", False),
        0.50: ("cp_50_pct.png", False),
        0.75: ("cp_75_pct.png", False),
        0.90: ("cp_90_pct.png", False),
    }

    start_time = time.perf_counter()
    last_log_time = start_time
    last_gif_sample = start_time
    renders = 1
    accepted = 0

    while True:
        now = time.perf_counter()
        elapsed = now - start_time
        progress_pct = min(1.0, elapsed / args.seconds)

        if elapsed >= args.seconds or best_fitness >= args.target_fitness:
            break

        # Checkpoint milestones by time progress
        for fraction, (fname, reached) in milestones.items():
            if not reached and progress_pct >= fraction:
                img = Image.fromarray(best_frame, "RGB").resize((256, 256), Image.Resampling.NEAREST)
                img.save(checkpoints_dir / fname)
                milestones[fraction] = (fname, True)
                print(f"[CHECKPOINT {int(fraction*100)}%] Guardado {fname} en {elapsed:.1f}s (Fitness: {best_fitness:.6f})", flush=True)

        # Periodic log every 20 seconds
        if now - last_log_time >= 20.0:
            speed = renders / max(1e-6, elapsed)
            print(
                f"[{elapsed:6.1f}s / {args.seconds:.0f}s ({progress_pct*100:4.1f}%)] "
                f"Renders: {renders:,} ({speed:.0f}/s) | Aceptados: {accepted:,} | Fitness: {best_fitness:.6f}",
                flush=True,
            )
            last_log_time = now
            metrics_rows.append((elapsed, accepted, renders, best_fitness))

        # Sample frame for GIF every ~15 seconds
        if now - last_gif_sample >= 15.0:
            gif_frames.append(best_frame)
            last_gif_sample = now

        # Adaptive mutation
        if best_fitness < 0.85:
            coord_sigma = 0.05
            color_sigma = 0.08
            alpha_sigma = 0.06
        elif best_fitness < 0.93:
            coord_sigma = 0.03
            color_sigma = 0.05
            alpha_sigma = 0.04
        else:
            coord_sigma = 0.015
            color_sigma = 0.03
            alpha_sigma = 0.025

        roll = rng.random()
        mutant = genes.copy()

        if roll < 0.05 and budget > 1:
            t1 = rng.integers(0, budget)
            t2 = rng.integers(0, budget)
            if t1 != t2:
                s1 = t1 * GENES_PER_TRIANGLE
                s2 = t2 * GENES_PER_TRIANGLE
                mutant[s1 : s1 + 11], mutant[s2 : s2 + 11] = (
                    genes[s2 : s2 + 11].copy(),
                    genes[s1 : s1 + 11].copy(),
                )
        else:
            count = 1 if roll < 0.85 else rng.integers(1, 3)
            tri_indices = rng.choice(budget, size=count, replace=False)
            for t_idx in tri_indices:
                start = t_idx * GENES_PER_TRIANGLE
                noise = np.array(
                    [
                        rng.normal(0, coord_sigma), rng.normal(0, coord_sigma),
                        rng.normal(0, coord_sigma), rng.normal(0, coord_sigma),
                        rng.normal(0, coord_sigma), rng.normal(0, coord_sigma),
                        rng.normal(0, color_sigma), rng.normal(0, color_sigma), rng.normal(0, color_sigma),
                        rng.normal(0, alpha_sigma),
                        0.0,
                    ],
                    dtype=np.float32,
                )
                sub_bounds = bounds[start : start + 11]
                mutant[start : start + 11] = reflect(mutant[start : start + 11] + noise, sub_bounds[:, 0], sub_bounds[:, 1])

        candidate_fitness, candidate_frame = evaluator.evaluate(mutant)
        renders += 1

        if candidate_fitness > best_fitness:
            genes = mutant
            best_fitness = candidate_fitness
            best_frame = candidate_frame
            accepted += 1

    total_time = time.perf_counter() - start_time
    gif_frames.append(best_frame)
    metrics_rows.append((total_time, accepted, renders, best_fitness))

    print(f"\n=== Corrida completada en {total_time:.1f}s ===", flush=True)
    print(f"Total renders evaluados: {renders:,} ({renders/total_time:.0f}/s)", flush=True)
    print(f"Mutaciones aceptadas: {accepted:,}", flush=True)
    print(f"Fitness final alcanzado: {best_fitness:.6f}", flush=True)

    final_img_256 = Image.fromarray(best_frame, "RGB").resize((256, 256), Image.Resampling.NEAREST)
    final_img_256.save(out_dir / "mona_lisa_final_256.png")
    final_img_512 = Image.fromarray(best_frame, "RGB").resize((512, 512), Image.Resampling.NEAREST)
    final_img_512.save(out_dir / "mona_lisa_final_512.png")
    final_img_256.save(checkpoints_dir / "cp_100_final.png")
    print(f"Imagen final guardada en: {out_dir / 'mona_lisa_final_512.png'}", flush=True)

    with open(out_dir / "metrics.csv", "w", encoding="utf-8") as f:
        f.write("time_seconds,accepted,renders,fitness\n")
        for row in metrics_rows:
            f.write(f"{row[0]:.2f},{row[1]},{row[2]},{row[3]:.6f}\n")

    print(f"Generando GIF animado con {len(gif_frames)} cuadros...", flush=True)
    gif_images = [
        Image.fromarray(f, "RGB").resize((256, 256), Image.Resampling.NEAREST)
        for f in gif_frames
    ]
    gif_images[0].save(
        out_dir / "mona_lisa_extended.gif",
        format="GIF",
        save_all=True,
        append_images=gif_images[1:],
        duration=100,
        loop=0,
        optimize=False,
    )
    print(f"GIF guardado en: {out_dir / 'mona_lisa_extended.gif'}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
