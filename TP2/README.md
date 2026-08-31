# TP2 — Algoritmos Genéticos

This project implements the course genetic algorithm from scratch. It approximates
an RGB target with an ordered stack of translucent triangles. NumPy and Pillow are
used solely for arrays and image rasterization; no genetic-algorithm library is
used.

## Representation: image structures and genes

An individual is a fixed-length `float32` vector with `11 × budget` values. It is
viewed as a `(budget, 11)` table only when it needs to be read or rendered:

| Loci | Meaning | Stored range | Rendered form |
|---|---|---|---|
| 0–5 | `(x1,y1), (x2,y2), (x3,y3)` | `[-0.1, 1.1]` | fractions of canvas width/height |
| 6–8 | red, green, blue | `[0, 1]` | 0–255 RGB |
| 9 | alpha | `[0.1, 0.9]` | 0–255 opacity |
| 10 | active flag | `[0, 1]` | active when `>= 0.5` |

So, a *gene* is one scalar allele; a triangle is an 11-gene block; and an
individual is a chromosome of `budget` blocks. The chromosome length is fixed so
one-point, two-point, uniform, and ring crossover all remain well defined. The
active flag gives a variable **effective** triangle count without requiring
unequal-length parents. Triangles are drawn in chromosome order over an opaque
white background, so later translucent triangles blend over earlier ones.

All coordinates are normalized. The small overshoot band lets triangles cover
canvas edges, and means changing `--canvas` needs no genome conversion. At the
disk boundary, the vector is decoded into named triangles in `triangles.json`, so
the output stays readable.

Current fitness is `max(1 - sqrt(SSE / max_SSE), 1e-12)`: higher is better, an
exact image is `1.0`, and the positive floor keeps roulette selection valid.

## Run locally (macOS, Linux, Windows/WSL)

Run these commands from `TP2/`; no absolute user paths are required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/make_assets.py
.venv/bin/python -m tp2 \
  --image assets/flag_ar.png --triangles 30 --population 8 \
  --canvas 128 --seed 42 --out runs/first-run
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` and replace
`.venv/bin/python` with `.venv\Scripts\python.exe`. The CLI prints the output
directory and writes `best.png`, `triangles.json`, `run.json`, and `metrics.csv`.
It will not overwrite an existing non-empty run directory unless `--force` is
given.

## Run in Google Colab

Use a notebook cell (replace the repository URL with your fork if needed):

```python
!git clone https://github.com/<owner>/SIA-G3.git
%cd SIA-G3/TP2
!python -m pip install -r requirements.txt
!python scripts/make_assets.py
!python -m tp2 --image assets/flag_ar.png --triangles 30 --population 8 --seed 42 --out runs/colab-run
```

Download `runs/colab-run/` when the run finishes. The engine is headless; a
display and a continuously powered personal laptop are not required.

## Current status

## Evolve the flag, with live notebook progress

The baseline loop uses cátedra-style elite parent/replacement selection,
one-point crossover, per-gene mutation, additive survival, and a generation
cap. It records one CSV row per generation:

```bash
.venv/bin/python -m tp2 \
  --image assets/flag_ar.png --triangles 80 --canvas 128 --seed 42 \
  --config configs/baseline.json --out runs/flag-evolution
```

For a live inline image in **Colab/Jupyter**, run the CLI in the *notebook
kernel* (not with `!python`, which starts a separate process that cannot redraw
the current output cell):

```python
from tp2.cli import main

main([
    "--image", "assets/flag_ar.png",
    "--triangles", "80",
    "--canvas", "128",
    "--seed", "42",
    "--config", "configs/baseline.json",
    "--notebook", "--notebook-every", "2",
    "--out", "runs/flag-live",
])
```

The cell refreshes with the current best image, generation, fitness, and render
count. `--notebook-every 5` is useful for longer runs.

Current implementation includes the first multi-generation operator slice;
the remaining operator variants and a standalone browser viewer follow in later
milestones.
