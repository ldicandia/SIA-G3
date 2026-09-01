# SIA TP2 — Algoritmos Genéticos

A hand-written genetic algorithm engine — no genetic-algorithm library anywhere near it,
per the enunciado's explicit prohibition — that approximates a target image with translucent
triangles on a blank canvas. Built for ITBA "Sistemas de Inteligencia Artificial" TP2,
Ejercicio 2 ("un compresor de imágenes un tanto peculiar"). Given a target image and a
triangle budget, the engine evolves a population of triangle sets, generation after
generation, until the rendered result resembles the original. The complete operator matrix
the cátedra mandates — all nine selection methods, all four crossover methods, all four
mutation scopes and all three survival strategies — is selectable from a single JSON
configuration file, with no code changes required to switch between them.

The deliverable is not "a renderer that happens to work" — it is a defensible, measured
comparison. Every operator choice documented below is backed by an experiment matrix and a
set of comparative figures built to defend that choice in front of the cátedra, not merely
a working image approximation.

## Installation

Python runs through a **WSL** virtual environment, never a Windows-side one — a Windows venv
is known broken for this project's dependencies (the same constraint TP1 in this repo hit
first). Open a WSL/Linux shell inside the cloned `TP2/` directory, or prefix every command
below with:

```
wsl.exe bash -lc 'cd /path/to/TP2 && <command>'
```

(replace `/path/to/TP2` with wherever you cloned this repo — do not copy a path from someone
else's machine). Every command in this README assumes you are inside that shell, in the
`TP2/` directory, and uses the `.venv/bin/python` form consistently rather than an activated
shell — the two are equivalent, but this README sticks to one so every command below is
copy-pasteable verbatim.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins five packages: **Pillow** (`==12.3.0`, pinned to an exact version
because a few of this project's must-haves are byte-reproducibility claims that would drift
across Pillow releases), **numpy** (`>=1.26,<3`; verified against `2.5.2`), **pygame-ce**
(`>=2.5,<3`; verified against `2.5.8` — used only by the optional live viewer), **matplotlib**
(`>=3.8,<4`; verified against `3.11.1` — used only by the experiment plots), and **pytest**
(`>=8,<10`; verified against `9.1.1`). Note the package name: it is `pygame-ce`, not `pygame`
— both install a module literally named `pygame`, so having both installed at once is a
broken environment; install only what `requirements.txt` names.

## Running the Engine

```bash
.venv/bin/python -m tp2 --image assets/flag_ar.png --triangles 30 --config configs/baseline.json --seed 42 --out runs/demo
```

`--image` is always required. `--triangles` (the chromosome's triangle budget) and
`--config` (the JSON hyperparameter file — see [Configuration](#configuration-json-hyperparameters)
below) are the two flags you will always want to set explicitly in practice;
`configs/baseline.json` is the shipped starting point and a reasonable default for a first
run. A successful run writes four artifacts under a fresh directory inside `runs/`:

- `best.png` — the rendered best individual found, as a PNG.
- `triangles.json` — the enumeration of that individual's triangles (position, colour, and
  active flag per triangle).
- `run.json` — the effective configuration actually used (including the resolved seed),
  library versions, the git commit, and the reason the run stopped.
- `metrics.csv` — one row per generation: fitness, error, generation number, render count,
  and elapsed time.

**Omitting `--seed`** still produces a fully reproducible run: a seed is drawn at random and
archived verbatim in `run.json`, so passing that same value back in with `--seed` replays the
exact same run. **Omitting `--out`** does not skip writing output — it defaults to a
timestamped directory created fresh under `runs/`. **Re-running into the SAME `--out`
directory without `--force`** does not silently overwrite what is there: it raises an error
naming the colliding path. Passing `--force` is the explicit way to opt into overwriting an
existing run directory.

## Running the Viewer

```bash
.venv/bin/python -m tp2 --image assets/flag_ar.png --triangles 30 --config configs/baseline.json --seed 42 --out runs/demo_viewer --viewer
```

Same command as above, with `--viewer` appended — the engine itself never imports `pygame`
and runs identically either way; `--viewer` only adds a live display on top. This opens a
window showing the best individual evolving generation by generation, with generation
number, fitness, and render count overlaid. `--viewer-scale` (default 4) controls the
window's pixel scale factor, and `--viewer-every` (default 1) controls how often the display
redraws (every Nth generation). Closing the viewer window ends the run cleanly rather than
crashing — the run's artifacts are still written, and `run.json`'s `stop_reason` records
`viewer_closed` so it is distinguishable from a run that reached its own stop condition
naturally.

## Running Experiments

```bash
.venv/bin/python -m tp2.experiments.runner --spec configs/experiments/main_matrix.json --out runs/matrix --jobs 8
```

This is a **long-running command** — multiple minutes on a multi-core machine, not a quick
check. The shipped matrix spec is 15 cells (7 selection methods, 6 survival/`K:N` ratio
combinations, and 2 crossover-honesty controls) times 5 seeds each = 75 independent runs,
distributed across a process pool sized by `--jobs` (defaults to `min(cpu_count, 16)`). Do
not expect this to finish in the time it takes to read this README.

Once a matrix run has completed, turn it into figures with a second, much smaller command:

```bash
.venv/bin/python scripts/generate_plots.py
```

This reads the `metrics.csv` / `run.json` files `runs/matrix/` and `runs/hillclimber/`
already contain and writes five comparative figures under `plots/` — it never re-runs the
GA, so it can be re-run any time (after tweaking a plot's styling, for example) at
effectively zero cost.

## Running the Hill-Climber Baseline

```bash
.venv/bin/python -m tp2.baselines.hillclimber --image assets/flag_ar.png --triangles 30 --config configs/baseline.json --seed 1 --out runs/hillclimber
```

This is a `(1+1)` stochastic hill climber — population size one, mutate-and-accept-if-better,
no crossover, no selection pressure over a population — used only as an honest,
equal-render-budget comparison point in the presentation. It is **never** "the baseline GA":
it is not a genetic algorithm at all, and calling it one misrepresents the comparison it
exists to make.

## Testing

```bash
.venv/bin/python -m pytest -q
```

Runs the full suite from the WSL venv. The suite includes one slow convergence-gate test
(seeded 3-run gate checking real convergence above a fitness threshold); exclude it with
`-m "not slow"` for a fast pass during normal development.

## Shipped Targets

Three target images ship under `assets/`: `flag_ar.png` (a flag), `silhouette.png`, and
`pictogram.png` — all deliberately simple, per the enunciado's own advice to start with
flags, silhouettes, pictograms, and similarly simple shapes rather than photographs. They
are generated by `scripts/make_assets.py` from committed drawing primitives, not sourced
from anywhere external.

