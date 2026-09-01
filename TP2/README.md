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

## Configuration (JSON Hyperparameters)

A config file is passed via `--config path/to/file.json`. Where a CLI flag and a JSON key
name the same thing (`--population`/`population`, `--canvas`/`canvas`), the CLI flag wins
when both are present. `--triangles` has no JSON equivalent — the triangle budget
(`triangle_budget`) is CLI-only.

**Top-level scalar keys:**

| Key | Type | Range | Meaning |
|---|---|---|---|
| `population` | int | 1–10000 | Population size `N` |
| `children` | int | 1–20000 | Number of children produced per generation, `K` |
| `canvas` | int | 8–1024 (pixels) | Square render side length |
| `recombination_probability` | float | 0.0–1.0 | Per-pairing crossover probability `Pc` (CATEDRA.md); if a pairing is not recombined, the children are identical copies of the parents but still go through mutation |
| `horizon` | int | ≥ 1 | Generation-count cap. Read by `stop.max_generations` as the cap it enforces, and independently by Michalewicz non-uniform mutation as the schedule's own progress denominator — used by both even when a different stop condition is what actually ends the run |

**`stop` object** — at least one condition must be enabled, or config validation rejects the
run outright as a run configured to never stop:

| Key | Type | Meaning |
|---|---|---|
| `max_generations` | bool | Enables the `horizon`-based generation cap. **Omitted from the config entirely, this defaults to `false`** — a config that leaves this key out does not get the generation cap for free; it must be stated explicitly as `true` to be active. |
| `wall_clock_seconds` | number | Stop once this many seconds have elapsed |
| `min_fitness` | number in (0, 1] | Stop once the best fitness reaches this value (STP-03, "acceptable solution") |
| `content_stagnation.window` / `.tolerance` | int / number | Stop once the best fitness has varied by less than `tolerance` over the last `window` generations (STP-04) |
| `structure_stagnation.window` / `.fraction` / `.tolerance` | int / number / number | Stop once at least `fraction` of the population has stayed "unchanged" (within `tolerance` genetic distance) for `window` consecutive generations (STP-05) |

When multiple stop conditions fire on the same generation, `max_generations` wins, then
`wall_clock`, then `min_fitness`, then `content_stagnation`, then `structure_stagnation` —
a fixed priority order, not an arbitrary one.

**`parents` / `replacement`** — both are selection slots and accept the identical shape;
the SAME set of methods is available in either slot (`replacement` is the new-generation
selector, and the cátedra allows it to reuse any of the parent-selection methods):

| Key | Applies to | Meaning |
|---|---|---|
| `method` | all | One of the 9 registered selection names below |
| `t0`, `tc`, `k` | `boltzmann` only | Initial temperature, final temperature, decay constant |
| `m` | `tournament_deterministic` only | Tournament size |
| `threshold` | `tournament_probabilistic` only | `Threshold ∈ [0.5, 1]` |
| `method_1`, `method_2`, `coefficient` | `blend` only, replaces `method` | Two nested selection specs mixed by `coefficient ∈ [0,1]` |

**`crossover`:**

| Key | Meaning |
|---|---|
| `method` | One of the 4 registered crossover names below |
| `boundary` | `"gene"` or `"triangle"` — where a cut may land: an individual gene locus, or only on an 11-gene triangle boundary. Applies to all four crossover methods. |
| `p` | `uniform` only — per-locus swap probability, default `0.5` |

**`mutation`:**

| Key | Meaning |
|---|---|
| `method` | One of the 4 registered mutation names below |
| `probability` | Mutation probability `Pm` |
| `schedule` | `"uniform"` or `"non_uniform"` (Michalewicz) — accepted by all four mutation methods |
| `b` | Michalewicz shape parameter — only meaningful under `schedule: "non_uniform"` |
| `m` | `multigen_limited` only — upper bound on the number of genes drawn per mutation |
| `sigma` | Optional object overriding the default per-gene-kind step size, with any of `coordinate` / `color` / `alpha` keys; a key left out of `sigma` keeps its built-in default rather than being treated as zero |

**`survival`:**

| Key | Meaning |
|---|---|
| `method` | One of the 3 registered survival names below |
| `gap` | `generational_gap` only — `G ∈ [0,1]`, meaningless for the other two methods |

### Registered Operator Names

The 20 names below (9 selection + 4 crossover + 4 mutation + 3 survival) are checked for
completeness against the live registry — every name the engine actually accepts is
documented here, and nothing here is a name the engine does not accept.

**Selection (9):**

- `elite` — Elite: sorted by fitness, individual at rank `i` is copied `n(i) = ceil((K-i)/N)` times; the best individuals are taken *multiple* times whenever `K > N` (not a plain "sort and take the top K").
- `random` — Muestreo Aleatorio: picks uniformly at random, ignoring fitness entirely.
- `blend` — Composes two selection methods by a coefficient `A` (`coefficient`); an approved optional extra beyond the enunciado's required set (SEL-09).
- `roulette` — Ruleta: `K` independent draws against the cumulative relative-fitness wheel.
- `universal` — Universal (SUS): the same cumulative wheel as roulette, but the `K` draws are stratified from a single random offset (`r_j = (r+j)/K`) instead of `K` independent draws.
- `ranking` — Ranking: pseudo-fitness `f'(i) = (N-rank(i))/N` by fitness rank, then roulette on the pseudo-fitness.
- `boltzmann` — Boltzmann (Entrópica): pseudo-fitness `ExpVal(i,g,T)` (an expected value against the generation's population average), then roulette; temperature follows `T(t) = Tc + (T0-Tc)·e^(-kt)`.
- `tournament_deterministic` — Torneo Determinístico: pick `M` individuals at random, keep the best, repeat `K` times.
- `tournament_probabilistic` — Torneo Probabilístico: pick 2 individuals at random, keep the fitter with probability `Threshold ∈ [0.5,1]` (otherwise keep the less fit), repeat `K` times.

**Crossover (4):** all four accept the shared `boundary` key — `"triangle"` cuts only on
11-gene triangle boundaries, `"gene"` cuts on individual gene loci.

- `one_point` — Un Punto: one random cut; everything from the cut onward is swapped between parents.
- `two_point` — Dos Puntos: two random cuts; the segment between them is swapped.
- `ring` — Anular: a random start position and length, wrapping around the end of the chromosome, defines the swapped segment.
- `uniform` — Uniforme: each locus is swapped independently with probability `p` (default 0.5) — per CATEDRA.md, the only crossover of the four that does not preserve positional correlation between alleles.

**Mutation (4):** all four accept the shared `schedule: uniform | non_uniform` (Michalewicz) key.

- `gene` — Gen: a single gene is altered, with probability `Pm`.
- `multigen_limited` — Multigen Limitada: a random count of genes in `[1, m]` is drawn to mutate, each with probability `Pm`.
- `multigen_uniform` — Multigen Uniforme: every gene independently has probability `Pm` of mutating.
- `complete` — Completa: with probability `Pm`, every gene of the individual mutates.

**Survival (3):**

- `additive` — Supervivencia Aditiva: select `N` individuals from the union of the `N` parents and `K` children.
- `exclusive` — Supervivencia Exclusiva: if `K > N`, select `N` from the `K` children only; if `K <= N`, take all `K` children plus `N-K` selected from the parents (branch boundary is strictly `K > N`, not `K >= N`).
- `generational_gap` — Brecha Generacional `G`: the new generation is `(1-G)*N` individuals carried from the previous generation plus `G*N` from the children.

9 (selection) + 4 (crossover) + 4 (mutation) + 3 (survival) = 20 names, checked against
`SELECTION.names() | CROSSOVER.names() | MUTATION.names() | SURVIVAL.names()` — not merely
asserted from memory.

