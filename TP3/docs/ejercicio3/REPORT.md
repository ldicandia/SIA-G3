# Ejercicio 3 — Accuracy Push (more_digits.csv)

## 1. Dataset Exploration & Comparison to digits.csv

### Dataset Composition & Dimensions

`data/data and documentation/more_digits.csv` (the dataset for this exercise) contains **15741** rows of the same 784-value (28x28) grayscale image vectors used in Ejercicio 2, per `docs/ejercicio3/dataset_stats.json`'s `more_digits_csv` block. `digits.csv` (Ejercicio 2's training pool) has 12449 rows, so `more_digits.csv` adds **3292** additional rows — a **+26.44%** increase in available training data (`row_count_delta_vs_digits_csv: 3292` in `dataset_stats.json`), while keeping the identical 784-dimensional, already-normalized `[0, 1]` image representation. No new feature engineering or additional preprocessing is required beyond what Ejercicio 2 already established.

### Class Distribution & the Digit-8/Digit-5 Gap Closing

The class-count comparison, cited directly from `docs/ejercicio3/dataset_stats.json`'s `more_digits_csv.label_counts` against `digits_csv_for_comparison.label_counts`:

| Digit Class | `digits.csv` (Ejercicio 2) | `more_digits.csv` (Ejercicio 3) | Change |
|:---:|:---:|:---:|:---:|
| 0 | 1,480 | 1,776 | +296 |
| 1 | 1,685 | 2,022 | +337 |
| 2 | 1,489 | 1,787 | +298 |
| 3 | 1,532 | 1,839 | +307 |
| 4 | 1,460 | 1,752 | +292 |
| 5 | **271** | **542** | +271 |
| 6 | 1,479 | 1,775 | +296 |
| 7 | 1,566 | 1,879 | +313 |
| 8 | **0** | **585** | +585 |
| 9 | 1,487 | 1,784 | +297 |

Two findings stand out, both cited plainly from the exact counts above (never qualitative placeholders):

1. **Digit 8 rises from 0 training rows in `digits.csv` to 585 rows in `more_digits.csv`.** Ejercicio 2's own final production check (`docs/ejercicio2/final_test_metrics.json`) reported a digit-8 recall of exactly **0.0** on `digits_test.csv` — a structural blindness caused entirely by digit 8 having zero training examples, not a modeling failure. `more_digits.csv` closes that structural gap: any model trained on it now has 585 digit-8 examples to learn from.
2. **Digit 5 rises from 271 rows (2.18% of `digits.csv`) to 542 rows in `more_digits.csv`.** It remains the smallest represented class (all other classes now range 1,752-2,022 rows), but its absolute count has doubled.

### Locked Three-Way Split

Following Plan 06-01's split methodology (`scripts/ejercicio2/digits_preprocess.py`'s `stratified_split_indices`), this plan extends it to a three-way partition via `scripts/ejercicio3/more_digits_preprocess.py`'s `stratified_three_way_split_indices`: each label's rows are grouped, shuffled independently with `random.Random(seed=42)`, and split into a 20% heldout portion, a 20% validation portion, and the ~60% remainder as the training portion — a **60/20/20 stratified split, seed 42**. This carves out a heldout third **in place of** the separate `digits_test.csv` file Ejercicio 2 had — `more_digits.csv` ships without its own held-out test file, so Ejercicio 3 must produce one itself.

The resulting partition (`docs/ejercicio3/dataset_stats.json`):

| Split | Rows | Share |
|---|:---:|:---:|
| `internal_train_split` | 9,447 | 60.03% |
| `internal_val_split` | 3,147 | 19.99% |
| `heldout_split` | 3,147 | 19.99% |
| **Total** | **15,741** | **100.00%** |

The three splits are a verified disjoint, complete partition of `more_digits.csv`'s 15741 rows (`tests/test_more_digits_preprocess.py`). Per this phase's reversibility/prohibition contract, `data/derived/more_digits/heldout.csv` is written here but is **not** opened again by this plan's baseline script, by any Plan 07-02/07-03 technique sweep, or by any other consumer — it is reserved exclusively for Plan 07-04's single ACC-01 close-out check, mirroring Ejercicio 2's exactly-once `digits_test.csv` discipline.

---

## 2. "More Data" Baseline (ACC-02)

ACC-02 requires the "more data" effect to be isolated **first**, before any technique change is attempted, using a configuration that is provably identical to Ejercicio 2's own winner — not a fresh guess or a retyped approximation.

### Loaded, Not Retyped

`scripts/ejercicio3/more_digits_baseline.py`'s `load_ejercicio2_best_config` reads `docs/ejercicio2/variant_comparison.json`'s `selected` object programmatically and returns it verbatim. The baseline training run in this plan used exactly that dict's `layer_sizes`/`learning_rate`/`optimizer`/`activation`/`epochs` values, unmodified:

- **Architecture:** `[784, 32, 16, 10]`
- **Learning rate:** `0.01`
- **Optimizer:** `sgd`
- **Activation:** `sigmoid`
- **Epochs:** `10`

This is the exact combined-best configuration Ejercicio 2 selected in Plan 06-04 via its `select_best` deterministic tie-break rule (fewer params, then lower learning rate, then alphabetically-first optimizer). Because the config is loaded from `docs/ejercicio2/variant_comparison.json` at runtime rather than hand-copied into this plan's code, "unmodified" is a provable, testable fact (`tests/test_more_digits_baseline_e2e.py`), not merely a claim.

### Result: More Data Alone, No Technique Change

The unmodified config was re-run completely on `more_digits.csv`'s own internal `train.csv`/`val.csv` split (never `heldout.csv`) via `scripts/ejercicio2/digits_train_variant.py`'s `train_and_harvest` — the same zero-additional-epoch harvest mechanism Ejercicio 2 already proved. `docs/ejercicio3/more_data_baseline.json` records:

| Metric | Ejercicio 2 (`digits.csv`, internal val split) | Ejercicio 3 "more data" baseline (`more_digits.csv`, internal val split) |
|---|:---:|:---:|
| Val accuracy | **0.9313** (93.13%) | **0.9161** (91.61%) |
| Digit 5 recall | 0.5741 | 0.6204 |
| Digit 8 recall | `null` (0 rows) | 0.8376 |

**`delta_val_accuracy = -0.0152`** (a -1.52 percentage-point change). Reported honestly, with the real number, regardless of direction: the more-data-only baseline, using the exact same architecture/hyperparameters as Ejercicio 2's combined-best config, did **not** improve overall validation accuracy over Ejercicio 2's own result — it is a small regression on the aggregate metric. This is not reframed as a success; the >=98% target from the enunciado is far from met by this baseline alone.

At the same time, per-class recall tells a more nuanced story consistent with Section 1's exploration: digit 8, structurally unlearnable in Ejercicio 2 (recall undefined, 0 training rows), now recalls **83.76%** of its held-out validation rows, and digit 5's recall improved from 57.41% to 62.04% with its doubled training count. The aggregate accuracy dip appears concentrated elsewhere (e.g. digit 7 dropped from 0.9617 to 0.8963, digit 9 from 0.8855 to 0.8964) — a possible side-effect of the larger, differently-composed dataset interacting with a config that was tuned specifically for `digits.csv`'s internal split, not `more_digits.csv`'s. This "more data alone is not automatically better" finding is exactly what ACC-02's isolation step is designed to surface before any technique change (Plan 07-02 onward) is layered on top and potentially confounded with it.
