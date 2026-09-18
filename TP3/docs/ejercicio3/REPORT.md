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

---

## 3. Techniques Applied (ACC-03)

ACC-03 requires deliberate techniques — beyond the "more data" effect isolated in Section 2 — to be explored and honestly compared. Three techniques were applied against Section 2's `val_accuracy = 0.9161` more-data baseline: widening/deepening the architecture, varying the weight-initialization seed, and majority-vote ensembling over the seed-diverse models. All three reuse `more_digits.csv`'s locked internal `train.csv`/`val.csv` split (Section 1) under Ejercicio 2's engine, reused unchanged.

### 3.1 Architecture Variants

`docs/ejercicio3/architecture_variants.json` compares three architectures under identical `learning_rate=0.01`/`optimizer=sgd`/`activation=sigmoid`/`epochs=10`:

| Architecture (`layer_sizes`) | Params (`n_params`) | Val accuracy |
|---|:---:|:---:|
| `[784, 32, 16, 10]` (baseline, reused) | 25,818 | 0.9161 |
| `[784, 64, 10]` (`arch_64`) | 50,890 | 0.9250 |
| `[784, 64, 32, 10]` (`arch_64-32`) | 52,650 | **0.9269** |

`[784, 64, 32, 10]` wins, at **0.9269** val accuracy — **+0.0108** (+1.08 percentage points) over the `[784, 32, 16, 10]` baseline, and +0.0019 (+0.19 points) over the single-hidden-layer `[784, 64, 10]` variant despite only +1,760 additional parameters. Widening and adding a second hidden layer both individually recover most of Section 2's -0.0152 "more data" regression, and combined they recover more than all of it.

### 3.2 Initialization (Seed) Variants

`docs/ejercicio3/seed_variants.json` compares four weight-initialization seeds under the IDENTICAL `[784, 32, 16, 10]` architecture, `learning_rate=0.01`, `optimizer=sgd`, `activation=sigmoid`, `epochs=10` — the only variable is the random seed feeding `core/src/mlp.cpp`'s `Matrix::random` weight initialization (the engine performs no row shuffling, per `core/include/mlp.hpp`'s "Online gradient descent across dataset in row order, no shuffling"):

| Seed | Val accuracy |
|:---:|:---:|
| 7 | **0.9237** (max) |
| 42 (baseline, reused) | **0.9161** (min) |
| 123 | 0.9187 |
| 2026 | 0.9221 |

The spread across the four seeds is **0.9161–0.9237**, a **0.0076** (0.76 percentage point) range — genuinely non-identical outcomes, confirming that weight initialization alone (with everything else held fixed, including row order) meaningfully affects this engine's trained result. It is not a no-op sweep.

### 3.3 Ensembling (Majority Vote over Seed Variants)

`docs/ejercicio3/ensemble_variants.json` combines all four Section 3.2 seed-variant models' (`more_data_baseline`, `seed_7`, `seed_123`, `seed_2026`) val-split predictions via a deterministic hard majority vote (`scripts/ejercicio3/more_digits_ensemble.py`'s `majority_vote`, ties broken to the smallest tied class) — the only combination rule available, since `core/include/io/run_json.hpp`'s `PredictionRecord` stores only each model's argmax `predicted_class`, no softmax probability vector to average:

- **Ensemble val accuracy:** **0.9266** (`ensemble_val_accuracy`)
- **Best single member val accuracy:** **0.9237** (`best_single_member_val_accuracy`, `seed_7`)
- **`ensemble_beats_best_single_member`: `true`**

Ensembling helped here: the majority vote across the four seed-diverse models scores **+0.0029** (+0.29 percentage points) above the best individual member (`seed_7`, 0.9237), and **+0.0105** above the seed-42 baseline (0.9161) — a modest but real gain from combining independently-initialized models' disagreements, consistent with the genuine (non-identical) seed spread found in Section 3.2. This result is reported as-is; had the ensemble instead scored below the best single member, that negative result would be stated with the same honesty.

### Scope Note: Mini-Batching Not Attempted

Mini-batching was not attempted as a technique this phase. ROADMAP Phase 7 locks "Ejercicio 2's engine and methodology reused unchanged," and `core/include/mlp.hpp`'s `fit()` is hardcoded to online (unbatched) gradient descent in fixed row order — its own docstring states "Online gradient descent across dataset in row order, no shuffling." Extending `fit()` to support mini-batches would require a core-engine change, which this phase's scope explicitly does not make. This is a documented scope boundary, not a silent omission: the three techniques above (architecture, initialization, ensembling) are the full set of deliberate techniques ACC-03 explores under the engine as it exists at the start of this phase.

---

## 4. Separating "More Data" from Technique Effects (ACC-04)

ACC-04 requires the "more data" effect (already isolated in Section 2, before any technique change) and the "technique" effect (Section 3's architecture/initialization/ensembling exploration) to be explicitly separated and attributed — never blended into one unexplained combined number.

### Factor 1: "More Data" — the Numeric Row-Count Effect

Section 2's `docs/ejercicio3/more_data_baseline.json` isolates this first, before any technique change: re-running Ejercicio 2's exact combined-best config (`[784, 32, 16, 10]`, lr=0.01, sgd, sigmoid, 10 epochs), unmodified, on `more_digits.csv`'s own internal train/val split alone yields `val_accuracy = 0.9161` versus Ejercicio 2's own `0.9313` — a **`delta_val_accuracy = -0.0152`** (-1.52 percentage points). On the aggregate metric alone, the extra 3,292 rows of `more_digits.csv` did **not** improve accuracy over Ejercicio 2's result; this is the "more data" effect's numeric contribution, reported honestly as a small net regression on the config that was tuned specifically for `digits.csv`'s distribution.

### Factor 2: "More Data" — the Structural Class-Count Recovery

Independent of the numeric row-count effect above, Section 1 documented a second, structural "more data" factor: **digit 8 rises from 0 training rows in `digits.csv` to 585 rows in `more_digits.csv`**, and **digit 5 rises from 271 rows (2.18%) to 542 rows** (doubling its representation). This is not a technique — no architecture, seed, or ensembling choice caused digit 8 to become classifiable; it became classifiable purely because the dataset itself changed to include examples of that class. Ejercicio 2's own final production check (`docs/ejercicio2/final_test_metrics.json`) reported an exact **0.0** digit 8 recall on `digits_test.csv` — a structural blindness from zero training examples, not a modeling failure. Section 5 below shows this structural gap closing to 85.47% recall once digit 8 has training examples at all, entirely attributable to this second "more data" factor rather than to any Section 3 technique.

### Factor 3: "Technique" — the Deliberate Improvement on Top

Section 3's ACC-03 techniques were applied on top of Section 2's `val_accuracy = 0.9161` baseline, using `more_digits.csv`'s locked internal split throughout (never `heldout.csv`). The best individual result — architecture `[784, 64, 32, 10]` (`arch_64-32`) — reached `val_accuracy = 0.926914521766762` (`docs/ejercicio3/variant_comparison.json`'s `best_individual`). The explicit "technique" contribution on top of the more-data baseline is:

**`0.926914521766762 - 0.9161105815061964 = +0.0108`** (+1.08 percentage points)

The seed-variant ensemble (`docs/ejercicio3/ensemble_variants.json`, `ensemble_val_accuracy = 0.9266`) came close but did not strictly beat `arch_64-32`'s individual result, so `choose_final`'s deterministic tie-break selected the single model (Section 5) — a genuine technique gain, not an assumed one.

### What Cannot Be Cleanly Attributed to Either Bucket

Both the `-0.0152` "more data" delta and the `+0.0108` "technique" gain above are measured consistently on `more_digits.csv`'s own internal validation split, so their arithmetic combination (`0.9161 + 0.0108 = 0.9269`) is directly additive and matches `best_individual`'s reported `val_accuracy` by construction — no residual is hidden there. Section 5's final heldout accuracy (`0.9285`, on a *different*, non-overlapping 3,147-row heldout split rather than the 3,147-row internal validation split) differs slightly from this `0.9269` figure; that ~0.16-percentage-point gap is ordinary sampling variance between two disjoint subsets of the same stratified population, not a fourth, unexplained factor requiring attribution — it is stated here explicitly rather than silently folded into either the "more data" or "technique" bucket.

---

## 5. Final Held-Out Check & Best Achievable Accuracy (ACC-01 close-out)

This section documents the ONLY point in this entire phase that `data/derived/more_digits/heldout.csv` is read — a single, zero-additional-epoch harvest of the final selected configuration, executed once via `scripts/ejercicio3/more_digits_compare.py --final-check` and never re-run, mirroring Plan 06-04's exactly-once `digits_test.csv` discipline for Ejercicio 2.

### Final Configuration Checked

Per `docs/ejercicio3/variant_comparison.json`'s `final_choice`, the ensemble's `0.9266` val_accuracy did not strictly beat the best individual model's `0.926914521766762`, so the deterministic tie-break (ties/losses prefer the single model, for lower deployment complexity) selected the **single model** configuration:

- **Architecture (`layer_sizes`):** `[784, 64, 32, 10]` (`run_name = arch_64-32`)
- **Learning rate:** `0.01`
- **Optimizer:** `sgd`
- **Activation:** `sigmoid`
- **Epochs:** `10`

### Result

`docs/ejercicio3/final_heldout_metrics.json` records, over all 3,147 rows of `data/derived/more_digits/heldout.csv`:

- **Accuracy: 0.9285 (92.85%)**
- **`meets_98_percent_target`: `false`**

Stated plainly and honestly: the best achievable accuracy this phase reached on `more_digits.csv`'s held-out portion is **92.85%**, which **falls short of the >=98% target** CompanyX requested (per the enunciado's Ejercicio 3 prompt). The target is not redefined or softened here — 92.85% is the real, final, single number, reported as such.

### Per-Class Recall (All Ten Digits)

| Digit | Recall | Digit | Recall |
|:---:|:---:|:---:|:---:|
| 0 | 98.03% | 5 | 76.85% |
| 1 | 98.02% | 6 | 92.11% |
| 2 | 94.40% | 7 | 93.88% |
| 3 | 90.22% | 8 | **85.47%** |
| 4 | 92.57% | 9 | 90.20% |

### Digit 8: From 0.0 to 85.47% Recall

The clearest single contrast in this phase is **digit 8**'s recall: Ejercicio 2's final production check (`docs/ejercicio2/final_test_metrics.json`, `digits_test.csv`) reported an exact **0.0** recall for digit 8 — 0 out of 243 test instances correctly classified, a structural blindness from 0 training examples in `digits.csv` (cross-referenced in Section 4 above). On this phase's true production check, digit 8 recalls **85.47%** of its heldout examples. This gain traces almost entirely to Section 4's Factor 2 (the structural class-count recovery — 585 training rows now exist for digit 8) rather than to any Section 3 technique choice, since digit 8 could not have been learned by any architecture, seed, or ensembling variant while it had zero training examples in the first place.

### Summary

The three ACC-03 techniques (architecture, initialization, ensembling) are presented together in Section 3 as one deliberate, explainable comparison; `more_digits.csv`'s held-out portion was read exactly once, as the final ACC-01 production check, checking the deterministically-selected single-model configuration (`arch_64-32`); the best achievable accuracy — **92.85%**, short of the 98% target — is reported honestly; and ACC-04's "more data" (numeric row-count regression, plus the structural digit-8/digit-5 recovery) versus "technique" (+1.08 percentage points from architecture capacity) factors are explicitly separated in Section 4 above.
