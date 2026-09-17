# Ejercicio 2 — Digits MLP

## 1. Dataset Exploration & Class Imbalance

### Dataset Composition & Dimensions

The training and development dataset (`data/data and documentation/digits.csv`) consists of 12,449 instances representing grayscale handwritten digit images of dimensions $28 \times 28$ (784 pixel intensity values). The pixel features are formatted as comma-separated lists of floating-point values already scaled to the normalized interval $[0, 1]$, requiring no additional feature standardization (unlike Ejercicio 1's heterogeneous transaction columns).

The held-out test dataset (`data/data and documentation/digits_test.csv`) contains 2,497 instances with the identical 784-dimensional representation.

### Class Distribution & Severe Imbalance

Descriptive label counts derived from `docs/ejercicio2/dataset_stats.json`:

| Digit Class | `digits.csv` (Train/Val Pool) | `digits.csv` Share | `digits_test.csv` (Test Pool) | `digits_test.csv` Share |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 1,480 | 11.89% | 245 | 9.81% |
| 1 | 1,685 | 13.54% | 283 | 11.33% |
| 2 | 1,489 | 11.96% | 258 | 10.33% |
| 3 | 1,532 | 12.31% | 252 | 10.09% |
| 4 | 1,460 | 11.73% | 245 | 9.81% |
| 5 | 271 | 2.18% | 223 | 8.93% |
| 6 | 1,479 | 11.88% | 239 | 9.57% |
| 7 | 1,566 | 12.58% | 257 | 10.29% |
| 8 | **0** | **0.00%** | 243 | 9.73% |
| 9 | 1,487 | 11.94% | 252 | 10.09% |
| **Total** | **12,449** | **100.00%** | **2,497** | **100.00%** |

Two critical findings emerge from this analysis:
1. **Digit 8 is entirely absent from `digits.csv` (0 instances):** While `digits_test.csv` includes 243 examples of digit 8 (~9.73%), the training dataset has 0 instances. By construction, any model trained solely on `digits.csv` cannot learn representations or discriminative boundaries for digit 8.
2. **Digit 5 is a severe minority class (271 instances, 2.18%):** Across the other eight available classes (0–4, 6, 7, 9), representation is uniformly balanced between 1,460 and 1,685 samples (~11.7% to 13.5% each). Digit 5 has fewer than a fifth of the samples of any other class in `digits.csv`, whereas in `digits_test.csv` it is fully represented (223 samples, 8.93%).

Consistent with the project's established design discipline from Phase 5 (no synthetic oversampling / no SMOTE), this imbalance is documented and reported honestly rather than artificially masked.

---

## 2. Model, Split & Evaluation Methodology (DIGIT-01, DIGIT-02, DIGIT-03)

### Core Engine Architecture (DIGIT-01)

The digit classifier is implemented as a Multilayer Perceptron (MLP) within the C++ core engine. Multiclass classification is natively handled using the Phase 4 engine seams:
- **Softmax Output Layer (`use_softmax_output: true`):** The output layer produces normalized probabilities across the 10 digit classes ($p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$) via the numerically stable subtract-max softmax implementation.
- **Injectable Cross-Entropy Loss (`loss: "cross_entropy"`):** Backpropagation uses the analytic softmax + cross-entropy gradient delta ($\boldsymbol{\delta}^{(L)} = \mathbf{p} - \mathbf{y}$), eliminating redundant sigmoid saturation or hand-rolled approximations.

### Internal Stratified Train/Validation Split (DIGIT-02)

To satisfy requirement DIGIT-02:
- All model exploration, hyperparameter tuning (learning rate, architecture, optimizer), and model selection must rely exclusively on `digits.csv`.
- `digits_test.csv` is strictly quarantined and touched **exactly once** at the conclusion of the study (Plan 06-04) to provide the unbiased final production benchmark.

To ensure deterministic, balanced, and reproducible evaluation across all candidate variants, an internal 80/20 stratified split is created using `scripts/digits_preprocess.py` (fixed random seed 42):
- **Internal Training Split (`train.csv`):** 9,960 instances (80% of each present class; digit 8 has 0; digit 5 has 217).
- **Internal Validation Split (`val.csv`):** 2,489 instances (20% of each present class; digit 8 has 0; digit 5 has 54).

Every variant model trained across Wave 2 and Wave 3 is trained on `train.csv` and harvested on `val.csv` under zero additional training epochs via `--resume-from`.

### Justification of Evaluation Metrics (DIGIT-03)

We evaluate models using two complementary metrics: **overall accuracy** and **per-class recall**.

#### Why Not Just Accuracy?
An aggregate accuracy score computed over `val.csv` would obscure class-level failure modes. In particular:
- Because digit 8 is entirely absent from the internal dataset, a classifier that never predicts digit 8 (or misclassifies all digit 8 instances) suffers zero penalty in validation accuracy.
- Digit 5 represents only 2.17% (54 / 2,489) of the validation set. A model with near-zero sensitivity to digit 5 could still easily attain >94% overall accuracy.

Therefore, aggregate accuracy measures general classification efficiency across the supported classes, while per-class recall ($R_c = \frac{TP_c}{TP_c + FN_c}$) isolates class-specific sensitivity and explicitly diagnoses whether minority classes (digit 5) or unobserved classes (digit 8) are properly identified. For class 8, where support in the internal split is 0, recall is explicitly reported as `None` (undefined, rather than a misleading 0.0 or runtime error).

### Baseline Tracer Verification

As an initial tracer to validate the pipeline end-to-end (DIGIT-01 and DIGIT-02), a baseline MLP architecture `[784, 32, 10]` with sigmoid hidden activations was trained with SGD ($\eta = 0.05$) for 10 epochs using `scripts/digits_train_variant.py`. 

Evaluating predictions harvested on the internal validation split (`runs/digits/baseline_val/mlp_42_1789682044604300.json`) yields:
- **Validation Accuracy:** **94.13%** (0.9413), significantly exceeding the >10% random-guess threshold.
- **Per-Class Recall:**
  - Class 0: 96.62%
  - Class 1: 98.22%
  - Class 2: 93.62%
  - Class 3: 88.89%
  - Class 4: 91.10%
  - Class 5: **77.78%** (reflecting the minority training sample count)
  - Class 6: 96.28%
  - Class 7: 94.89%
  - Class 8: **None** (0 ground-truth samples available)
  - Class 9: 95.96%

This baseline confirms that the softmax/cross-entropy engine learns effective digit representations in 10 epochs and clearly demonstrates the utility of per-class recall in exposing the vulnerability on digit 5 and the structural blindness to digit 8.

---

## 3. Learning-Rate Variants (DIGIT-04)

We investigated learning-rate sensitivity by sweeping learning rate $\eta \in \{0.01, 0.05, 2.0\}$ on the baseline architecture (`[784, 32, 10]`, sigmoid, SGD, 10 epochs). The values and metrics recorded in `docs/ejercicio2/lr_variants.json`:

| Variant | Learning Rate $\eta$ | Val Accuracy | Val Accuracy (%) | Non-Finite Loss? | Loss Trajectory (Epoch 5 $\to$ 10) |
|---|:---:|:---:|:---:|:---:|:---:|
| `lr_0.01` | 0.01 | 0.9413 | 94.13% | False | 0.1794 $\to$ 0.1201 |
| `baseline` | 0.05 | 0.9413 | 94.13% | False | 0.0821 $\to$ 0.0399 |
| `lr_2.0` | 2.00 | 0.5629 | 56.29% | False | 3.0034 $\to$ 5.0398 |

### Boundary Analysis
- At $\eta = 0.01$ and $\eta = 0.05$, training progresses stably with monotonic loss decreases, reaching an identical validation accuracy of **94.13%**. The Phase 4 numerical guard in `CrossEntropyLoss` ensures all probabilities remain properly bounded without NaN or null loss entries.
- The deliberate boundary test at $\eta = 2.0$ causes catastrophic gradient overshoot: training loss escalates from 3.00 at epoch 5 to 5.04 at epoch 10, and validation accuracy plummets to **56.29%** (with class 2 and class 5 recall collapsing to 0.0%). This demonstrates that the swept range spans a genuine regime transition from stability to divergence.

---

## 4. Architecture Variants (DIGIT-05)

Holding learning rate ($\eta = 0.05$), optimizer (`sgd`), activation (`sigmoid`), epochs ($10$), and the internal split fixed, we evaluated three network depths and capacities (`docs/ejercicio2/architecture_variants.json`):

| Architecture | Hidden Layers | Parameters ($N_{params}$) | Val Accuracy | Val Accuracy (%) |
|---|:---:|:---:|:---:|:---:|
| `[784, 16, 10]` | 1 hidden (16 units) | 12,730 | 0.9389 | 93.89% |
| `[784, 32, 10]` | 1 hidden (32 units) | 25,450 | 0.9413 | 94.13% |
| `[784, 32, 16, 10]` | 2 hidden (32, 16 units) | 25,818 | **0.9478** | **94.78%** |

### Capacity and Depth Findings
- Halving the hidden layer to 16 units (`[784, 16, 10]`) reduces parameter count by 50% with only a marginal 0.24 percentage point drop in accuracy (93.89% vs 94.13%).
- Adding a second hidden layer (`[784, 32, 16, 10]`) improves validation accuracy to **94.78%** (+0.65% over baseline) with minimal parameter growth (25,818 vs 25,450), indicating that hierarchical feature representation aids digit separation.

---

## 5. Optimization-Mechanism Variants (DIGIT-06)

We evaluated Phase 4's three optimization mechanisms—Stochastic Gradient Descent (`sgd`), Classical Momentum (`momentum`, $\beta = 0.9$), and Adam (`adam`, $\beta_1 = 0.9, \beta_2 = 0.999$)—on the baseline architecture (`[784, 32, 10]`) under identical conditions ($\eta = 0.05$, 10 epochs, sigmoid activation, stratified split). Results recorded in `docs/ejercicio2/optimizer_variants.json`:

| Optimizer | Hyperparameters | Val Accuracy | Val Accuracy (%) | Loss Trajectory (Epoch 5 $\to$ 10) |
|---|---|:---:|:---:|:---:|
| `sgd` | $\eta = 0.05$ | **0.9413** | **94.13%** | 0.0821 $\to$ 0.0399 |
| `momentum` | $\eta = 0.05, \beta = 0.9$ | 0.8923 | 89.23% | 0.3347 $\to$ 0.3398 |
| `adam` | $\eta = 0.05, \beta_1 = 0.9, \beta_2 = 0.999$ | 0.8674 | 86.74% | 0.7535 $\to$ 0.6148 |

### Fair Comparison & Methodological Caveat
Under a strictly controlled comparison with identical learning rate $\eta = 0.05$, plain SGD outperformed Momentum (89.23%) and Adam (86.74%). 

**Methodological Note:** As specified in REQUIREMENTS.md, the shared learning rate across all three optimizers is a deliberate simplification to isolate the optimizer mechanism itself without confounding variables. While adaptive optimizers such as Adam typically excel at lower default learning rates (e.g., $10^{-3}$), conducting an automated multi-hyperparameter grid search per optimizer was explicitly excluded from the TP scope.

---

## 6. Combined Comparison & Selected Configuration (DIGIT-07)

### Selection Rule
Model selection is governed by the deterministic decision rule:
> **Selection Rule:** *"maximize val_accuracy; ties broken by fewer n_params, then lower learning_rate, then alphabetically-first optimizer name."*

Applying this rule independently to each axis yields:
1. **Best Learning Rate:** $\eta = 0.01$ (tied at 94.13% with 0.05; broken by lower learning rate).
2. **Best Architecture:** `[784, 32, 16, 10]` (highest validation accuracy: 94.78%).
3. **Best Optimizer:** `sgd` (highest validation accuracy: 94.13%).

### Composed Combined-Best Configuration
The composed configuration (`docs/ejercicio2/variant_comparison.json`) combines these axis winners:
- **Architecture:** `[784, 32, 16, 10]` (25,818 parameters)
- **Learning Rate:** $\eta = 0.01$
- **Optimizer:** `sgd`
- **Activation:** `sigmoid`
- **Epochs:** 10

### Validation Outcome
Training the combined-best model on `train.csv` and harvesting predictions on `val.csv` yields:
- **Combined-Best Val Accuracy:** **93.13%** (0.9313).

**Observation:** The combined configuration attained 93.13%, which is slightly lower than the single-axis architecture peak (94.78% achieved under $\eta = 0.05$). This illustrates a key principle in empirical machine learning: **hyperparameter interactions are non-additive**. When the deeper 2-layer architecture was combined with the conservative $\eta = 0.01$ rate, 10 epochs of training proved slightly insufficient to reach full convergence compared to $\eta = 0.05$. However, the configuration was deterministically selected by protocol and frozen for the production test.

---

## 7. Final Production Check on digits_test.csv (DIGIT-02 close-out)

In strict accordance with requirement DIGIT-02, the held-out production test set (`data/data and documentation/digits_test.csv`) was quarantined throughout all tuning and touched **exactly once** via `scripts/digits_compare.py --final-check`.

The frozen combined-best model snapshot (`runs/digits/combined_best/model.json`) was loaded with zero additional training epochs (`--resume-from`) to evaluate all 2,497 test instances. Recorded in `docs/ejercicio2/final_test_metrics.json`:

### Overall Test Performance
- **Total Test Instances:** 2,497
- **Final Test Accuracy:** **82.78%** (0.8278)

### Per-Class Recall Breakdown

| Digit Class | Test Set Instances | Recalled | Recall (%) | Status / Explanation |
|:---:|:---:|:---:|:---:|---|
| 0 | 245 | 243 | **99.18%** | Excellent recognition |
| 1 | 283 | 280 | **98.94%** | Excellent recognition |
| 2 | 258 | 236 | **91.47%** | Strong recognition |
| 3 | 252 | 242 | **96.03%** | Excellent recognition |
| 4 | 245 | 232 | **94.69%** | Strong recognition |
| 5 | 223 | 141 | **63.23%** | Impaired due to severe training minority (only 271 train examples) |
| 6 | 239 | 228 | **95.40%** | Excellent recognition |
| 7 | 257 | 241 | **93.77%** | Strong recognition |
| 8 | 243 | 0 | **0.00%** | **Completely blind (0 training samples in `digits.csv`)** |
| 9 | 252 | 224 | **88.89%** | Good recognition |

### Analysis of the Final Test Benchmark
1. **The Digit 8 Zero-Recall Phenomenon:** Digit 8 achieves exactly **0.00% recall** (0 out of 243 instances correctly classified). This confirms the structural blindness identified in Section 1: because `digits.csv` contained 0 instances of digit 8, the model never formed weights to activate the output neuron for class 8.
2. **Impact on Aggregate Accuracy:** Digit 8 accounts for 243 out of 2,497 test samples (~9.73% of the test set). Excluding digit 8, accuracy on the 9 trained classes is **91.70%** (2,067 / 2,254).
3. **Digit 5 Minority Effect:** Digit 5 achieved 63.23% recall, noticeably lower than the ~91%–99% achieved by well-represented classes (0–4, 6, 7), mirroring its 2.18% representation in the training dataset.

This rigorous, audit-proof single test check concludes Phase 6 (Ejercicio 2) and establishes the clean baseline for the accuracy push in Phase 7 (Ejercicio 3).

