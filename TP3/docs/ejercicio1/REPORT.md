# Ejercicio 1 — Fraud TinyModel

## 1. Dataset Exploration (FRAUD-01)

### Column Meanings & Definitions

The dataset (`data/data and documentation/fraud_dataset.csv`) contains historical transaction records for e-commerce purchases. The column specifications below are transcribed from `fraud_dataset_documentation.pdf`:

| Column | Type | Description |
|---|---|---|
| `timestamp` | Integer | Unix epoch timestamp (seconds) when the transaction was initiated. |
| `account_age_days` | Integer | Number of days since the user account was created. |
| `session_duration_seconds` | Float | Duration of the user's shopping session in seconds. |
| `device_screen_resolution` | Integer | Screen resolution in total pixels ($width \times height$, e.g., 1,049,088 corresponds to $1366 \times 768$). |
| `time_since_last_login_s` | Float | Elapsed time in seconds since the user's prior login. |
| `amount_usd` | Float | Total dollar transaction amount in USD. |
| `quantity_purchased` | Integer | Number of units purchased in the transaction. |
| `days_since_last_purchase` | Float | Number of days elapsed since the user's prior transaction. |
| `items_viewed_before_purchase` | Integer | Number of catalog items inspected during the current session before completing checkout. |
| `big_model_fraud_probability` | Float | Output score from CompanyX's existing complex BigModel teacher, representing an estimated probability $[0, 1]$. "A threshold needs to be determined to claim a certain transaction is to be flagged as fraud or not." |
| `flagged_fraud` | Binary (0/1) | Ground-truth fraud indicator collected by CompanyX post-incident from chargebacks and user reports. The official documentation explicitly mandates: **"This MUST NOT be used when training the model."** |

### Column Value Ranges

Descriptive statistics computed directly from all 7,500 rows (`docs/ejercicio1/dataset_stats.json`):

| Column | Min | Max | Mean | Count |
|---|---|---|---|---|
| `timestamp` | 1,700,001,808.0 | 1,731,534,222.0 | 1,715,783,865.29 | 7,500 |
| `account_age_days` | 1.0 | 3,649.0 | 1,664.80 | 7,500 |
| `session_duration_seconds` | 5.0 | 726.8 | 280.19 | 7,500 |
| `device_screen_resolution` | 1,006,733.0 | 8,310,940.0 | 3,175,686.44 | 7,500 |
| `time_since_last_login_s` | 10.0 | 40,160.8 | 3,589.36 | 7,500 |
| `amount_usd` | 1.0 | 2,000.0 | 109.07 | 7,500 |
| `quantity_purchased` | 1.0 | 24.0 | 5.86 | 7,500 |
| `days_since_last_purchase` | 0.0 | 142.32 | 13.70 | 7,500 |
| `items_viewed_before_purchase` | 1.0 | 29.0 | 8.61 | 7,500 |
| `big_model_fraud_probability` | 0.000897 | 1.0 | 0.422785 | 7,500 |
| `flagged_fraud` | 0.0 | 1.0 | 0.115867 | 7,500 |

### Composition & Class Balance

- **Total Transactions ($N$):** 7,500 rows, 11 columns.
- **Class Balance (`flagged_fraud`):**
  - Legitimate transactions (`flagged_fraud = 0`): 6,631 (88.41%)
  - Fraudulent transactions (`flagged_fraud = 1`): 869 (11.59%)
  - The ground-truth fraud label is highly imbalanced (~1:7.6 ratio). An uninformative baseline predictor predicting constant negative (`flagged_fraud = 0`) trivially attains 88.41% accuracy.

### Cleanliness & Integrity

- **Missing / Null Values:** 0 across all 11 columns (7,500 non-null values for each feature).
- **Duplicate Rows:** 0 identical rows detected (`duplicates = 0`).
- **Target Integrity:** `big_model_fraud_probability` lies strictly within $[0.000897, 1.0] \subset [0, 1]$.
- **Label Integrity:** `flagged_fraud` takes discrete binary values strictly in $\{0, 1\}$.

Because the documentation prohibits training on `flagged_fraud`, this phase trains on `big_model_fraud_probability` and reserves `flagged_fraud` exclusively for evaluation (Plans 05-02/05-03).

## 2. Feature Scaling & Split Strategy (FRAUD-02)

### Train/Test Split Discipline

To prevent data leakage, feature preprocessing is strictly isolated to the training split:
- **Partition:** 80% train ($N_{\text{train}} = 6,000$) and 20% test ($N_{\text{test}} = 1,500$), generated with fixed seed 42.
- **Split-Before-Scale:** The standardization parameters (mean and standard deviation) are computed exclusively over the 6,000 training instances (`data/derived/fraud_scaler.json`).
- **Application:** The exact training-derived statistics are then applied to standardize `fraud_train.csv`, `fraud_test.csv`, and `fraud_full.csv`.
- **Target Integrity:** The training target `big_model_fraud_probability` is passed through unscaled, and `flagged_fraud` is strictly excluded from all feature matrices.

### Scaling Method & Zero-Variance Guard

- **Method:** Z-score standardization ($z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$) across all 9 feature columns (`timestamp`, `amount_usd`, `quantity_purchased`, `session_duration_seconds`, `days_since_last_purchase`, `account_age_days`, `device_screen_resolution`, `time_since_last_login_s`, `items_viewed_before_purchase`).
- **Zero-Variance Guard:** A defensive check treats any column with standard deviation $\sigma = 0$ as having $\sigma = 1.0$, preventing downstream division-by-zero exceptions. On this dataset, all 9 features exhibit non-zero variance, so no guard was triggered.
- **Target Selection:** As established in Section 1, `big_model_fraud_probability` is used as the regression target column during model training, while `flagged_fraud` is held out exclusively for post-training classification evaluation (FRAUD-07).

## 3. Linear vs Non-linear Perceptron (FRAUD-03)

### Training & Evaluation Setup

Both single-layer perceptron models were trained using gradient descent with mean squared error (MSE) loss:
- **Linear Perceptron:** Activation function `identity`, trained for 50 epochs ($\eta = 0.01$, seed 42) on `data/derived/fraud_train.csv`.
- **Non-linear Perceptron:** Activation function `sigmoid`, trained for 50 epochs ($\eta = 0.01$, seed 42) on `data/derived/fraud_train.csv`.
- **Harvesting Predictions:** Using `tp3 train --resume-from` with identical epoch count (0 additional training epochs), both models were evaluated across all 7,500 samples in `data/derived/fraud_full.csv` without modifying the trained weights.

### Probability Framing & Clipping Caveat

When interpreting model outputs as fraud probabilities:
- **Sigmoid Perceptron:** The logistic sigmoid activation $\sigma(h) = \frac{1}{1 + e^{-h}}$ has an inherent codomain of $(0, 1)$. All 7,500 harvested predictions strictly lie within $[0, 1]$ (minimum: 0.007639, maximum: 0.992882), making them directly interpretable as probabilities without transformation.
- **Linear Perceptron:** The identity activation function $f(h) = h$ is unbounded ($(-\infty, \infty)$). Out of the 7,500 full-dataset predictions, exactly **711** predictions fall outside the $[0, 1]$ probability range before clipping:
  - 186 predictions are negative ($< 0.0$, minimum: $-0.510236$).
  - 525 predictions exceed unity ($> 1.0$, maximum: $2.654552$).
- **Post-Hoc Correction:** To frame the linear perceptron's predictions as valid probabilities, an explicit post-hoc clipping transformation must be applied:
  $$\hat{p}_{\text{linear}} = \max(0.0, \min(1.0, \hat{y}))$$

## 4. Model Comparison — Underfitting & Capacity Saturation (FRAUD-04)

### Underfitting Analysis

Evaluating both models on the held-out test split ($N_{\text{test}} = 1,500$) versus their training split ($N_{\text{train}} = 6,000$) reveals marked differences in model capacity:

| Metric | Linear (`identity`) | Non-linear (`sigmoid`) |
|---|---|---|
| Train MSE | 0.030937 | 0.011011 |
| Test MSE | 0.027762 | 0.010312 |
| Test $R^2$ | 0.689463 | 0.884658 |

- **Underfitting Diagnosis:** The linear perceptron exhibits severe underfitting, with a training MSE nearly $3\times$ higher than that of the sigmoid perceptron (0.030937 vs. 0.011011) and an $R^2$ of only 0.6895. Because the target continuous distribution is generated by BigModel (a complex non-linear network mapping transactions to $[0, 1]$), a single hyperplanar decision boundary lacks the capacity to model curvature and feature interactions without hidden layers. The sigmoid perceptron, by incorporating an S-shaped non-linear activation directly matching the sigmoidal output distribution of probability scores, captures significantly more of the variance ($R^2 = 0.8847$).
- **Generalization Gap:** For both models, Test MSE is slightly lower than or comparable to Train MSE, indicating that neither model is overfitting; the primary bottleneck is underfitting in the linear model due to insufficient hypothesis class expressiveness.

### Capacity Saturation & Boundary Clipping

We examine how the models behave near the extremities of the probability scale $[0, 1]$:

- **Sigmoid Saturation:** The sigmoid perceptron records a `saturation_fraction` (fraction of predictions within $\epsilon = 0.02$ of 0.0 or 1.0) of **0.0804** (8.04%). This indicates that approximately 8% of transactions land in regions where $|\sigma'(h)| \approx 0$. In backpropagation, the derivative $\sigma'(h) = \sigma(h)(1 - \sigma(h))$ vanishes in these saturated tails, which naturally diminishes weight update magnitudes for high-confidence predictions. However, because BigModel's own outputs cluster near 0 for legitimate transactions, this saturation reflects true alignment with confident predictions rather than pathological gradient starvation.
- **Linear Out-of-Range Clipping:** In contrast, the linear model records a `clip_fraction` of **0.0948** (9.48%, 711 transactions). The identity model has constant derivative $f'(h) = 1$, meaning it never suffers from vanishing gradients. However, its lack of bounded saturation causes 9.5% of its outputs to escape $[0, 1]$ entirely, requiring artificial truncation. Its `saturation_fraction` is 0.103867 simply because linear outputs happen to pass through the extreme zones $[0, 0.02]$ and $[0.98, 1.00]$.

### Model Selection

Based on quantitative performance on the held-out test split, the non-linear sigmoid perceptron decisively outperforms the linear perceptron, improving test $R^2$ from 0.6895 to 0.8847 (an absolute gain of +0.1952) while providing naturally bounded probabilities that require no post-hoc clipping.

**Selected model:** sigmoid

The selection rule prioritizes higher test $R^2$ (with ties $\le 0.01$ breaking to sigmoid to avoid post-hoc $[0, 1]$ clipping); here, sigmoid's test $R^2$ of 0.884658 exceeds identity's test $R^2$ of 0.689463 by 0.195195, well above any tie threshold. Consequently, `sigmoid` is selected as the model carried into the 5-fold generalization study.

## 5. Generalization Study (FRAUD-05)

### Methodology & Fold Sizing

The generalization ability of the selected model (`sigmoid`) was evaluated using a rigorous 5-fold cross-validation scheme over all 7,500 transactions:
- **Fold Sizing:** Each fold holds out an independent test partition of exactly 1,500 samples ($20\%$), while training on the remaining 6,000 samples ($80\%$).
- **Positive Representation:** With 869 total positive fraud cases across the dataset, each 1,500-sample fold contains an average of ~174 positive cases (specifically: Fold 0: 174, Fold 1: 175, Fold 2: 174, Fold 3: 175, Fold 4: 171). This ensures substantial representation of fraud instances in every validation fold, guaranteeing statistically stable metric estimates.
- **Leakage Prevention:** Standardizer statistics are fit afresh within each fold exclusively on that fold's 6,000 training rows; test rows are never observed during normalization fitting.

### Cross-Validation Results

The per-fold and summary metrics at the nominal threshold $\tau = 0.5$ (`docs/ejercicio1/generalization_metrics.json`):

| Fold | $N_{\text{test}}$ | $N_{\text{pos}}$ | Precision | Recall | $F_1$-Score |
|---|---|---|---|---|---|
| Fold 0 | 1,500 | 174 | 0.3158 | 1.0000 | 0.4800 |
| Fold 1 | 1,500 | 175 | 0.3333 | 1.0000 | 0.5000 |
| Fold 2 | 1,500 | 174 | 0.3508 | 1.0000 | 0.5194 |
| Fold 3 | 1,500 | 175 | 0.3352 | 1.0000 | 0.5022 |
| Fold 4 | 1,500 | 171 | 0.3301 | 1.0000 | 0.4964 |
| **Mean $\pm$ Std** | **1,500** | **173.8** | **0.3331 $\pm$ 0.0112** | **1.0000 $\pm$ 0.0000** | **0.4996 $\pm$ 0.0126** |

### Why Not Just Accuracy? (FRAUD-05a)

Because legitimate transactions make up 88.41% of the dataset (6,631 / 7,500), a trivial baseline model that blindly predicts "legitimate" for every transaction (`baseline_accuracy_always_negative = 0.8841`) scores **88.41% accuracy** while missing 100% of fraud incidents.

Reporting raw accuracy on this problem is misleading:
- A model with 88.4% accuracy can have a recall of 0.0.
- Conversely, at threshold $\tau = 0.50$, the sigmoid perceptron achieves perfect recall ($1.0000$), catching all 869 frauds, but suffers from low precision ($0.3331$), generating two false alarms for every genuine fraud.
- Consequently, precision, recall, and the harmonic mean ($F_1$-score) are the only informative metrics that expose the operational trade-offs of the fraud detector.

## 6. Detection Threshold (FRAUD-06)

To establish a principled operating point for CompanyX's fraud operations, we performed a precision/recall threshold sweep over all 7,500 out-of-fold pooled predictions (`data/derived/oof_predictions.json`), evaluating 19 equidistant thresholds $\tau \in [0.05, 0.95]$ in increments of 0.05 (`docs/ejercicio1/threshold_sweep.json`).

### Operating Point Contrast: Default 0.5 vs. Recommended 0.9

| Parameter / Metric | Nominal Default ($\tau = 0.50$) | Recommended Optimal ($\tau = 0.90$) | $\Delta$ Impact |
|---|---|---|---|
| **Threshold ($\tau$)** | 0.50 | **0.90** | +0.40 |
| **Precision** | 0.3327 (33.27%) | **0.8915 (89.15%)** | **+55.88%** |
| **Recall** | 1.0000 (100.0%) | **0.8412 (84.12%)** | -15.88% |
| **$F_1$-Score** | 0.4993 | **0.8656** | **+0.3663** |
| False Positives ($FP$) | 1,745 | **90** | **-94.8% false alarms** |
| True Positives ($TP$) | 869 | **731** | 731 frauds intercepted |

- **Default ($\tau = 0.50$) Pathology:** At the naive 0.5 default, the model flags 2,614 transactions as fraudulent, of which 1,745 are false positives ($FP$). Only 33.3% of flagged transactions are actual fraud, causing severe operational fatigue and unnecessary customer friction.
- **Recommended Threshold ($\tau = 0.90$):** Shifting the decision threshold to **$\tau = 0.90$** cuts false positives by nearly 95% (from 1,745 down to just 90), boosting precision to **89.15%** while retaining an **84.12%** recall rate (731 of 869 true frauds caught). This yields the global maximum $F_1$-score of **0.8656**.

**Selection Rule:**
`"argmax F1 over a 0.05-step grid from 0.05 to 0.95; no business cost weighting was provided, so F1 is the neutral default (not a default-0.5 threshold)"`

## 7. Target Column & Knowledge-Distillation Note (FRAUD-07)

### Target Column Usage

Throughout Phase 5:
- The continuous score `big_model_fraud_probability` was utilized as the sole regression training target column ($y \in [0, 1]$), enabling the perceptron to learn the smooth teacher probability landscape.
- The ground-truth binary label `flagged_fraud` was held out entirely from model training per the dataset documentation's explicit mandate, and was utilized solely as an external evaluation standard to compute precision, recall, $F_1$, and the threshold sweep.

### Clarification on Knowledge Distillation

CompanyX's framing of TinyModel as 'learning from BigModel' is business narrative, not a literal knowledge-distillation training procedure — no temperature-scaled soft-label matching or logit distillation is implemented; this is ordinary supervised regression against a provided continuous target column.
