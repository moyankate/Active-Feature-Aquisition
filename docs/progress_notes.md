# Progress Notes — AFA with SEU
**Date:** 2026-03-30
**Contributors:** Kate Zhang, Sorro Sun

---

## Project Files

| File | Description |
|------|-------------|
| `breast+cancer/breast-cancer.data` | UCI Breast Cancer dataset (downloaded locally, 277 clean instances) |
| `data_utils.py / .ipynb` | Data loading, encoding, random masking pipeline |
| `naive_bayes.py / .ipynb` | Custom Naive Bayes with NaN support + P(F=v\|obs) estimation |
| `seu.py / .ipynb` | SEU scoring, SEU-US/ES/Accuracy variants, acquisition loop |
| `run_experiments.py / .ipynb` | Full experiment runner, plots, summary table |
| `utility_debug.txt` | Per-round log-gain utility distribution log (v2 diagnostic) |

### Result Files (two versions for comparison)

| File | Version | Settings |
|------|---------|---------|
| `results_v1_missing10.png` | v1 | N_TRIALS=5, sample_fraction=0.3 |
| `results_v1_missing20.png` | v1 | N_TRIALS=5, sample_fraction=0.3 |
| `results_v1_combined.png` | v1 | side-by-side 10% + 20% |
| `results_v1_early_stage.png` | v1 | first 50% budget |
| `results_v2_learning_curves.png` | **v2** | N_TRIALS=20, sample_fraction=1.0 |
| `results_v2_auc_bars.png` | **v2** | AUC bar chart |
| `results_v2_early_stage.png` | **v2** | first 50% budget |

---

## Implementation Details

### Data Pipeline (Week 1)
- 277 instances after dropping rows with original `?` values (286 raw)
- 9 categorical features, 2 classes (196 no-recurrence / 81 recurrence)
- `LabelEncoder` on all columns; `apply_mask()` creates NaN entries at specified rate

### Naive Bayes (Week 2–3)
- `NaiveBayesCategorical`: skips NaN entries in training and prediction
- Laplace smoothing α=1
- `feature_value_proba(x_row, j)`: estimates P(F=v_k | observed) via class-posterior weighting — used by SEU scoring

### SEU Acquisition Loop (Week 2–3)
- 80/20 train/val split, fixed per trial
- Per round: sample candidates → score with SEU → acquire best → retrain NB
- **v1**: `sample_fraction=0.3` (score 30% of remaining candidates)
- **v2**: `sample_fraction=1.0` (score all candidates — removes subsampling noise)

### Variants (Week 4)
- **SEU-US**: uniform random candidate sampling
- **SEU-ES**: uncertainty-weighted sampling (1 − max_proba)
- **SEU-Accuracy**: accuracy gain utility instead of log-gain (ablation)
- **Uniform**: random acquisition baseline
- **Fully observed ceiling**: 0.7500

---

## Experimental Results

### v1 Results (N_TRIALS=5, sample_fraction=0.3)

| Strategy | @25% budget | @50% budget | @75% budget | @100% |
|---|---|---|---|---|
| Uniform | 0.725 | 0.729 | 0.729 | 0.732 |
| SEU-US | 0.725 | 0.721 | 0.721 | 0.732 |
| SEU-ES | 0.725 | 0.725 | 0.721 | 0.732 |
| **SEU-Accuracy** | **0.761** | **0.761** | **0.761** | 0.732 |

### v2 Results (N_TRIALS=20, sample_fraction=1.0, with AUC)

#### 10% Missingness

| Strategy | @25% | @50% | @75% | @100% | AUC |
|---|---|---|---|---|---|
| Uniform | 0.7441 | 0.7460 | 0.7436 | 0.7420 | 0.7419 ± 0.0460 |
| SEU-US | 0.7375 | 0.7428 | 0.7368 | 0.7420 | 0.7386 ± 0.0458 |
| SEU-ES | 0.7375 | 0.7428 | 0.7368 | 0.7420 | 0.7386 ± 0.0458 |
| **SEU-Accuracy** | **0.7606** | **0.7612** | **0.7591** | 0.7420 | **0.7578 ± 0.0453** |

#### 20% Missingness

| Strategy | @25% | @50% | @75% | @100% | AUC |
|---|---|---|---|---|---|
| Uniform | 0.7415 | 0.7376 | 0.7380 | 0.7420 | 0.7387 ± 0.0469 |
| SEU-US | 0.7319 | 0.7277 | 0.7305 | 0.7420 | 0.7303 ± 0.0441 |
| SEU-ES | 0.7319 | 0.7277 | 0.7305 | 0.7420 | 0.7303 ± 0.0441 |
| **SEU-Accuracy** | **0.7694** | **0.7635** | **0.7640** | 0.7420 | **0.7635 ± 0.0509** |

**Fully observed ceiling: 0.7500**

---

## Key Findings

### 1. SEU-Accuracy consistently outperforms all other methods
At 25% budget, SEU-Accuracy achieves 0.761 (10%) and 0.769 (20%) vs Uniform's 0.744 and 0.742 — a gap of ~0.02. This finding is **consistent across v1 and v2**, and becomes cleaner with full scoring (v2). AUC confirms this: SEU-Accuracy AUC = 0.7578 (10%) and 0.7635 (20%), highest of all strategies.

### 2. Log-gain utility is near-uniformly zero or negative — explains SEU-US/ES failure
`utility_debug.txt` reveals: **all log-gain scores are negative (mean ≈ −0.02), with #Pos=0 across all rounds**. This means SEU with log-gain always selects the "least bad" acquisition, which reduces to near-random behavior. The Naive Bayes model is already well-calibrated on the observed features, so acquiring any new feature slightly decreases the log-probability of the true class (because it redistributes probability mass). Accuracy gain, being a discrete 0/1 signal, identifies acquisitions that genuinely flip predictions.

### 3. SEU-US and SEU-ES are equivalent on this dataset
The two sampling strategies yield identical AUC. With full candidate scoring (v2), the only difference is *which candidates are nominated* — but since all candidates are scored equally badly under log-gain, the max-score selection is essentially arbitrary.

### 4. Variance remains high (±0.045–0.051)
Even with 20 trials, std is ≈ 6% of accuracy. This reflects genuine sensitivity to which specific feature entries are masked — small dataset effect.

### 5. v1 → v2 comparison
Removing subsampling noise (sample_fraction: 0.3 → 1.0) did not change the ranking, confirming the v1 findings were not artifacts of sampling. The SEU-Accuracy advantage is real.

---

## What Remains (Week 5 — Final Report)
- [ ] Discuss log-gain pathology in depth: why does Naive Bayes produce systematically negative log-gain? (Likely: acquiring features always hurts calibrated posteriors on a balanced prior, but accuracy gain captures boundary changes)
- [ ] Consider alternative utility: expected reduction in prediction entropy
- [ ] AUC significance test (t-test or Wilcoxon) between SEU-Accuracy and Uniform
- [ ] Write final report discussion section comparing to Saar et al. [1] findings
