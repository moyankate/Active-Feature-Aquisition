"""
SEU acquisition strategies for AFA — Decision Tree base learner variant.

Key difference from NB model:
  - Base learner: sklearn DecisionTreeClassifier (with mode imputation for NaN)
  - Distribution estimator: NaiveBayesCategorical (for P(F_{i,j} = v_k))
  - Log Gain should now work correctly, matching the original paper (Saar-Tsechansky et al. 2009)

Strategies:
  - Uniform      : random acquisition baseline
  - SEU-US       : SEU with uniform candidate sampling, log-gain utility
  - SEU-ES       : SEU with uncertainty-based candidate sampling, log-gain utility
  - SEU-Accuracy : SEU with accuracy gain (ablation)
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from naive_bayes import NaiveBayesCategorical
from data_utils import impute_mode


def _fit_dt(X_masked, y, train_idx):
    """Fit a decision tree on mode-imputed training data."""
    X_train = impute_mode(X_masked[train_idx])
    dt = DecisionTreeClassifier(random_state=0, min_samples_leaf=10)
    dt.fit(X_train, y[train_idx])
    return dt


def _predict_proba_dt(dt, X_masked, indices):
    """Predict class probabilities with mode imputation for missing values."""
    X_imp = impute_mode(X_masked[indices])
    return dt.predict_proba(X_imp)


def _predict_dt(dt, X_masked, indices):
    """Predict classes with mode imputation."""
    X_imp = impute_mode(X_masked[indices])
    return dt.predict(X_imp)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def log_gain(proba_before, proba_after, y_val, eps=1e-15):
    p_before = np.clip(proba_before[y_val], eps, 1.0)
    p_after = np.clip(proba_after[y_val], eps, 1.0)
    return np.log(p_after) - np.log(p_before)


def accuracy_gain(pred_before, pred_after, y_val):
    return float(pred_after == y_val) - float(pred_before == y_val)


# ---------------------------------------------------------------------------
# SEU core
# ---------------------------------------------------------------------------

def compute_seu_score(dt, nb_model, X_masked, y, i, j,
                      utility="log_gain", acquisition_cost=1.0):
    """
    Compute SEU score for acquiring feature j of instance i.

    Uses NB to estimate P(F_{i,j} = v_k), and DT to evaluate utility
    of each hypothetical acquisition outcome.
    """
    vals, val_probs = nb_model.feature_value_proba(X_masked[i], j)
    if len(vals) == 0:
        return 0.0

    row_before = X_masked[i:i+1]
    proba_before = _predict_proba_dt(dt, row_before, [0])[0]
    pred_before = dt.classes_[np.argmax(proba_before)]

    expected_utility = 0.0
    for v_k, p_k in zip(vals, val_probs):
        row_hyp = X_masked[i].copy()
        row_hyp[j] = v_k
        proba_after = _predict_proba_dt(dt, row_hyp[None, :], [0])[0]
        pred_after = dt.classes_[np.argmax(proba_after)]

        if utility == "log_gain":
            u = log_gain(proba_before, proba_after, y[i])
        else:
            u = accuracy_gain(pred_before, pred_after, y[i])

        expected_utility += u * p_k

    return expected_utility / acquisition_cost


def select_candidates_us(missing_entries, n_sample, rng):
    if len(missing_entries) <= n_sample:
        return missing_entries
    idx = rng.choice(len(missing_entries), size=n_sample, replace=False)
    return [missing_entries[k] for k in idx]


def select_candidates_es(missing_entries, dt, X_masked, n_sample, rng):
    """
    Error/uncertainty sampling: prioritize instances with high prediction
    uncertainty under the DT model.
    """
    if len(missing_entries) <= n_sample:
        return missing_entries

    instances = list({i for i, j in missing_entries})
    proba = _predict_proba_dt(dt, X_masked, instances)
    uncertainty = 1.0 - proba.max(axis=1)
    inst_unc = dict(zip(instances, uncertainty))

    scores = np.array([inst_unc[i] for i, j in missing_entries])
    scores = scores + 1e-6
    probs = scores / scores.sum()
    idx = rng.choice(len(missing_entries), size=n_sample, replace=False, p=probs)
    return [missing_entries[k] for k in idx]


# ---------------------------------------------------------------------------
# Acquisition loop
# ---------------------------------------------------------------------------

def run_acquisition(X_full, y, missing_rate, strategy, seed=0,
                    acquisition_cost=1.0, sample_fraction=1.0,
                    utility="log_gain", verbose=False, debug_file=None,
                    max_rounds=None, batch_size=10):
    rng = np.random.default_rng(seed)

    import pandas as pd
    from data_utils import apply_mask
    X_masked, mask = apply_mask(pd.DataFrame(X_full), missing_rate, rng)
    X_masked = X_masked.values.copy()

    n, d = X_masked.shape

    missing_entries = [(i, j) for i in range(n)
                       for j in range(d) if mask[i, j]]

    rng_split = np.random.default_rng(seed + 9999)
    idx = rng_split.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    if strategy == "seu-accuracy":
        utility = "accuracy"
        strategy_base = "seu-us"
    else:
        strategy_base = strategy

    n_sample = max(1, int(len(missing_entries) * sample_fraction))

    cost_curve = [0.0]
    cumulative_cost = 0.0

    # Fit DT (base learner) and NB (distribution estimator)
    dt = _fit_dt(X_masked, y, train_idx)
    nb_model = NaiveBayesCategorical()
    nb_model.fit(X_masked[train_idx], y[train_idx])

    acc = (_predict_dt(dt, X_masked, val_idx) == y[val_idx]).mean()
    accuracy_curve = [acc]

    remaining = list(missing_entries)
    remaining_set = set(missing_entries)
    round_num = 0

    if verbose:
        log_path = debug_file if debug_file else "utility_debug.txt"
        log_fh = open(log_path, "a")
        log_fh.write(f"\n=== strategy={strategy}  seed={seed}  missing_rate={missing_rate} ===\n")
        log_fh.write(f"{'Round':>6}  {'#Remaining':>10}  {'Mean':>10}  {'Min':>10}  {'Max':>10}  {'#Pos':>6}\n")
    else:
        log_fh = None

    while remaining:
        bs = min(batch_size, len(remaining))

        if strategy == "uniform":
            idxs = rng.choice(len(remaining), size=bs, replace=False)
            to_acquire = [remaining[k] for k in idxs]
        else:
            if strategy_base == "seu-us":
                candidates = select_candidates_us(remaining, n_sample, rng)
            else:
                candidates = select_candidates_es(remaining, dt, X_masked, n_sample, rng)

            scores = [
                compute_seu_score(dt, nb_model, X_masked, y, i, j,
                                  utility=utility,
                                  acquisition_cost=acquisition_cost)
                for (i, j) in candidates
            ]

            if verbose and log_fh:
                s_arr = np.array(scores)
                log_fh.write(
                    f"{round_num:>6}  {len(remaining):>10}  "
                    f"{s_arr.mean():>10.5f}  {s_arr.min():>10.5f}  "
                    f"{s_arr.max():>10.5f}  {(s_arr > 0).sum():>6}\n"
                )

            top_idxs = np.argsort(scores)[-bs:][::-1]
            to_acquire = [candidates[k] for k in top_idxs]

        for (i, j) in to_acquire:
            X_masked[i, j] = X_full[i, j]
            cumulative_cost += acquisition_cost
            remaining_set.discard((i, j))

        remaining = [e for e in remaining if e in remaining_set]

        # Re-fit both models after acquisition
        dt = _fit_dt(X_masked, y, train_idx)
        nb_model = NaiveBayesCategorical()
        nb_model.fit(X_masked[train_idx], y[train_idx])

        acc = (_predict_dt(dt, X_masked, val_idx) == y[val_idx]).mean()

        cost_curve.append(cumulative_cost)
        accuracy_curve.append(acc)
        round_num += 1

        if max_rounds is not None and round_num >= max_rounds:
            break

    if log_fh:
        log_fh.close()

    return cost_curve, accuracy_curve


def run_fully_observed_baseline(X_full, y, seed=9999):
    rng = np.random.default_rng(seed)
    n = X_full.shape[0]
    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    dt = DecisionTreeClassifier(random_state=0, min_samples_leaf=10)
    dt.fit(X_full[train_idx], y[train_idx])
    return (dt.predict(X_full[val_idx]) == y[val_idx]).mean()
