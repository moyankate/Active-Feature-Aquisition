"""
SEU acquisition strategies for AFA — HistGradientBoosting base learner.

Key design (following Saar-Tsechansky et al. 2009):
  - Base learner: HistGradientBoostingClassifier (native NaN support, no imputation)
  - Distribution estimator: NaiveBayesCategorical (for P(F_{i,j} = v_k))
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from sklearn.experimental import enable_hist_gradient_boosting
except ImportError:
    pass
from sklearn.ensemble import HistGradientBoostingClassifier
from naive_bayes import NaiveBayesCategorical


def _fit_model(X_masked, y, train_idx):
    hgb = HistGradientBoostingClassifier(
        random_state=0, max_leaf_nodes=16, min_samples_leaf=10,
        max_iter=100,
    )
    hgb.fit(X_masked[train_idx], y[train_idx])
    return hgb


def _predict_proba(model, X):
    return model.predict_proba(X)


def _predict(model, X):
    return model.predict(X)


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

def compute_seu_score(model, nb_model, X_masked, y, i, j,
                      utility="log_gain", acquisition_cost=1.0):
    vals, val_probs = nb_model.feature_value_proba(X_masked[i], j)
    if len(vals) == 0:
        return 0.0

    row_before = X_masked[i:i+1]
    proba_before = _predict_proba(model, row_before)[0]
    pred_before = model.classes_[np.argmax(proba_before)]

    expected_utility = 0.0
    for v_k, p_k in zip(vals, val_probs):
        row_hyp = X_masked[i].copy()
        row_hyp[j] = v_k
        proba_after = _predict_proba(model, row_hyp[None, :])[0]
        pred_after = model.classes_[np.argmax(proba_after)]

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


def select_candidates_es(missing_entries, model, X_masked, n_sample, rng):
    if len(missing_entries) <= n_sample:
        return missing_entries

    instances = list({i for i, j in missing_entries})
    proba = _predict_proba(model, X_masked[instances])
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

    model = _fit_model(X_masked, y, train_idx)
    nb_model = NaiveBayesCategorical()
    nb_model.fit(X_masked[train_idx], y[train_idx])

    acc = (_predict(model, X_masked[val_idx]) == y[val_idx]).mean()
    accuracy_curve = [acc]

    remaining = list(missing_entries)
    remaining_set = set(missing_entries)
    round_num = 0

    if verbose:
        log_path = debug_file if debug_file else "utility_debug.txt"
        log_fh = open(log_path, "a")
        log_fh.write(f"\n=== strategy={strategy}  seed={seed}  missing_rate={missing_rate} ===\n")
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
                candidates = select_candidates_es(remaining, model, X_masked, n_sample, rng)

            scores = [
                compute_seu_score(model, nb_model, X_masked, y, i, j,
                                  utility=utility,
                                  acquisition_cost=acquisition_cost)
                for (i, j) in candidates
            ]

            if verbose and log_fh:
                s_arr = np.array(scores)
                log_fh.write(
                    f"Round {round_num}: remaining={len(remaining)} "
                    f"mean={s_arr.mean():.5f} min={s_arr.min():.5f} "
                    f"max={s_arr.max():.5f} #pos={(s_arr>0).sum()}\n"
                )

            top_idxs = np.argsort(scores)[-bs:][::-1]
            to_acquire = [candidates[k] for k in top_idxs]

        for (i, j) in to_acquire:
            X_masked[i, j] = X_full[i, j]
            cumulative_cost += acquisition_cost
            remaining_set.discard((i, j))

        remaining = [e for e in remaining if e in remaining_set]

        model = _fit_model(X_masked, y, train_idx)
        nb_model = NaiveBayesCategorical()
        nb_model.fit(X_masked[train_idx], y[train_idx])

        acc = (_predict(model, X_masked[val_idx]) == y[val_idx]).mean()
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
    hgb = HistGradientBoostingClassifier(
        random_state=0, max_leaf_nodes=16, min_samples_leaf=10, max_iter=100,
    )
    hgb.fit(X_full[train_idx], y[train_idx])
    return (hgb.predict(X_full[val_idx]) == y[val_idx]).mean()
