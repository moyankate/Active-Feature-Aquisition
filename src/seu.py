"""
SEU (Sampled Expected Utility) acquisition strategies for Active Feature-Value Acquisition.

Implements:
  - SEU-US  : uniform random candidate sampling
  - SEU-ES  : error/uncertainty-based candidate sampling
  - SEU-Accuracy : SEU using raw accuracy instead of log-gain (ablation)
  - Uniform baseline : random acquisition
"""

import numpy as np
from naive_bayes import NaiveBayesCategorical


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def log_gain(proba_before, proba_after, y_val, eps=1e-15):
    """
    Log-gain utility for a single instance.
    log P(correct class | new data) - log P(correct class | old data)
    """
    p_before = np.clip(proba_before[y_val], eps, 1.0)
    p_after = np.clip(proba_after[y_val], eps, 1.0)
    return np.log(p_after) - np.log(p_before)


def accuracy_gain(pred_before, pred_after, y_val):
    """Accuracy gain (0/1) for a single instance."""
    return float(pred_after == y_val) - float(pred_before == y_val)


# ---------------------------------------------------------------------------
# SEU core
# ---------------------------------------------------------------------------

def compute_seu_score(model, X_masked, y, i, j, utility="log_gain",
                      acquisition_cost=1.0):
    """
    Compute SEU score for acquiring feature j of instance i.

    E(q_{i,j}) = sum_k  U(F_{i,j} = v_k) * P(F_{i,j} = v_k)

    Returns float score.
    """
    vals, val_probs = model.feature_value_proba(X_masked[i], j)
    if len(vals) == 0:
        return 0.0

    # Current prediction for instance i
    row_before = X_masked[i:i+1]
    proba_before = model.predict_proba(row_before)[0]
    pred_before = model.classes_[np.argmax(proba_before)]

    expected_utility = 0.0
    for v_k, p_k in zip(vals, val_probs):
        # Hypothetically set F_{i,j} = v_k
        row_hyp = X_masked[i].copy()
        row_hyp[j] = v_k
        proba_after = model.predict_proba(row_hyp[None, :])[0]
        pred_after = model.classes_[np.argmax(proba_after)]

        if utility == "log_gain":
            u = log_gain(proba_before, proba_after, y[i])
        else:  # accuracy
            u = accuracy_gain(pred_before, pred_after, y[i])

        expected_utility += u * p_k

    return expected_utility / acquisition_cost


def select_candidates_us(missing_entries, n_sample, rng):
    """Uniform random sampling of candidate (i, j) pairs."""
    if len(missing_entries) <= n_sample:
        return missing_entries
    idx = rng.choice(len(missing_entries), size=n_sample, replace=False)
    return [missing_entries[k] for k in idx]


def select_candidates_es(missing_entries, model, X_masked, n_sample, rng):
    """
    Error/uncertainty sampling: prioritize instances that are misclassified
    or have high prediction uncertainty.
    """
    if len(missing_entries) <= n_sample:
        return missing_entries

    # Get unique instance indices from missing entries
    instances = list({i for i, j in missing_entries})
    proba = model.predict_proba(X_masked[instances])
    # Uncertainty = 1 - max probability
    uncertainty = 1.0 - proba.max(axis=1)
    inst_unc = dict(zip(instances, uncertainty))

    # Score each candidate entry by instance uncertainty
    scores = np.array([inst_unc[i] for i, j in missing_entries])
    # Sample proportional to uncertainty (with small uniform floor)
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
    """
    Run one full acquisition experiment.

    Parameters
    ----------
    X_full          : np.ndarray (n, d)  fully observed encoded features
    y               : np.ndarray (n,)    class labels
    missing_rate    : float
    strategy        : str  'uniform' | 'seu-us' | 'seu-es' | 'seu-accuracy'
    seed            : int
    acquisition_cost: float  cost per feature acquisition
    sample_fraction : float  fraction of missing entries scored per SEU round
    utility         : str    'log_gain' | 'accuracy' (overridden for seu-accuracy)
    verbose         : bool   if True, log per-round utility stats to debug_file
    debug_file      : str    path to write verbose log
    max_rounds      : int    stop after this many rounds (None = run to completion)
    batch_size      : int    number of features to acquire per round

    Returns
    -------
    cost_curve     : list of cumulative costs at each acquisition step
    accuracy_curve : list of validation accuracy after each acquisition
    """
    rng = np.random.default_rng(seed)

    from data_utils import apply_mask
    import pandas as pd
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

    model = NaiveBayesCategorical()
    model.fit(X_masked[train_idx], y[train_idx])
    acc = (model.predict(X_masked[val_idx]) == y[val_idx]).mean()
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
                candidates = select_candidates_es(remaining, model, X_masked, n_sample, rng)

            scores = [
                compute_seu_score(model, X_masked, y, i, j,
                                  utility=utility,
                                  acquisition_cost=acquisition_cost)
                for (i, j) in candidates
            ]

            if verbose and log_fh and utility == "log_gain":
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

        model = NaiveBayesCategorical()
        model.fit(X_masked[train_idx], y[train_idx])
        acc = (model.predict(X_masked[val_idx]) == y[val_idx]).mean()

        cost_curve.append(cumulative_cost)
        accuracy_curve.append(acc)
        round_num += 1

        if max_rounds is not None and round_num >= max_rounds:
            break

    if log_fh:
        log_fh.close()

    return cost_curve, accuracy_curve


def run_fully_observed_baseline(X_full, y, seed=9999):
    """
    Train on fully observed data. seed=9999 matches run_acquisition(seed=0).
    For fair per-trial comparison use run_fully_observed_per_trial().
    """
    rng = np.random.default_rng(seed)
    n = X_full.shape[0]
    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    model = NaiveBayesCategorical()
    model.fit(X_full[train_idx], y[train_idx])
    return (model.predict(X_full[val_idx]) == y[val_idx]).mean()


def run_fully_observed_per_trial(X_full, y, n_trials=20):
    """
    Compute fully observed accuracy for each trial's own train/val split,
    matching the splits used by run_acquisition(seed=0..n_trials-1).
    Returns (mean, std) — the correct ceiling for multi-trial comparisons.
    """
    accs = [run_fully_observed_baseline(X_full, y, seed=s + 9999)
            for s in range(n_trials)]
    return float(np.mean(accs)), float(np.std(accs))
