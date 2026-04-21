"""
Data loading and masking utilities for AFA project (Decision Tree variant).
Dataset: UCI Breast Cancer (Wisconsin Diagnostic).

Same data pipeline as NB model, but adds mode imputation for DT compatibility.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


SELECTED_FEATURES = [11, 14, 17, 18, 21, 22, 24, 28]


def load_breast_cancer_data(n_bins=10):
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X_raw = data.data[:, SELECTED_FEATURES]
    try:
        disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile",
                                quantile_method="averaged_inverted_cdf")
    except TypeError:
        disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    X = disc.fit_transform(X_raw).astype(int)
    y = data.target
    return X, y


def apply_mask(X, missing_rate, rng):
    X_masked = X.astype(float).copy()
    n_rows, n_cols = X_masked.shape
    total_entries = n_rows * n_cols
    n_missing = int(total_entries * missing_rate)

    flat_indices = rng.choice(total_entries, size=n_missing, replace=False)
    row_idx, col_idx = np.unravel_index(flat_indices, (n_rows, n_cols))

    mask = np.zeros((n_rows, n_cols), dtype=bool)
    for r, c in zip(row_idx, col_idx):
        X_masked.iat[r, c] = np.nan
        mask[r, c] = True

    return X_masked, mask


def impute_mode(X):
    """
    Replace NaN entries with per-column mode (most frequent value).
    Returns a copy — does not modify X in place.
    """
    X_imp = X.copy()
    for j in range(X_imp.shape[1]):
        col = X_imp[:, j]
        nan_mask = np.isnan(col)
        if not nan_mask.any():
            continue
        observed = col[~nan_mask]
        if len(observed) > 0:
            vals, counts = np.unique(observed.astype(int), return_counts=True)
            mode_val = vals[np.argmax(counts)]
            X_imp[nan_mask, j] = mode_val
        else:
            X_imp[nan_mask, j] = 0
    return X_imp.astype(int)
