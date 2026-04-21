"""
Data loading and masking utilities for Active Feature-Value Acquisition project.
Dataset: UCI Breast Cancer (Wisconsin Diagnostic).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


SELECTED_FEATURES = [11, 14, 17, 18, 21, 22, 24, 28]


def load_breast_cancer_data(n_bins=10):
    """
    Load the Wisconsin Diagnostic Breast Cancer dataset (sklearn built-in).
    Selects 8 low-correlation, high-MI features to avoid redundancy.
    Numeric features are discretized into `n_bins` quantile bins so that
    the categorical Naive Bayes and SEU framework can be applied directly.

    Returns (X, y) as numpy int arrays.
      X : (569, 8) discretized feature matrix
      y : (569,)   0 = malignant, 1 = benign
    """
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X_raw = data.data[:, SELECTED_FEATURES]
    disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile",
                               quantile_method="averaged_inverted_cdf")
    X = disc.fit_transform(X_raw).astype(int)
    y = data.target
    return X, y


def apply_mask(X, missing_rate, rng):
    """
    Randomly mask `missing_rate` fraction of feature entries with NaN.

    Parameters
    ----------
    X : pd.DataFrame  (integer-encoded, no NaNs)
    missing_rate : float  e.g. 0.10 or 0.20
    rng : np.random.Generator

    Returns
    -------
    X_masked : pd.DataFrame  (float dtype; NaN where masked)
    mask : np.ndarray bool  True where value is MISSING
    """
    X_masked = X.astype(float).copy()
    n_rows, n_cols = X_masked.shape
    total_entries = n_rows * n_cols
    n_missing = int(total_entries * missing_rate)

    # Flatten, sample indices, reshape
    flat_indices = rng.choice(total_entries, size=n_missing, replace=False)
    row_idx, col_idx = np.unravel_index(flat_indices, (n_rows, n_cols))

    mask = np.zeros((n_rows, n_cols), dtype=bool)
    for r, c in zip(row_idx, col_idx):
        X_masked.iat[r, c] = np.nan
        mask[r, c] = True

    return X_masked, mask



if __name__ == "__main__":
    X, y = load_breast_cancer_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    rng = np.random.default_rng(seed=42)
    X_m10, mask10 = apply_mask(pd.DataFrame(X), 0.10, rng)
    rng2 = np.random.default_rng(seed=42)
    X_m20, mask20 = apply_mask(pd.DataFrame(X), 0.20, rng2)

    print(f"\n10% mask: {mask10.sum()} missing entries "
          f"({mask10.mean()*100:.1f}%)")
    print(f"20% mask: {mask20.sum()} missing entries "
          f"({mask20.mean()*100:.1f}%)")
