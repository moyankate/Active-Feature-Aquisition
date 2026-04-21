"""
Custom Naive Bayes classifier that handles NaN (missing feature values).

Vectorized implementation for performance on larger datasets.
"""

import numpy as np


class NaiveBayesCategorical:
    """Naive Bayes for categorical features with missing-value support."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        n, d = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.d_ = d

        class_counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        self.log_prior_ = np.log(class_counts / class_counts.sum())

        self.feature_values_ = []
        self.log_likelihood_ = []
        self.val_to_idx_ = []

        for j in range(d):
            col = X[:, j]
            observed = ~np.isnan(col)
            vals = np.unique(col[observed]).astype(int)
            self.feature_values_.append(vals)
            n_vals = len(vals)
            val_to_idx = {v: i for i, v in enumerate(vals)}
            self.val_to_idx_.append(val_to_idx)

            counts = np.full((self.n_classes_, n_vals), self.alpha)
            if observed.any():
                obs_col = col[observed].astype(int)
                obs_y = y[observed]
                for ci, c in enumerate(self.classes_):
                    class_mask = obs_y == c
                    if class_mask.any():
                        class_vals = obs_col[class_mask]
                        for v, idx in val_to_idx.items():
                            counts[ci, idx] += (class_vals == v).sum()

            log_probs = np.log(counts / counts.sum(axis=1, keepdims=True))
            self.log_likelihood_.append(log_probs)

        self._build_lookup()
        return self

    def _build_lookup(self):
        """Build dense lookup arrays for vectorized predict."""
        self._max_val = max(v.max() for v in self.feature_values_ if len(v) > 0) + 1
        self._log_lk_dense = np.zeros((self.d_, self.n_classes_, self._max_val))
        self._val_known = np.zeros((self.d_, self._max_val), dtype=bool)
        for j in range(self.d_):
            log_probs = self.log_likelihood_[j]
            for v, idx in self.val_to_idx_[j].items():
                self._log_lk_dense[j, :, v] = log_probs[:, idx]
                self._val_known[j, v] = True

    def predict_log_proba(self, X):
        n = X.shape[0]
        log_prob = np.tile(self.log_prior_, (n, 1))

        for j in range(self.d_):
            col = X[:, j]
            observed = ~np.isnan(col)
            if not observed.any():
                continue
            obs_idx = np.where(observed)[0]
            vals = col[obs_idx].astype(int)
            valid = (vals >= 0) & (vals < self._max_val) & self._val_known[j, vals]
            if valid.any():
                vi = obs_idx[valid]
                vv = vals[valid]
                contrib = self._log_lk_dense[j][:, vv].T
                np.add.at(log_prob, vi, contrib)

        return log_prob

    def predict_proba(self, X):
        log_prob = self.predict_log_proba(X)
        log_prob -= log_prob.max(axis=1, keepdims=True)
        prob = np.exp(log_prob)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob

    def predict(self, X):
        prob = self.predict_proba(X)
        return self.classes_[np.argmax(prob, axis=1)]

    def feature_value_proba(self, X_row, feature_idx):
        log_prob = self.log_prior_.copy()
        for j in range(self.d_):
            if j == feature_idx:
                continue
            v = X_row[j]
            if not np.isnan(v):
                vi = int(v)
                if vi < self._max_val and self._val_known[j, vi]:
                    log_prob += self._log_lk_dense[j, :, vi]

        log_prob -= log_prob.max()
        post_class = np.exp(log_prob)
        post_class /= post_class.sum()

        vals = self.feature_values_[feature_idx]
        cond_probs = np.exp(self.log_likelihood_[feature_idx])
        marginal = cond_probs.T @ post_class
        marginal /= marginal.sum()
        return vals, marginal
