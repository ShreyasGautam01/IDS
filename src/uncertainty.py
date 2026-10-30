"""Reliability signals for the IDS benchmark."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class RawConfidenceEstimator:
    """Binary classifier uncertainty: 2 * (1 - max class probability)."""

    def __init__(self, classifier):
        self.clf = classifier

    def compute_uncertainty(self, X: np.ndarray) -> np.ndarray:
        probs = self.clf.predict_proba(X)
        return np.clip(2.0 * (1.0 - np.max(probs, axis=1)), 0.0, 1.0)


class EnsembleDisagreementEstimator:
    """Hard-vote disagreement across Random-Forest trees."""

    def __init__(self, classifier):
        self.clf = classifier

    def compute_uncertainty(self, X: np.ndarray) -> np.ndarray:
        tree_probs = self.clf.get_individual_tree_predictions(X)
        tree_votes = np.argmax(tree_probs, axis=2)
        max_vote_fraction = np.max(
            np.stack([np.mean(tree_votes == 0, axis=0), np.mean(tree_votes == 1, axis=0)], axis=1),
            axis=1,
        )
        return np.clip(2.0 * (1.0 - max_vote_fraction), 0.0, 1.0)


class DistanceBasedEstimator:
    """k-NN feature-space novelty with explicit reference-set handling.

    The reference distribution is drawn only from model-training data. Reference
    distances used for normalization exclude the point itself, preventing zero
    self-neighbor distances from artificially shrinking the scale.
    """

    def __init__(self, k_neighbors: int = 15, max_ref_samples: int = 5000,
                 random_state: int = 42, metric: str = "euclidean"):
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        if metric not in {"euclidean", "cosine"}:
            raise ValueError("metric must be 'euclidean' or 'cosine'")
        self.k = int(k_neighbors)
        self.max_ref = int(max_ref_samples)
        self.random_state = int(random_state)
        self.metric = metric
        self.nn = None
        self.max_dist_ref = 1.0
        self.ref_X = None

    def fit(self, X_train: np.ndarray):
        X_train = np.asarray(X_train)
        rng = np.random.RandomState(self.random_state)
        if len(X_train) > self.max_ref:
            idx = rng.choice(len(X_train), size=self.max_ref, replace=False)
            self.ref_X = X_train[idx]
        else:
            self.ref_X = X_train.copy()

        if len(self.ref_X) <= self.k:
            raise ValueError("Reference set must contain more samples than k_neighbors")

        self.nn = NearestNeighbors(
            n_neighbors=self.k,
            metric=self.metric,
            n_jobs=-1,
        ).fit(self.ref_X)

        # Query the reference set with one extra neighbour and drop the self match.
        nn_norm = NearestNeighbors(
            n_neighbors=self.k + 1,
            metric=self.metric,
            n_jobs=-1,
        ).fit(self.ref_X)
        ref_dists, ref_idx = nn_norm.kneighbors(self.ref_X)
        mean_ref = np.empty(len(self.ref_X), dtype=float)
        for i in range(len(self.ref_X)):
            row_d = ref_dists[i]
            row_i = ref_idx[i]
            keep = row_i != i
            kept = row_d[keep][:self.k]
            if len(kept) < self.k:
                kept = row_d[1:self.k + 1]
            mean_ref[i] = np.mean(kept)
        self.max_dist_ref = float(np.percentile(mean_ref, 95))
        if not np.isfinite(self.max_dist_ref) or self.max_dist_ref <= 0:
            self.max_dist_ref = 1.0
        return self

    def compute_uncertainty(self, X: np.ndarray) -> np.ndarray:
        if self.nn is None:
            raise RuntimeError("Distance estimator must be fitted first")
        dists, _ = self.nn.kneighbors(np.asarray(X))
        mean_dists = np.mean(dists, axis=1)
        norm_dist = mean_dists / self.max_dist_ref
        return np.clip(1.0 - np.exp(-norm_dist), 0.0, 1.0)


class ConformalPredictionEstimator:
    """Split conformal prediction with APS-free binary nonconformity.

    The ranking induced by the predicted-label nonconformity score is equivalent
    to raw confidence for binary classification. The main reported contribution
    is therefore empirical coverage and prediction-set behavior, not a novel
    ranking signal.
    """

    def __init__(self, classifier, alpha: float = 0.10):
        self.clf = classifier
        self.alpha = float(alpha)
        self.q_hat = None
        self.cal_scores = None

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        probs = self.clf.predict_proba(X_cal)
        y_cal = np.asarray(y_cal, dtype=int)
        n = len(y_cal)
        self.cal_scores = 1.0 - probs[np.arange(n), y_cal]
        # Standard split-conformal quantile index with finite-sample correction.
        rank = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        rank = min(max(rank, 1), n)
        self.q_hat = float(np.sort(self.cal_scores)[rank - 1])
        return self

    def predict_sets(self, X: np.ndarray) -> list[list[int]]:
        if self.q_hat is None:
            raise RuntimeError("Conformal predictor must be calibrated first")
        probs = self.clf.predict_proba(X)
        out = []
        for row in probs:
            s = []
            if 1.0 - row[0] <= self.q_hat:
                s.append(0)
            if 1.0 - row[1] <= self.q_hat:
                s.append(1)
            out.append(s)
        return out

    def empirical_coverage(self, X: np.ndarray, y: np.ndarray) -> float:
        sets = self.predict_sets(X)
        y = np.asarray(y, dtype=int)
        return float(np.mean([yi in si for yi, si in zip(y, sets)]))

    def set_size_distribution(self, X: np.ndarray) -> dict[int, float]:
        sizes = np.array([len(s) for s in self.predict_sets(X)], dtype=int)
        n = len(sizes)
        return {k: float(np.mean(sizes == k)) for k in range(3)} if n else {0: 0.0, 1: 0.0, 2: 0.0}

    def compute_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Continuous uncertainty used only for comparison with legacy rankings."""
        if self.q_hat is None:
            raise RuntimeError("Conformal predictor must be calibrated first")
        probs = self.clf.predict_proba(X)
        top_prob = np.max(probs, axis=1)
        pvals = np.column_stack([
            np.mean(self.cal_scores[None, :] >= (1.0 - probs[:, 0, None]), axis=1),
            np.mean(self.cal_scores[None, :] >= (1.0 - probs[:, 1, None]), axis=1),
        ])
        ambiguity = 2.0 * np.min(pvals, axis=1)
        empty_set = 1.0 - np.max(pvals, axis=1)
        return np.clip(np.maximum(ambiguity, empty_set), 0.0, 1.0)
