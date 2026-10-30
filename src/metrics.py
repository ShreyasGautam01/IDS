"""Classification and reliability metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_classification(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def auroc_errors(y_true, y_pred, uncertainty):
    errors = (np.asarray(y_true) != np.asarray(y_pred)).astype(int)
    if len(np.unique(errors)) < 2:
        return np.nan
    return float(roc_auc_score(errors, uncertainty))


def auroc_shift(reference_scores, target_scores):
    labels = np.concatenate([np.zeros(len(reference_scores)), np.ones(len(target_scores))])
    scores = np.concatenate([reference_scores, target_scores])
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, scores))


def ci_t(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    from scipy.stats import t
    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(len(values))
    h = t.ppf(0.5 + confidence / 2.0, df=len(values) - 1) * se
    return float(mean - h), float(mean + h)
