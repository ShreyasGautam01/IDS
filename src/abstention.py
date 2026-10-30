"""Leakage-safe selective prediction and risk-coverage evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibration_threshold(scores: np.ndarray, target_coverage: float) -> float:
    if not 0 < target_coverage <= 1:
        raise ValueError("target_coverage must lie in (0, 1]")
    return float(np.quantile(scores, target_coverage, method="higher"))


def evaluate_at_threshold(y_true, y_pred, scores, tau):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = np.asarray(scores)
    accepted = scores <= tau
    n = len(y_true)
    coverage = float(np.mean(accepted))
    if accepted.any():
        risk = float(np.mean(y_true[accepted] != y_pred[accepted]))
    else:
        risk = 0.0
    cer = float(np.sum((y_true != y_pred) & accepted) / n)
    return coverage, risk, cer, int(accepted.sum())


def risk_coverage_curve(y_true, y_pred, scores, n_points=101):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = np.asarray(scores)
    order = np.argsort(scores, kind="mergesort")
    errors = (y_true[order] != y_pred[order]).astype(int)
    cum_errors = np.cumsum(errors)
    n = len(y_true)
    rows = []
    for k in np.unique(np.linspace(1, n, min(n_points, n), dtype=int)):
        rows.append({
            "coverage": k / n,
            "risk": float(cum_errors[k - 1] / k),
            "cer": float(cum_errors[k - 1] / n),
            "n_accepted": int(k),
        })
    return pd.DataFrame(rows)


def calibrated_operating_points(cal_scores, y_test_true, y_test_pred, test_scores, coverages=(0.90, 0.80, 0.70)):
    y_test_true = np.asarray(y_test_true)
    y_test_pred = np.asarray(y_test_pred)
    base_risk = float(np.mean(y_test_true != y_test_pred))
    rows = []
    for cov in coverages:
        tau = calibration_threshold(cal_scores, cov)
        actual, risk, cer, n_acc = evaluate_at_threshold(y_test_true, y_test_pred, test_scores, tau)
        reduction = ((base_risk - risk) / base_risk * 100.0) if base_risk > 0 else 0.0
        rows.append({
            "target_coverage": cov,
            "threshold": tau,
            "actual_coverage": actual,
            "risk": risk,
            "cer": cer,
            "risk_reduction_percent": reduction,
            "n_accepted": n_acc,
        })
    return pd.DataFrame(rows)
