"""Corrected multi-seed benchmark for the reliability study.

Design decisions:
- Dataset sampling is fixed at seed 42 so repeated runs isolate model/split randomness.
- Experiment C compares CIC-IDS2017 against an independent held-out UNSW test set.
- All preprocessing statistics are fitted on the source/model-training partition only.
- Distance normalization excludes self-neighbours from its reference calibration.
- Abstention thresholds are chosen on calibration data and applied to the test data.
- Experiment B evaluates multiple withheld attack families when available.
- Isolation Forest provides one dedicated open-set baseline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import (
    load_unsw_nb15,
    load_cicids2017_sample,
    numeric_feature_columns,
    extract_harmonized_features,
    prepare_unseen_attack_split,
)
from src.models import IDSClassifier
from src.uncertainty import RawConfidenceEstimator, EnsembleDisagreementEstimator, DistanceBasedEstimator, ConformalPredictionEstimator
from src.metrics import evaluate_classification, auroc_errors, auroc_shift, ci_t
from src.abstention import risk_coverage_curve, calibrated_operating_points
from src.baselines import fit_isolation_forest, anomaly_score

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_revised"
RESULTS.mkdir(exist_ok=True)
DATA_SAMPLE_SEED = 42
RF_PARAMS = dict(n_estimators=100, max_depth=15)
DIST_K = 15
DIST_MAX_REF = 5000
CP_ALPHA = 0.10
ABSTENTION_COVERAGES = (0.90, 0.80, 0.70)
WITHHELD_FAMILIES = ("DoS", "Exploits", "Fuzzers", "Reconnaissance")
DIST_ABLATION_K = (1, 5, 10, 15, 20)
DIST_ABLATION_METRICS = ("euclidean", "cosine")


def split_train_calibration(X, y, seed):
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)


def fit_scaler(X_train, X_cal, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_cal), scaler.transform(X_test), scaler


def fit_signals(rf, X_model_train, X_cal, y_cal, seed, dist_k=DIST_K, dist_metric="euclidean"):
    raw = RawConfidenceEstimator(rf)
    ens = EnsembleDisagreementEstimator(rf)
    dist = DistanceBasedEstimator(dist_k, DIST_MAX_REF, seed, dist_metric).fit(X_model_train)
    cp = ConformalPredictionEstimator(rf, CP_ALPHA).calibrate(X_cal, y_cal)
    return raw, ens, dist, cp


def signal_scores(signals, X):
    raw, ens, dist, cp = signals
    return {
        "raw_confidence": raw.compute_uncertainty(X),
        "ensemble_disagreement": ens.compute_uncertainty(X),
        "distance_based": dist.compute_uncertainty(X),
        "conformal": cp.compute_uncertainty(X),
    }


def run_exp_a(unsw_train, unsw_test, feature_cols, seed):
    X_all = unsw_train[feature_cols].to_numpy(dtype=float)
    y_all = unsw_train["label"].to_numpy(dtype=int)
    X_test = unsw_test[feature_cols].to_numpy(dtype=float)
    y_test = unsw_test["label"].to_numpy(dtype=int)

    X_m, X_cal, y_m, y_cal = split_train_calibration(X_all, y_all, seed)
    X_m, X_cal, X_test, _ = fit_scaler(X_m, X_cal, X_test)

    rf = IDSClassifier(model_type="rf", **RF_PARAMS, random_state=seed).fit(X_m, y_m)
    pred = rf.predict(X_test)
    signals = fit_signals(rf, X_m, X_cal, y_cal, seed)
    test_scores = signal_scores(signals, X_test)
    cal_scores = signal_scores(signals, X_cal)

    metrics = evaluate_classification(y_test, pred)
    for name, scores in test_scores.items():
        metrics[f"error_auroc__{name}"] = auroc_errors(y_test, pred, scores)

    for name in test_scores:
        rc = calibrated_operating_points(cal_scores[name], y_test, pred, test_scores[name], ABSTENTION_COVERAGES)
        for row in rc.to_dict("records"):
            tag = int(round(row["target_coverage"] * 100))
            metrics[f"risk__{name}__{tag}"] = row["risk"]
            metrics[f"cer__{name}__{tag}"] = row["cer"]
            metrics[f"coverage__{name}__{tag}"] = row["actual_coverage"]
            metrics[f"risk_reduction__{name}__{tag}"] = row["risk_reduction_percent"]

    cp = signals[3]
    metrics["conformal_nominal_coverage"] = 1.0 - CP_ALPHA
    metrics["conformal_empirical_coverage"] = cp.empirical_coverage(X_test, y_test)
    metrics["conformal_set_size_0"] = cp.set_size_distribution(X_test)[0]
    metrics["conformal_set_size_1"] = cp.set_size_distribution(X_test)[1]
    metrics["conformal_set_size_2"] = cp.set_size_distribution(X_test)[2]
    return metrics


def run_exp_b_for_family(unsw_train, unsw_test, feature_cols, seed, family):
    if family not in set(unsw_test["attack_cat"].astype(str)):
        return None
    train_known, test_known, test_unseen = prepare_unseen_attack_split(unsw_train, unsw_test, family)
    if len(test_unseen) < 50 or test_unseen["label"].nunique() < 1:
        return None
    X_known = train_known[feature_cols].to_numpy(dtype=float)
    y_known = train_known["label"].to_numpy(dtype=int)
    X_test_known_raw = test_known[feature_cols].to_numpy(dtype=float)
    y_test_known = test_known["label"].to_numpy(dtype=int)
    X_unseen_raw = test_unseen[feature_cols].to_numpy(dtype=float)
    y_unseen = test_unseen["label"].to_numpy(dtype=int)

    X_m, X_cal, y_m, y_cal = split_train_calibration(X_known, y_known, seed)
    X_m, X_cal, X_known_test, scaler = fit_scaler(X_m, X_cal, X_test_known_raw)
    X_unseen = scaler.transform(X_unseen_raw)

    rf = IDSClassifier(model_type="rf", **RF_PARAMS, random_state=seed).fit(X_m, y_m)
    pred_unseen = rf.predict(X_unseen)
    clf_metrics = evaluate_classification(y_unseen, pred_unseen)
    signals = fit_signals(rf, X_m, X_cal, y_cal, seed)
    scores_known = signal_scores(signals, X_known_test)
    scores_unseen = signal_scores(signals, X_unseen)

    rows = {
        "family": family,
        "seed": seed,
        "n_train_known": len(train_known),
        "n_test_known": len(test_known),
        "n_test_unseen": len(test_unseen),
        "unseen_accuracy": clf_metrics["accuracy"],
        "unseen_recall": clf_metrics["recall"],
        "unseen_f1": clf_metrics["f1"],
    }
    for name in scores_known:
        rows[f"auroc__{name}"] = auroc_shift(scores_known[name], scores_unseen[name])

    iso = fit_isolation_forest(X_m, seed)
    rows["auroc__isolation_forest"] = auroc_shift(anomaly_score(iso, X_known_test), anomaly_score(iso, X_unseen))
    return rows


def run_exp_c(unsw_train, unsw_test, cic_raw, seed):
    unsw_h_tr_raw, y_unsw_tr, cic_h, y_cic = extract_harmonized_features(unsw_train, cic_raw)
    unsw_h_te_raw, y_unsw_te, _, _ = extract_harmonized_features(unsw_test.iloc[:0], cic_raw.iloc[:0])
    # Extract the source-test harmonized features through the same feature mapping.
    source_test_h, y_source_test, _, _ = extract_harmonized_features(unsw_test, cic_raw.iloc[:0])

    X_all = unsw_h_tr_raw.to_numpy(dtype=float)
    y_all = y_unsw_tr.astype(int)
    X_source_test_raw = source_test_h.to_numpy(dtype=float)
    y_source_test = y_source_test.astype(int)
    X_target_raw = cic_h.to_numpy(dtype=float)

    X_m_raw, X_cal_raw, y_m, y_cal = split_train_calibration(X_all, y_all, seed)
    scaler = StandardScaler().fit(X_m_raw)
    X_m = scaler.transform(X_m_raw)
    X_cal = scaler.transform(X_cal_raw)
    X_source_test = scaler.transform(X_source_test_raw)
    X_target = scaler.transform(X_target_raw)

    rf = IDSClassifier(model_type="rf", **RF_PARAMS, random_state=seed).fit(X_m, y_m)
    pred_target = rf.predict(X_target)
    target_cls = evaluate_classification(y_cic, pred_target)
    signals = fit_signals(rf, X_m, X_cal, y_cal, seed)
    source_scores = signal_scores(signals, X_source_test)
    target_scores = signal_scores(signals, X_target)

    rows = {
        "seed": seed,
        "source_dataset": "UNSW-NB15",
        "target_dataset": "CIC-IDS2017",
        "n_source_test": len(X_source_test),
        "n_target": len(X_target),
        "target_accuracy": target_cls["accuracy"],
        "target_precision": target_cls["precision"],
        "target_recall": target_cls["recall"],
        "target_f1": target_cls["f1"],
    }
    for name in source_scores:
        rows[f"auroc__{name}"] = auroc_shift(source_scores[name], target_scores[name])
    return rows, (rf, signals, X_m, X_cal, y_m, y_cal, X_source_test, y_source_test, X_target, y_cic)


def run_distance_ablation(exp_b_context, seed, family="DoS"):
    train_known, test_known, test_unseen = exp_b_context
    feature_cols = numeric_feature_columns(train_known)
    X_all = train_known[feature_cols].to_numpy(dtype=float)
    y_all = train_known["label"].to_numpy(dtype=int)
    X_known_raw = test_known[feature_cols].to_numpy(dtype=float)
    X_unseen_raw = test_unseen[feature_cols].to_numpy(dtype=float)
    X_m_raw, X_cal_raw, y_m, y_cal = split_train_calibration(X_all, y_all, seed)
    scaler = StandardScaler().fit(X_m_raw)
    X_m = scaler.transform(X_m_raw)
    X_known = scaler.transform(X_known_raw)
    X_unseen = scaler.transform(X_unseen_raw)
    rows = []
    for metric in DIST_ABLATION_METRICS:
        for k in DIST_ABLATION_K:
            est = DistanceBasedEstimator(k, DIST_MAX_REF, seed, metric).fit(X_m)
            rows.append({
                "family": family,
                "seed": seed,
                "metric": metric,
                "k": k,
                "auroc": auroc_shift(est.compute_uncertainty(X_known), est.compute_uncertainty(X_unseen)),
            })
    return rows


def summarize(df: pd.DataFrame, group_cols):
    numeric_cols = [c for c in df.columns if c not in group_cols and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple): keys = (keys,)
        row = dict(zip(group_cols, keys))
        for c in numeric_cols:
            vals = g[c].to_numpy(dtype=float)
            lo, hi = ci_t(vals)
            row[f"{c}__mean"] = float(np.nanmean(vals))
            row[f"{c}__std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else np.nan
            row[f"{c}__ci95_low"] = lo
            row[f"{c}__ci95_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def main(seeds):
    print("Loading fixed dataset samples...")
    unsw_train, unsw_test = load_unsw_nb15(sample_size=40000, random_state=DATA_SAMPLE_SEED)
    cic = load_cicids2017_sample(num_samples=None, random_state=DATA_SAMPLE_SEED)
    feature_cols = numeric_feature_columns(unsw_train)

    manifest = {
        "data_sample_seed": DATA_SAMPLE_SEED,
        "seeds": list(seeds),
        "rf_params": RF_PARAMS,
        "abstention_coverages": ABSTENTION_COVERAGES,
        "distance": {"k": DIST_K, "max_ref_samples": DIST_MAX_REF, "metrics": DIST_ABLATION_METRICS},
        "conformal_alpha": CP_ALPHA,
        "exp_c_source": "UNSW-NB15 held-out test split",
        "exp_c_target": "CIC-IDS2017",
        "withheld_families_requested": list(WITHHELD_FAMILIES),
        "dataset_sizes": {"unsw_train": len(unsw_train), "unsw_test": len(unsw_test), "cic": len(cic)},
        "feature_count": len(feature_cols),
        "features": feature_cols,
    }
    (RESULTS / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    a_rows = []
    b_rows = []
    c_rows = []
    dist_rows = []

    for seed in seeds:
        print(f"=== Seed {seed} ===")
        a_rows.append({"seed": seed, **run_exp_a(unsw_train, unsw_test, feature_cols, seed)})

        available = [f for f in WITHHELD_FAMILIES if f in set(unsw_test["attack_cat"].astype(str))]
        for family in available:
            row = run_exp_b_for_family(unsw_train, unsw_test, feature_cols, seed, family)
            if row is not None:
                b_rows.append(row)

        c_row, c_context = run_exp_c(unsw_train, unsw_test, cic, seed)
        c_rows.append(c_row)

        # Distance ablation is deliberately focused on DoS to diagnose the legacy pathology.
        train_known, test_known, test_unseen = prepare_unseen_attack_split(unsw_train, unsw_test, "DoS")
        if len(test_unseen) >= 50:
            dist_rows.extend(run_distance_ablation((train_known, test_known, test_unseen), seed))

    a = pd.DataFrame(a_rows)
    b = pd.DataFrame(b_rows)
    c = pd.DataFrame(c_rows)
    d = pd.DataFrame(dist_rows)
    a.to_csv(RESULTS / "exp_a_by_seed.csv", index=False)
    b.to_csv(RESULTS / "exp_b_by_seed.csv", index=False)
    c.to_csv(RESULTS / "exp_c_by_seed.csv", index=False)
    d.to_csv(RESULTS / "distance_ablation.csv", index=False)

    summarize(a, ["seed"]).to_csv(RESULTS / "exp_a_summary_by_seed.csv", index=False)
    if not b.empty:
        summarize(b, ["family"]).to_csv(RESULTS / "exp_b_summary.csv", index=False)
    c_summary = c.copy()
    c_summary["_group"] = "all"
    summarize(c_summary, ["_group"]).drop(columns=["_group"]).to_csv(RESULTS / "exp_c_summary.csv", index=False)
    if not d.empty:
        summarize(d, ["family", "metric", "k"]).to_csv(RESULTS / "distance_ablation_summary.csv", index=False)

    print(f"Results written to {RESULTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    main(args.seeds)
