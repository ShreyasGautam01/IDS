"""Dataset loading and leakage-safe preprocessing for the benchmark."""

from __future__ import annotations

import io
import os
import urllib.request
import requests
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
UNSW_TRAIN_URL = "https://huggingface.co/datasets/wwydmanski/UNSW-NB15/resolve/main/train.parquet"
UNSW_TEST_URL = "https://huggingface.co/datasets/wwydmanski/UNSW-NB15/resolve/main/test.parquet"
CICIDS_URL = "https://huggingface.co/datasets/bvk/CICIDS-2017/resolve/main/friday.csv"

HARMONIZED_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload",
    "sinpkt", "dinpkt", "smean", "dmean"
]


def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path):
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    return dest_path


def _clean_common(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df.reset_index(drop=True)


def load_unsw_nb15(sample_size: int | None = None, random_state: int = 42):
    train_path = download_file(UNSW_TRAIN_URL, os.path.join(DATA_DIR, "unsw_train.parquet"))
    test_path = download_file(UNSW_TEST_URL, os.path.join(DATA_DIR, "unsw_test.parquet"))
    train_df = _clean_common(pd.read_parquet(train_path))
    test_df = _clean_common(pd.read_parquet(test_path))
    if sample_size:
        if len(train_df) > sample_size:
            train_df = train_df.sample(sample_size, random_state=random_state).reset_index(drop=True)
        if len(test_df) > sample_size:
            test_df = test_df.sample(sample_size, random_state=random_state).reset_index(drop=True)
    return train_df, test_df


def load_cicids2017_sample(num_samples: int | None = None, random_state: int = 42):
    sample_path = os.path.join(DATA_DIR, "cicids2017_sample.parquet")
    if os.path.exists(sample_path):
        df = _clean_common(pd.read_parquet(sample_path))
        if num_samples and len(df) > num_samples:
            # Preserve the existing balanced cache while selecting a fixed subset.
            return df.sample(num_samples, random_state=random_state).reset_index(drop=True)
        return df

    os.makedirs(DATA_DIR, exist_ok=True)
    benign_rows, attack_rows = [], []
    with requests.get(CICIDS_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        raw_stream = io.BytesIO(r.raw.read(45 * 1024 * 1024))
        for chunk in pd.read_csv(raw_stream, chunksize=5000, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            label_col = next(c for c in chunk.columns if "label" in c.lower())
            b = chunk[chunk[label_col].astype(str).str.upper() == "BENIGN"]
            a = chunk[chunk[label_col].astype(str).str.upper() != "BENIGN"]
            if b is not None and len(b): benign_rows.append(b)
            if a is not None and len(a): attack_rows.append(a)
            current = sum(map(len, benign_rows)) + sum(map(len, attack_rows))
            target = num_samples or 12000
            if current >= target * 2:
                break
    if not benign_rows or not attack_rows:
        raise RuntimeError("Unable to construct CIC-IDS2017 cached sample")
    target = num_samples or 12000
    b = pd.concat(benign_rows, ignore_index=True)
    a = pd.concat(attack_rows, ignore_index=True)
    half = target // 2
    combined = pd.concat([
        b.sample(min(half, len(b)), random_state=random_state),
        a.sample(min(target - half, len(a)), random_state=random_state),
    ], ignore_index=True).sample(frac=1.0, random_state=random_state)
    combined = _clean_common(combined)
    combined.to_parquet(sample_path, index=False)
    return combined.reset_index(drop=True)


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "attack_cat", "label", "id", "flow id", "flow_id", "timestamp",
        "srcip", "dstip", "source ip", "destination ip"
    }
    cols = []
    for c in df.select_dtypes(include=[np.number]).columns:
        normalized = c.strip().lower()
        if normalized in excluded:
            continue
        if "ip" in normalized or "timestamp" in normalized or normalized.endswith(" flow id"):
            continue
        cols.append(c)
    return cols


def extract_harmonized_features(unsw_df: pd.DataFrame, cicids_df: pd.DataFrame):
    unsw_h = pd.DataFrame(index=unsw_df.index)
    for name in HARMONIZED_FEATURES:
        unsw_h[name] = pd.to_numeric(unsw_df[name], errors="coerce").clip(lower=0)
    y_unsw = unsw_df["label"].astype(int).to_numpy()

    label_col = next(c for c in cicids_df.columns if "label" in c.lower())
    cic_h = pd.DataFrame(index=cicids_df.index)
    cic_h["dur"] = pd.to_numeric(cicids_df[next(c for c in cicids_df.columns if "flow duration" in c.lower())], errors="coerce") / 1e6
    cic_h["spkts"] = pd.to_numeric(cicids_df[next(c for c in cicids_df.columns if "total fwd packet" in c.lower())], errors="coerce")
    cic_h["dpkts"] = pd.to_numeric(cicids_df[next(c for c in cicids_df.columns if "total bwd packet" in c.lower() or "total backward packet" in c.lower())], errors="coerce")
    cic_h["sbytes"] = pd.to_numeric(cicids_df[next(c for c in cicids_df.columns if "fwd packet length" in c.lower())], errors="coerce")
    cic_h["dbytes"] = pd.to_numeric(cicids_df[next(c for c in cicids_df.columns if "bwd packet length" in c.lower())], errors="coerce")
    cic_h["rate"] = pd.to_numeric(cicids_df[next(c for c in cicids_df.columns if "flow packets/s" in c.lower())], errors="coerce")
    cic_h["sload"] = pd.to_numeric(cicids_df[next(c for c in cicids_df.columns if "flow bytes/s" in c.lower())], errors="coerce") * 8
    cic_h["sinpkt"] = pd.to_numeric(cicids_df[next(c for c in cicids_df.columns if "fwd iat mean" in c.lower())], errors="coerce") / 1e3
    cic_h["dinpkt"] = pd.to_numeric(cicids_df[next(c for c in cicids_df.columns if "bwd iat mean" in c.lower())], errors="coerce") / 1e3
    cic_h["smean"] = pd.to_numeric(cicids_df[next(c for c in cicids_df.columns if "fwd packet length mean" in c.lower())], errors="coerce")
    cic_h["dmean"] = pd.to_numeric(cicids_df[next(c for c in cicids_df.columns if "bwd packet length mean" in c.lower())], errors="coerce")
    y_cic = (cicids_df[label_col].astype(str).str.strip().str.upper() != "BENIGN").astype(int).to_numpy()

    for df in (unsw_h, cic_h):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)
        df.fillna(0.0, inplace=True)
        for c in df.columns:
            df[c] = df[c].clip(lower=0)
    return unsw_h.reset_index(drop=True), y_unsw, cic_h.reset_index(drop=True), y_cic


def prepare_unseen_attack_split(train_df: pd.DataFrame, test_df: pd.DataFrame, withheld_category: str):
    train_known = train_df[train_df["attack_cat"] != withheld_category].copy()
    test_known = test_df[test_df["attack_cat"] != withheld_category].copy()
    test_unseen = test_df[test_df["attack_cat"] == withheld_category].copy()
    return train_known, test_known, test_unseen
