from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results_revised'


def ci_t(values, confidence=0.95):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    crit = float(t.ppf((1 + confidence) / 2, len(x) - 1))
    return mean - crit * se, mean + crit * se


def summarize_all(df):
    numeric_cols = [c for c in df.columns if c != 'seed' and pd.api.types.is_numeric_dtype(df[c])]
    row = {}
    for c in numeric_cols:
        vals = pd.to_numeric(df[c], errors='coerce').to_numpy(dtype=float)
        lo, hi = ci_t(vals)
        row[f'{c}__mean'] = float(np.nanmean(vals))
        row[f'{c}__std'] = float(np.nanstd(vals, ddof=1)) if np.sum(np.isfinite(vals)) > 1 else np.nan
        row[f'{c}__ci95_low'] = lo
        row[f'{c}__ci95_high'] = hi
    return pd.DataFrame([row])

path = RESULTS / 'exp_c_by_seed.csv'
if not path.exists():
    raise SystemExit(f'Missing {path}. The benchmark results were not written.')

df = pd.read_csv(path)
summary = summarize_all(df)
summary.to_csv(RESULTS / 'exp_c_summary.csv', index=False)
print(f'Created {RESULTS / "exp_c_summary.csv"}')
print(f'Found {len(df)} completed Experiment C seed rows: {sorted(df["seed"].tolist())}')
