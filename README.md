# Benchmarking Lightweight Reliability Signals for Intrusion Detection

This repository contains the experimental implementation for a study of **whether lightweight reliability signals can identify unreliable intrusion-detection predictions when the traffic is unfamiliar or comes from a different dataset**.

The central question is not only whether a classifier predicts benign traffic and attacks accurately. It is also whether the classifier can indicate when its own prediction should not be trusted—for example, when it encounters an attack family that was withheld during training or traffic whose feature distribution differs from the training dataset.

The implementation uses a binary Random Forest intrusion detector and compares confidence-, ensemble-, distance-, and conformal-based signals under three evaluation conditions.

> **Reproducibility note:** The committed `results_revised/` files are archived results supplied with the rework package. They are not a claim that the benchmark was rerun from a fresh checkout. Re-running the commands below will regenerate result files from the data available to the loader. Exact reproduction of the archived run requires the original Parquet cache files, which are not committed.

## 1. What the project does

The benchmark trains a Random Forest classifier to distinguish **benign traffic (0)** from **attack traffic (1)**. It then evaluates four signals that may help identify uncertain or out-of-distribution predictions:

| Signal | What it measures | How to interpret a high value |
| --- | --- | --- |
| Raw confidence | One minus the model's maximum class probability, rescaled for binary classification | The classifier assigns less probability to its most likely class |
| Ensemble disagreement | Disagreement among the individual trees in the Random Forest | The trees do not strongly agree on the predicted class |
| Distance-based uncertainty | Mean distance to nearby training examples in the standardized feature space, normalized using training-reference distances | The example is relatively far from the training reference data |
| Split conformal prediction | Calibration-based nonconformity and prediction-set behaviour | The observed label may be difficult to include in a small prediction set; coverage and set size are the primary conformal outputs |

The Random Forest is configured with 100 trees, maximum depth 15, and `class_weight="balanced_subsample"`. The principal benchmark uses five model/split seeds: `42, 43, 44, 45, 46`. Dataset sampling itself uses a fixed seed of 42 so that the repeated runs vary the experiment seed rather than the sampled rows.

### The three experiments

**Experiment A — In-distribution error reliability (UNSW-NB15)**

The model is trained using the supplied UNSW-NB15 training partition. Twenty-five percent of that training partition is held out for calibration; the remaining 75% is used to fit the model and training-only standardization. Evaluation is performed on the separate UNSW-NB15 test partition.

This experiment asks whether the reliability signals rank incorrect predictions above correct predictions. It also evaluates calibrated abstention operating points at target coverages of 90%, 80%, and 70%. Thresholds are derived from calibration scores, then applied to the test data. Conformal prediction is additionally evaluated using empirical test coverage and prediction-set-size frequencies.

**Experiment B — Withheld attack-family evaluation (UNSW-NB15)**

The runner attempts separate open-set evaluations for four attack categories: `DoS`, `Exploits`, `Fuzzers`, and `Reconnaissance`. For each category, that family is removed from the known training data, and the held-out test examples from that category are treated as unseen. A minimum unseen-test sample check is applied before an experiment is run.

The benchmark compares the uncertainty-score distributions of known test traffic and withheld-family traffic using AUROC. It also reports binary classifier metrics on the withheld examples and includes Isolation Forest as an additional novelty-detection baseline. A distance ablation evaluates k-NN neighbourhood sizes `1, 5, 10, 15, 20` with Euclidean and cosine distance for the DoS case.

**Experiment C — Cross-dataset shift (UNSW-NB15 to CIC-IDS2017)**

A source-domain Random Forest is trained on UNSW-NB15. The implementation maps 11 flow-level features to a shared feature representation, then evaluates CIC-IDS2017 as the target domain. Reliability-score distributions on the target data are compared with scores on the independent held-out UNSW-NB15 test partition. The runner also reports binary classification metrics on the CIC sample.

### Metrics and interpretation

The result files include conventional classification metrics (accuracy, precision, recall, and F1), AUROC for prediction-error ranking or distribution-shift separation, selective risk/coverage measurements, conformal coverage and prediction-set size, and Isolation Forest AUROC for the open-set experiment.

AUROC values are experiment-specific: an error-detection AUROC, an unseen-family AUROC, and a cross-dataset AUROC answer different questions and should not be compared as though they measured the same task.

Conformal prediction is not presented as a novel independent ranking signal. In binary classification, the predicted-label nonconformity ranking can be closely related to ordinary confidence. Its main purpose here is calibration-based prediction-set behaviour—especially empirical coverage and set size.

## 2. Repository structure

```text
.
├── data/
│   └── README.md                 # Dataset cache and reproduction notes
├── docs/
│   ├── METHOD_REWORK.md          # Summary of methodological changes
│   └── PAPER_UPDATE_RULES.md     # Rules for transferring results to the paper
├── results_revised/
│   ├── exp_a_by_seed.csv         # Experiment A results for each seed
│   ├── exp_b_by_seed.csv         # Experiment B results by family and seed
│   ├── exp_b_summary.csv         # Experiment B aggregate statistics
│   ├── exp_c_by_seed.csv         # Experiment C results for each seed
│   ├── exp_c_summary.csv         # Experiment C aggregate statistics
│   ├── distance_ablation.csv     # DoS distance ablation results
│   └── run_manifest.json         # Parameters, seeds, data sizes, and features
├── src/
│   ├── abstention.py             # Calibration thresholds and risk/coverage
│   ├── baselines.py              # Isolation Forest baseline
│   ├── data_loader.py            # Data loading and feature preparation
│   ├── metrics.py                # Classification and AUROC metrics
│   ├── models.py                 # Random Forest classifier wrapper
│   ├── uncertainty.py            # Reliability-signal estimators
│   └── visualize.py              # Plotting utilities
├── finalize_results.py           # Rebuilds the Experiment C summary
├── requirements.txt              # Pinned Python dependencies
├── run_revised_benchmark.py      # Main benchmark entry point
├── run_windows_rework.ps1        # Windows setup-and-run script
└── RUN_ME_FIRST.md               # Quick-start instructions
```

Generated virtual environments, Python caches, and Parquet data caches are excluded from version control.

## 3. Python version and dependencies

**Reference interpreter: Python 3.12.13.** This is the interpreter recorded in the virtual environment distributed inside the original rework archive. The virtual environment itself is intentionally not committed because it contains machine-specific paths and binaries.

Use Python 3.12.x (preferably 3.12.13) to reproduce the packaged environment. The repository's `requirements.txt` pins the main dependencies, including NumPy, pandas, scikit-learn, SciPy, Matplotlib, PyArrow, and joblib.

Check the interpreter before installing:

```bash
python --version
```

The output should be `Python 3.12.13` when using the reference version.

## 4. Reproduce the benchmark

Run commands from the repository root—the directory containing `run_revised_benchmark.py` and `requirements.txt`.

### Windows PowerShell

Install Python 3.12.13 first, then:

```powershell
py -3.12 --version
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python .\run_revised_benchmark.py --seeds 42 43 44 45 46
```

The repository also provides a convenience script:

```powershell
.\run_windows_rework.ps1
```

That script uses `venv\Scripts\python.exe` when it exists; otherwise, it uses the currently available `python` command. It installs the requirements and runs the five-seed benchmark. Therefore, either use the manual installation and direct runner command, or use the convenience script; running both installation sequences is unnecessary.

If PowerShell blocks environment activation, the policy can be relaxed for the current PowerShell process only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Linux / macOS

With Python 3.12 installed:

```bash
python3.12 --version
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_revised_benchmark.py --seeds 42 43 44 45 46
```

The PowerShell convenience script is Windows-specific; on Linux or macOS, invoke the Python runner directly.

### Run selected seeds

For a quick smoke run, a single seed can be requested:

```bash
python run_revised_benchmark.py --seeds 42
```

This runs the same experiment pipeline for one seed, not the full five-seed evaluation. Use all five listed seeds for the benchmark configuration described above.

## 5. Datasets and cache behaviour

The loader expects three files under `data/`:

- `data/unsw_train.parquet`
- `data/unsw_test.parquet`
- `data/cicids2017_sample.parquet`

When the UNSW files are absent, `src/data_loader.py` downloads the train and test Parquet files from the configured Hugging Face dataset source. When the CIC cache is absent, it reads a bounded portion (up to 45 MB) of the configured CIC-IDS2017 Friday CSV source, constructs a benign/attack sample, cleans it, and saves it as Parquet. The default target sample size in the loader is 12,000 rows; the actual number can be lower if the bounded source portion does not contain enough rows in both classes.

The original package included pre-generated cache files, but these large Parquet files were not committed to this Git repository. Consequently:

1. A fresh run needs internet access to obtain the dataset inputs.
2. The downloaded/generated files are cached under `data/` for subsequent runs.
3. The data provider's current files and the loader's bounded CIC CSV read may not produce byte-identical samples to the archived caches.
4. To reproduce the archived results as closely as possible, obtain the original three Parquet files and place them in `data/` using the exact filenames above.

The loader's configured sources are:

- UNSW-NB15 train: [Hugging Face Parquet file](https://huggingface.co/datasets/wwydmanski/UNSW-NB15/resolve/main/train.parquet)
- UNSW-NB15 test: [Hugging Face Parquet file](https://huggingface.co/datasets/wwydmanski/UNSW-NB15/resolve/main/test.parquet)
- CIC-IDS2017 Friday CSV: [Hugging Face CSV file](https://huggingface.co/datasets/bvk/CICIDS-2017/resolve/main/friday.csv)

Do not commit local dataset caches or credentials to the repository.

## 6. Generated outputs

The main runner writes results to `results_revised/`:

| Output | Contents |
| --- | --- |
| `exp_a_by_seed.csv` | Experiment A classification, error-ranking, abstention, and conformal metrics per seed |
| `exp_b_by_seed.csv` | Experiment B metrics for each evaluated withheld family and seed |
| `exp_b_summary.csv` | Experiment B grouped means, standard deviations, and 95% t-intervals |
| `exp_c_by_seed.csv` | Experiment C metrics per seed |
| `exp_c_summary.csv` | Experiment C aggregate means, standard deviations, and 95% t-intervals |
| `distance_ablation.csv` | DoS AUROC values for each distance metric, k, and seed |
| `distance_ablation_summary.csv` | Aggregate distance-ablation statistics |
| `exp_a_summary_by_seed.csv` | Per-seed grouped summary output |
| `run_manifest.json` | Run configuration, data sizes, feature list, and seed information |

The runner writes a manifest before the repeated-seed loop. A completed run should be checked for all expected output files and rows; the presence of a manifest alone does not prove that all experiments completed.

To rebuild the Experiment C aggregate summary from an existing `exp_c_by_seed.csv`, run:

```bash
python finalize_results.py
```

## 7. Methodology and manuscript alignment

Read these files before using the outputs in a paper:

- [Methodological rework](docs/METHOD_REWORK.md) describes the corrections implemented in the benchmark.
- [Paper update rules](docs/PAPER_UPDATE_RULES.md) specifies how the revised results should be represented in the manuscript.

Use the values from the corrected `results_revised/` outputs when updating the revised paper. Do not mix them with legacy values from earlier experiments. Report the evaluation condition and metric definition alongside every AUROC or risk/coverage value.

## 8. Current scope and limitations

- The main classifier is a binary Random Forest; the benchmark is not a multi-class attack-family classifier.
- Experiment B treats the selected attack family as withheld from model training, but the classifier's prediction target remains benign versus attack.
- Experiment C uses a harmonized 11-feature representation, not every feature available in either dataset.
- The CIC-IDS2017 cache is built from a bounded portion of one configured CSV source; it should not be described as the full CIC-IDS2017 dataset.
- The archived CSV outputs are provided for inspection. A clean checkout must rerun the pipeline to generate results for its own downloaded or supplied data.
- The scripts and archived results should be independently validated before treating them as final publication evidence.

## 9. Citation and dataset acknowledgement

When using this implementation or its outputs in academic work, cite the underlying UNSW-NB15 and CIC-IDS2017 dataset publications and describe the precise dataset partitions, sampling, preprocessing, seeds, model settings, and evaluation protocol used. This repository is an implementation artifact; it does not replace citations to the original dataset papers.
