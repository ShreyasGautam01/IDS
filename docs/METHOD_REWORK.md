# Methodological Rework Specification

1. Exp. C no longer compares CIC-IDS2017 against source training rows. The comparator is the independent UNSW-NB15 test partition.
2. Standardization is fitted on model-training data only.
3. Identifier-like fields are excluded explicitly and duplicate rows are removed.
4. Distance normalization excludes self-neighbours in the reference set.
5. Abstention thresholds are determined from calibration uncertainty and applied to the test split.
6. Experiment B is expanded to multiple attack families where present in the supplied dataset.
7. Isolation Forest is added as a dedicated novelty-detection baseline.
8. Five repeated seeds are used for the main benchmark.
9. Conformal prediction is evaluated using empirical coverage and prediction-set cardinality, while its binary confidence-equivalent ranking is treated as non-independent.
10. The final paper must not report legacy numerical values as corrected results.
