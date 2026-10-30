# Dataset cache files

The source package included these generated cache files, which are intentionally not tracked in Git:

- `unsw_train.parquet`
- `unsw_test.parquet`
- `cicids2017_sample.parquet`

The loader in `../src/data_loader.py` downloads the UNSW-NB15 train/test Parquet inputs when absent and builds the CIC-IDS2017 sample from its configured upstream CSV source. A fresh run therefore requires network access. To reproduce the archived run using the exact cached inputs distributed in the original package, put the three original files here with the filenames above. The generated cache files are excluded by the repository's `.gitignore`.
