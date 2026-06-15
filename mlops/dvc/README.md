# DVC

The active DVC pipeline lives at the repository root:

- `dvc.yaml`
- `params.yaml`
- `DVC.md`

This folder is reserved for DVC operational assets such as remote-storage
runbooks, cache policies, and data-retention notes.

## Pipeline

```powershell
dvc repro
dvc dag
dvc status
```

## Stage Ownership

- `ingest`: raw Excel discovery and summary
- `merge`: market/weather merge into `master_dataset.csv`
- `clean`: duplicate and missing-value handling
- `feature_engineering`: forecasting feature generation
- `train`: XGBoost model training and model diagnostics
