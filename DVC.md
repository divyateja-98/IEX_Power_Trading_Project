# DVC Pipeline Documentation

This project uses DVC to version large forecasting artifacts and reproduce the MCP forecasting pipeline from raw IEX and weather inputs through model training.

## Tracked Artifacts

These files are versioned by DVC:

- `data/raw/IEX_combined.xlsx`
- `data/raw/IEX_weather.xlsx`
- `data/processed/master_dataset.csv`
- `data/processed/cleaned_dataset.csv`
- `data/processed/engineered_dataset.csv`
- `models/xgboost_model.pkl`

Raw Excel inputs are tracked with `.dvc` files in `data/raw/`. The processed
datasets, model, generated reports, and model plots are tracked as DVC stage
outputs in `dvc.yaml` and pinned by `dvc.lock`.

## One-Time Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

Initialize Git and DVC if this folder has not already been initialized:

```powershell
git init
dvc init
```

This repository has already been initialized locally. The current raw artifact
metadata files are:

- `data/raw/IEX_combined.xlsx.dvc`
- `data/raw/IEX_weather.xlsx.dvc`

Add the raw source files to DVC:

```powershell
dvc add data/raw/IEX_combined.xlsx
dvc add data/raw/IEX_weather.xlsx
```

The generated datasets and trained model are tracked as pipeline outputs in
`dvc.yaml`. Do not run `dvc add` for those same output files, because DVC should
manage them through the stage graph and `dvc.lock`.

Commit the DVC metadata and pipeline files:

```powershell
git add .gitignore dvc.yaml params.yaml DVC.md requirements.txt data_loader.py merge_data.py
git add .dvc .dvcignore dvc.lock data/raw/*.dvc
git commit -m "Add DVC forecasting pipeline"
```

## Pipeline Stages

The pipeline is defined in `dvc.yaml` and configured by `params.yaml`.

| Stage | Command | Main Outputs |
| --- | --- | --- |
| `ingest` | `python data_loader.py` | `reports/data_summary.txt` |
| `merge` | `python merge_data.py` | `data/processed/master_dataset.csv`, `reports/merge_report.txt` |
| `clean` | `python clean_data.py` | `data/processed/cleaned_dataset.csv`, `reports/cleaning_report.txt` |
| `feature_engineering` | `python feature_engineering.py` | `data/processed/engineered_dataset.csv`, `reports/feature_engineering_report.txt` |
| `train` | `python xgboost_pipeline.py` | `models/xgboost_model.pkl`, `reports/xgboost_model_report.txt`, `reports/modeling/` |

Feature windows and XGBoost hyperparameters are controlled in `params.yaml` and
passed into the DVC stage commands.

## Reproduce the Pipeline

Run every stage:

```powershell
dvc repro
```

Commit the pipeline lock file after a successful reproduction:

```powershell
git add dvc.lock
git commit -m "Reproduce DVC forecasting pipeline"
```

Run from a specific stage:

```powershell
dvc repro merge
```

Inspect the stage graph:

```powershell
dvc dag
```

Check changed data, code, parameters, and outputs:

```powershell
dvc status
```

The current pipeline has been reproduced successfully, and `dvc status` reports:

```text
Data and pipelines are up to date.
```

## Remote Storage

Configure a DVC remote so the team can pull the exact data and model artifacts referenced by Git:

```powershell
dvc remote add -d storage <remote-url>
dvc push
git add .dvc/config
git commit -m "Configure DVC remote"
```

On another machine:

```powershell
git clone <repo-url>
cd IEX_Power_Trading_Project
pip install -r requirements.txt
dvc pull
dvc repro
```

## Notes

- The weather file has been standardized to `data/raw/IEX_weather.xlsx`.
- Keep large datasets, model binaries, and generated reports out of Git. Commit only DVC metadata, source code, `dvc.yaml`, `params.yaml`, and documentation.
- When changing reproducibility settings, update `params.yaml` first and rerun `dvc repro`.
