# Lineage

Lineage records explain how artifacts are produced and consumed.

## Current Flow

`IEX_combined.xlsx` + `IEX_weather.xlsx`
-> `master_dataset.csv`
-> `cleaned_dataset.csv`
-> `engineered_dataset.csv`
-> `xgboost_model.pkl`
-> risk analytics and recommendations.

Use DVC lock metadata after `dvc repro` to pin exact artifact versions.
