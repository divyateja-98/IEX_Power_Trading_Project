# Feast Feature Repository

This is the local Feast repository for IEX power trading forecasting.

## Feature Source

`mlops/feature_store/prepare_feature_store_data.py` converts
`data/processed/engineered_dataset.csv` into the canonical Feast source:

`data/processed/feature_engineered.csv`

The Feast source includes timestamp, date, and hour entity keys plus canonical
features for MCP, demand, renewable generation, load forecast, rolling averages,
lag features, and weather aggregates.

This installed Feast version materializes local file sources through parquet, so
the preparation script also writes `data/processed/feature_engineered.parquet`
with the same rows and columns for the Feast `FileSource`.

## Feature View

- `iex_power_features`
  - Entities: `timestamp`, `date`, `hour`
  - Canonical source: `../data/processed/feature_engineered.csv`
  - Feast FileSource: `../data/processed/feature_engineered.parquet`
  - Online store: `feature_repo/data/online_store.db`
  - Registry: `feature_repo/data/registry.db`

## Commands

```powershell
python mlops/feature_store/prepare_feature_store_data.py
feast -c feature_repo apply
python feature_repo/scripts/materialize_features.py
python feature_repo/scripts/get_historical_features.py
python feature_repo/scripts/get_online_features.py
python xgboost_pipeline.py --use-feast --disable-mlflow
```

The default XGBoost pipeline still reads `data/processed/engineered_dataset.csv`.
Feast is used only when `--use-feast` is passed.
