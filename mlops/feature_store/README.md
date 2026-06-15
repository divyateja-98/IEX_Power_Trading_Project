# Feast Feature Store

This project uses a local Feast repository in `feature_repo/` for MCP forecasting
features.

## Offline Source

- Raw engineered source: `data/processed/engineered_dataset.csv`
- Feast source table: `data/feature_store/mcp_features.parquet`
- Producer: `mlops/feature_store/prepare_feature_store_data.py`
- Entity: `forecast_id`, default value `IEX_DAM`
- Event timestamp: `Date + (Hour - 1)` hours

## Feature Views

- `mcp_lag_features`: `lag_1`, `lag_24`, `lag_48`
- `mcp_rolling_features`: `rolling_mean_24`, `rolling_std_24`
- `mcp_weather_features`: `temperature`, `humidity`, `wind_speed`, `solar_radiation`
- `mcp_calendar_features`: `hour`, `weekday`

## Feature Flow

1. `feature_engineering.py` creates leakage-safe lag and rolling features.
2. `prepare_feature_store_data.py` aggregates site weather columns into canonical
   weather features and writes Feast parquet.
3. `feature_repo/features.py` declares Feast entities, source, and feature views.
4. `feature_repo/scripts/materialize_features.py` loads offline features into the
   local SQLite online store.
5. `feature_repo/scripts/get_training_features.py` retrieves historical training
   features.
6. `feature_repo/scripts/get_online_features.py` retrieves online inference
   features for `forecast_id`.
