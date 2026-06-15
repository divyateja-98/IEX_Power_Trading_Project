# Enterprise MLOps Architecture

The project now separates product surfaces from MLOps control-plane assets while
preserving the existing forecasting functionality.

## Layers

| Layer | Path | Responsibility |
| --- | --- | --- |
| Backend | `backend/` | FastAPI service entry point for APIs and model-serving endpoints |
| Frontend | `frontend/` | Streamlit application entry point for operator workflows |
| Core logic | `src/` and root scripts | Existing ingestion, cleaning, feature engineering, training, risk, and recommendation logic |
| MLOps | `mlops/` | Validation, DVC operations, feature-store metadata, lineage, monitoring, deployment, CI/CD, and Kubernetes assets |
| Artifacts | `data/`, `models/`, `reports/` | DVC-managed data, model, and diagnostic outputs |

## Data and Model Flow

1. `data_loader.py` validates raw Excel files and writes an ingestion summary.
2. `merge_data.py` joins market and weather data into `master_dataset.csv`.
3. `clean_data.py` removes duplicates and imputes missing values.
4. `feature_engineering.py` creates calendar, lag, and rolling MCP features.
5. `xgboost_pipeline.py` trains and evaluates the XGBoost forecaster.
6. `risk_analytics.py` and `recommendation_engine.py` consume processed data and
   forecasts to support procurement decisions.
7. `backend/` and `frontend/` expose the current application surfaces.

## Reproducibility

`dvc.yaml` defines the executable pipeline and `params.yaml` centralizes paths
and model settings. After DVC is installed and initialized, `dvc repro` rebuilds
the pipeline from raw inputs to trained model.

## Productionization Path

- Add schema checks under `mlops/validation/`.
- Promote `mlops/feature_store/feature_registry.yaml` into a managed feature
  store if online serving is required.
- Store exact run metadata in `mlops/experiments/` or an experiment tracker.
- Use `mlops/lineage/pipeline_lineage.yaml` with DVC lock metadata for audit.
- Add drift and quality checks under `mlops/monitoring/`.
- Build images from `mlops/deployment/`.
- Deploy workloads with `mlops/kubernetes/`.
- Automate gates with `mlops/ci_cd/`.
