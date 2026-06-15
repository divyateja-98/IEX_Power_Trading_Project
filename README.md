# IEX Power Trading Risk Optimization

Electricity price forecasting and risk optimization project for Indian Energy Exchange
(IEX) power trading workflows.

## Objective

Build a Python project that can ingest power market data, prepare forecasting features,
train price prediction models, quantify trading risk, and generate decision support
recommendations for power trading.

## Project Structure

```text
IEX_Power_Trading_Project/
+-- backend/                # Enterprise FastAPI entry point
+-- frontend/               # Enterprise Streamlit entry point
+-- api/                    # Legacy FastAPI service kept for compatibility
+-- dashboard/              # Legacy dashboard files kept for compatibility
+-- data/
|   +-- raw/                # Source data; keep large files out of Git
|   +-- processed/          # Cleaned feature and model-ready datasets
+-- docs/                   # Architecture documentation
+-- mlops/                  # MLOps control plane
+|   +-- dvc/               # DVC operations and data-versioning notes
+|   +-- validation/        # Data, model, and artifact validation
+|   +-- feature_store/     # Feature definitions and registry metadata
+|   +-- experiments/       # Experiment tracking conventions
+|   +-- lineage/           # Artifact lineage metadata
+|   +-- monitoring/        # Drift, quality, and service monitoring config
+|   +-- deployment/        # Docker and runtime deployment assets
+|   +-- ci_cd/             # CI/CD workflow templates
+|   +-- kubernetes/        # Kubernetes service and deployment manifests
+-- notebooks/              # Exploration, diagnostics, and experiments
+-- reports/                # Model reports, charts, and analysis outputs
+-- src/
|   +-- features/           # Feature engineering utilities
|   +-- ingestion/          # Data collection and loading
|   +-- models/             # Training, evaluation, and inference code
|   +-- preprocessing/      # Cleaning and transformation logic
|   +-- recommendation/     # Trading action recommendation logic
|   +-- risk/               # Risk scoring and optimization modules
+-- .gitignore
+-- dvc.yaml                # Reproducible DVC pipeline stages
+-- params.yaml             # Pipeline paths and model parameters
+-- DVC.md                  # DVC setup and reproduction guide
+-- README.md
+-- requirements.txt
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the API:

```bash
uvicorn backend.app.main:app --reload
```

Reproduce the forecasting pipeline with DVC:

```bash
dvc repro
```

See `DVC.md` for DVC initialization, artifact tracking, and remote storage setup.

Train the XGBoost forecaster with MLflow experiment tracking:

```bash
python xgboost_pipeline.py
```

By default, MLflow logs run metadata to `./mlflow.db`, stores artifacts under
`./mlruns`, creates or reuses the `IEX_Power_Forecasting` experiment, and
registers the trained model as `IEX_Power_Forecasting_Model`.

Open the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000` and select the `IEX_Power_Forecasting`
experiment. To use a remote tracking server, set `MLFLOW_TRACKING_URI` or pass
`--mlflow-tracking-uri`. To run training without experiment logging, pass
`--disable-mlflow`.

Run the Streamlit frontend:

```bash
streamlit run frontend/Home.py
```

Run the MLOps smoke check:

```bash
python mlops/validation/smoke_check.py
```

## Recommended Workflow

1. Add raw market, demand, weather, and calendar data to `data/raw/`.
2. Use `src/ingestion/` to load external or local data sources.
3. Use `src/preprocessing/` to clean and normalize datasets.
4. Build model-ready predictors in `src/features/`.
5. Train and evaluate forecasting models in `src/models/`.
6. Estimate exposure, volatility, and downside risk in `src/risk/`.
7. Generate buy, sell, hold, or hedge recommendations in `src/recommendation/`.
8. Publish outputs through `backend/` or `frontend/`.

## MLflow Tracking

`xgboost_pipeline.py` logs the production forecasting run to MLflow with:

- Parameters: `max_depth`, `learning_rate`, `n_estimators`, and `train_size`
- Metrics: `RMSE`, `MAE`, `MAPE`, and `R2`
- Artifacts: `feature_importance.png` and `actual_vs_predicted.png`
- Experiment: `IEX_Power_Forecasting`
- Registry model: `IEX_Power_Forecasting_Model`

The local artifacts are still written to `reports/modeling/`, and the trained
pickle model is still written to `models/xgboost_model.pkl` for compatibility
with the existing project workflow.

## Enterprise MLOps Layout

The project now separates application surfaces from MLOps operations:

- `backend/` exposes the FastAPI service through the enterprise layout while
  preserving `api/` compatibility.
- `frontend/` exposes the Streamlit dashboard while preserving `dashboard/`
  compatibility.
- `mlops/` contains validation, feature-store metadata, lineage, monitoring,
  deployment, CI/CD, Kubernetes, and DVC operational assets.
- `docs/ARCHITECTURE.md` documents the data flow, model flow, and production
  path.

## Python Best Practices

- Keep reusable code in `src/` and experiments in `notebooks/`.
- Keep secrets in `.env` and never commit them.
- Keep large raw and processed datasets out of Git.
- Prefer typed functions, small modules, and explicit configuration.
- Add tests as the project matures.

## License

Add the project license before distribution.
