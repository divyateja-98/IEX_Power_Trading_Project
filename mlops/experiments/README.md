# Experiments

Use this area for experiment tracking notes, model comparison outputs, and
controlled changes to `params.yaml`.

## Baseline

- Model: XGBoost regressor
- Split: chronological train-test split
- Test size: `params.yaml -> train.test_size`
- Artifact: `models/xgboost_model.pkl`
- Report: `reports/xgboost_model_report.txt`
- MLflow experiment: `IEX_Power_Forecasting`
- MLflow registered model: `IEX_Power_Forecasting_Model`

## MLflow Usage

Run the training pipeline:

```bash
python xgboost_pipeline.py
```

Start the local MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`, select `IEX_Power_Forecasting`, and inspect the
run parameters, regression metrics, plots, and registered model versions.

For shared environments, configure a remote tracking server with
`MLFLOW_TRACKING_URI` or the `--mlflow-tracking-uri` CLI option.

## Recommended Experiment Record

Capture:

- Git commit
- DVC lock version
- Parameter changes
- Training metrics
- Feature set changes
- MLflow run ID and registered model version
- Business notes for procurement impact
