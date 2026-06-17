"""Train an XGBoost regression model for IEX MCP forecasting."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

from lineage.openlineage_config import lineage_run
from mlops.mlflow_utils import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_MODEL_REGISTRY_NAME,
    log_xgboost_artifacts,
    log_xgboost_model,
    promote_latest_model_version,
    setup_mlflow,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "feature_store" / "historical_features.parquet"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.pkl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "xgboost_model_report.txt"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "reports" / "modeling"
DEFAULT_FEAST_REPO_PATH = PROJECT_ROOT / "feature_repo"
DEFAULT_FEAST_ENTITY_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered.csv"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 500
DEFAULT_LEARNING_RATE = 0.04
DEFAULT_MAX_DEPTH = 5
DEFAULT_SUBSAMPLE = 0.85
DEFAULT_COLSAMPLE_BYTREE = 0.85
DEFAULT_EVAL_METRIC = "rmse"
DEFAULT_MLFLOW_RUN_NAME = "xgboost_mcp_forecast"


def configure_logging() -> None:
    """Configure console logging for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load the training dataset from CSV or parquet."""
    if not input_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {input_path}")

    logging.info("Loading training dataset: %s", input_path)
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        dataframe = pd.read_parquet(input_path)
    elif suffix == ".csv":
        dataframe = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported training dataset format: {input_path.suffix}")
    dataframe.columns = dataframe.columns.astype(str).str.strip()
    return dataframe


def load_dataset_from_feast(repo_path: Path, entity_path: Path) -> pd.DataFrame:
    """Load point-in-time training features from Feast when explicitly requested."""
    if not entity_path.exists():
        raise FileNotFoundError(f"Feast entity dataframe not found: {entity_path}")

    feature_repo_path = repo_path.resolve()
    if str(feature_repo_path) not in sys.path:
        sys.path.insert(0, str(feature_repo_path))

    from feast_env import configure_feast_environment
    from feature_refs import FEATURE_REFS

    configure_feast_environment()
    from feast import FeatureStore

    entity_df = pd.read_csv(entity_path)
    required_columns = ["timestamp", "date", "hour", "event_timestamp"]
    missing_columns = [column for column in required_columns if column not in entity_df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing Feast entity column(s): {missing}")

    entity_df = entity_df[required_columns].copy()
    entity_df["event_timestamp"] = pd.to_datetime(
        entity_df["event_timestamp"],
        utc=True,
        errors="coerce",
    )
    entity_df = entity_df.dropna(subset=["event_timestamp"]).reset_index(drop=True)
    entity_df["hour"] = pd.to_numeric(entity_df["hour"], errors="coerce").astype("int64")

    store = FeatureStore(repo_path=str(repo_path))
    feature_df = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURE_REFS,
    ).to_df()
    feature_df.columns = feature_df.columns.astype(str).str.strip()
    feature_df["Date"] = pd.to_datetime(feature_df["date"], errors="coerce").dt.date
    feature_df["Hour"] = pd.to_numeric(feature_df["hour"], errors="coerce")
    logging.info("Loaded Feast training features with shape: %s", feature_df.shape)
    return feature_df


def resolve_target_column(dataframe: pd.DataFrame) -> str:
    """Resolve the MCP target column from possible merged dataset names."""
    for candidate in ["mcp", "MCP", "mcp_actual"]:
        if candidate in dataframe.columns:
            return candidate

    preferred_columns = [
        column
        for column in dataframe.columns
        if "MCP" in column and "Rs/MWh" in column and column.endswith("_market")
    ]
    if preferred_columns:
        return preferred_columns[0]

    fallback_columns = [
        column for column in dataframe.columns if "MCP" in column and "Rs/MWh" in column
    ]
    if fallback_columns:
        return fallback_columns[0]

    raise ValueError("No MCP target column found in the dataset.")


def prepare_model_data(
    dataframe: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    """Prepare leakage-safe feature matrix and target vector."""
    model_df = dataframe.copy()
    model_df[target_column] = pd.to_numeric(model_df[target_column], errors="coerce")

    if "event_timestamp" in model_df.columns:
        model_df["event_timestamp"] = pd.to_datetime(
            model_df["event_timestamp"],
            utc=True,
            errors="coerce",
        )
        model_df = model_df.sort_values("event_timestamp").reset_index(drop=True)
    elif {"Date", "Hour"}.issubset(model_df.columns):
        model_df["Date"] = pd.to_datetime(model_df["Date"], errors="coerce")
        model_df["Hour"] = pd.to_numeric(model_df["Hour"], errors="coerce")
        model_df = model_df.sort_values(["Date", "Hour"]).reset_index(drop=True)
    elif {"date", "hour"}.issubset(model_df.columns):
        model_df["date"] = pd.to_datetime(model_df["date"], errors="coerce")
        model_df["hour"] = pd.to_numeric(model_df["hour"], errors="coerce")
        model_df = model_df.sort_values(["date", "hour"]).reset_index(drop=True)
    else:
        logging.warning(
            "No timestamp/date-hour columns found; preserving source row order."
        )

    leakage_columns = [
        column
        for column in model_df.columns
        if "MCP" in column and column != target_column
    ]
    non_feature_columns = [
        "Date",
        "date",
        "timestamp",
        "event_timestamp",
        "time",
        "Time Block_market",
        "Time Block_weather",
    ]
    drop_columns = [target_column, *leakage_columns, *non_feature_columns]

    numeric_columns = model_df.select_dtypes(include="number").columns
    feature_columns = [
        column for column in numeric_columns if column not in set(drop_columns)
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns found after excluding target and metadata.")

    selected_columns = [target_column, *feature_columns]
    model_df = model_df[selected_columns].dropna().reset_index(drop=True)

    X = model_df[feature_columns]
    y = model_df[target_column]
    logging.info("Prepared model matrix with shape: %s", X.shape)
    return X, y, model_df, feature_columns


def time_based_train_test_split(
    X: pd.DataFrame, y: pd.Series, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data chronologically, preserving time order."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_index = int(len(X) * (1 - test_size))
    if split_index <= 0 or split_index >= len(X):
        raise ValueError("Train-test split produced an empty train or test set.")

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    logging.info("Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    colsample_bytree: float,
    random_state: int,
    eval_metric: str,
) -> XGBRegressor:
    """Train the XGBoost regression model."""
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
        eval_metric=eval_metric,
    )
    model.fit(X_train, y_train)
    logging.info("XGBoost model training complete")
    return model


def evaluate_model(y_test: pd.Series, predictions: pd.Series) -> dict[str, float]:
    """Calculate regression metrics."""
    metrics = {
        "MAPE": mean_absolute_percentage_error(y_test, predictions),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "MAE": mean_absolute_error(y_test, predictions),
        "R2": r2_score(y_test, predictions),
    }
    return metrics


def plot_feature_importance(
    model: XGBRegressor, feature_columns: list[str], output_path: Path
) -> None:
    """Save a feature importance plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    top_features = importance.head(25).sort_values("importance")
    plt.figure(figsize=(10, 9))
    sns.barplot(data=top_features, x="importance", y="feature", color="#4f6fbd")
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Saved feature importance plot: %s", output_path)


def plot_actual_vs_predicted(
    y_test: pd.Series, predictions: pd.Series, output_path: Path
) -> None:
    """Save an actual vs predicted plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = pd.DataFrame(
        {
            "actual": y_test.reset_index(drop=True),
            "predicted": pd.Series(predictions),
        }
    )
    if len(plot_df) > 5000:
        plot_df = plot_df.sample(n=5000, random_state=42)

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=plot_df, x="actual", y="predicted", alpha=0.35, s=18)
    min_value = min(plot_df["actual"].min(), plot_df["predicted"].min())
    max_value = max(plot_df["actual"].max(), plot_df["predicted"].max())
    plt.plot([min_value, max_value], [min_value, max_value], color="#b33b3b")
    plt.title("Actual vs Predicted MCP")
    plt.xlabel("Actual MCP (Rs/MWh)")
    plt.ylabel("Predicted MCP (Rs/MWh)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Saved actual vs predicted plot: %s", output_path)


def save_model(model: XGBRegressor, model_path: Path) -> None:
    """Save the trained model."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logging.info("Saved model: %s", model_path)


def build_report(
    target_column: str,
    feature_columns: list[str],
    train_rows: int,
    test_rows: int,
    metrics: dict[str, float],
    model_path: Path,
    feature_importance_path: Path,
    actual_vs_predicted_path: Path,
) -> str:
    """Build a model training report."""
    return "\n".join(
        [
            "XGBoost MCP Forecasting Report",
            "==============================",
            "",
            "Target:",
            f"- {target_column}",
            "",
            "Split:",
            "- Method: Time-based chronological train-test split",
            f"- Train rows: {train_rows}",
            f"- Test rows: {test_rows}",
            "",
            "Metrics:",
            f"- MAPE: {metrics['MAPE']:.6f}",
            f"- RMSE: {metrics['RMSE']:.6f}",
            f"- MAE: {metrics['MAE']:.6f}",
            f"- R2: {metrics['R2']:.6f}",
            "",
            "Features:",
            f"- Feature count: {len(feature_columns)}",
            "- MCP leakage columns were excluded from predictors.",
            "",
            "Artifacts:",
            f"- Model: {model_path}",
            f"- Feature importance plot: {feature_importance_path}",
            f"- Actual vs predicted plot: {actual_vs_predicted_path}",
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an XGBoost regression model for MCP forecasting."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to training data (.parquet or .csv).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path where the trained model should be saved.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path where the model report should be saved.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="Directory where model plots should be saved.",
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--subsample", type=float, default=DEFAULT_SUBSAMPLE)
    parser.add_argument(
        "--colsample-bytree", type=float, default=DEFAULT_COLSAMPLE_BYTREE
    )
    parser.add_argument("--eval-metric", default=DEFAULT_EVAL_METRIC)
    parser.add_argument(
        "--mlflow-experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI or local SQLite.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=DEFAULT_MLFLOW_RUN_NAME,
        help="MLflow run name.",
    )
    parser.add_argument(
        "--registered-model-name",
        default=DEFAULT_MODEL_REGISTRY_NAME,
        help="MLflow Model Registry name.",
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Skip MLflow experiment tracking and model registration.",
    )
    parser.add_argument(
        "--promote-stage",
        default="Production",
        help="MLflow registry stage for the newest model version. Use empty string to skip.",
    )
    parser.add_argument(
        "--use-feast",
        action="store_true",
        help="Load training features from Feast instead of --input-path.",
    )
    parser.add_argument(
        "--feast-repo-path",
        type=Path,
        default=DEFAULT_FEAST_REPO_PATH,
        help="Path to the Feast feature repository.",
    )
    parser.add_argument(
        "--feast-entity-path",
        type=Path,
        default=DEFAULT_FEAST_ENTITY_PATH,
        help="CSV entity dataframe used for Feast historical retrieval.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the XGBoost modeling pipeline."""
    configure_logging()
    args = parse_args()

    feature_importance_path = args.plot_dir / "feature_importance.png"
    actual_vs_predicted_path = args.plot_dir / "actual_vs_predicted.png"
    lineage_inputs = [args.feast_entity_path] if args.use_feast else [args.input_path]
    lineage_outputs = [
        args.model_path,
        args.report_path,
        feature_importance_path,
        actual_vs_predicted_path,
    ]

    with lineage_run(
        "model_training",
        inputs=lineage_inputs,
        outputs=lineage_outputs,
        metadata={
            "stage": "model_training",
            "model_family": "xgboost",
            "use_feast": args.use_feast,
            "mlflow_enabled": not args.disable_mlflow,
            "registered_model_name": args.registered_model_name,
        },
    ):
        if args.use_feast:
            dataframe = load_dataset_from_feast(args.feast_repo_path, args.feast_entity_path)
        else:
            dataframe = load_dataset(args.input_path)
        target_column = resolve_target_column(dataframe)
        X, y, _, feature_columns = prepare_model_data(dataframe, target_column)
        X_train, X_test, y_train, y_test = time_based_train_test_split(
            X, y, args.test_size
        )

        mlflow_run_id = None
        promoted_model_version = None
        if not args.disable_mlflow:
            setup_mlflow(
                experiment_name=args.mlflow_experiment_name,
                tracking_uri=args.mlflow_tracking_uri,
            )
            with mlflow.start_run(run_name=args.mlflow_run_name) as run:
                model = train_model(
                    X_train=X_train,
                    y_train=y_train,
                    n_estimators=args.n_estimators,
                    learning_rate=args.learning_rate,
                    max_depth=args.max_depth,
                    subsample=args.subsample,
                    colsample_bytree=args.colsample_bytree,
                    random_state=args.random_state,
                    eval_metric=args.eval_metric,
                )
                predictions = model.predict(X_test)
                prediction_series = pd.Series(predictions)
                metrics = evaluate_model(y_test, prediction_series)

                plot_feature_importance(model, feature_columns, feature_importance_path)
                plot_actual_vs_predicted(
                    y_test, prediction_series, actual_vs_predicted_path
                )
                save_model(model, args.model_path)

                mlflow.log_params(
                    {
                        "max_depth": args.max_depth,
                        "learning_rate": args.learning_rate,
                        "n_estimators": args.n_estimators,
                        "train_size": round(1 - args.test_size, 6),
                    }
                )
                mlflow.log_metrics(
                    {
                        "RMSE": float(metrics["RMSE"]),
                        "MAE": float(metrics["MAE"]),
                        "MAPE": float(metrics["MAPE"]),
                        "R2": float(metrics["R2"]),
                    }
                )
                log_xgboost_artifacts(
                    {
                        "feature_importance": feature_importance_path,
                        "actual_vs_predicted": actual_vs_predicted_path,
                    }
                )
                registered_model_version = log_xgboost_model(
                    model=model,
                    X_test=X_test,
                    predictions=prediction_series,
                    registered_model_name=args.registered_model_name,
                )
                promoted_model_version = registered_model_version
                mlflow.set_tags(
                    {
                        "project": "IEX Power Trading Risk Optimization",
                        "model_family": "xgboost",
                        "task": "mcp_forecasting",
                        "train_rows": str(len(X_train)),
                        "test_rows": str(len(X_test)),
                    }
                )
                mlflow_run_id = run.info.run_id
                logging.info(
                    "Logged MLflow run %s and registered model as %s",
                    mlflow_run_id,
                    args.registered_model_name,
                )
            if args.promote_stage:
                promoted_model_version = promote_latest_model_version(
                    args.registered_model_name,
                    args.promote_stage,
                    tracking_uri=args.mlflow_tracking_uri,
                )
        else:
            model = train_model(
                X_train=X_train,
                y_train=y_train,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                random_state=args.random_state,
                eval_metric=args.eval_metric,
            )
            predictions = model.predict(X_test)
            prediction_series = pd.Series(predictions)
            metrics = evaluate_model(y_test, prediction_series)
            plot_feature_importance(model, feature_columns, feature_importance_path)
            plot_actual_vs_predicted(y_test, prediction_series, actual_vs_predicted_path)
            save_model(model, args.model_path)

        report = build_report(
            target_column=target_column,
            feature_columns=feature_columns,
            train_rows=len(X_train),
            test_rows=len(X_test),
            metrics=metrics,
            model_path=args.model_path,
            feature_importance_path=feature_importance_path,
            actual_vs_predicted_path=actual_vs_predicted_path,
        )
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        if mlflow_run_id:
            report = "\n".join(
                [
                    report,
                    "",
                    "MLflow:",
                    f"- Experiment: {args.mlflow_experiment_name}",
                    f"- Run ID: {mlflow_run_id}",
                    f"- Registered model: {args.registered_model_name}",
                    f"- Promoted version: {promoted_model_version}",
                    f"- Stage: {args.promote_stage}",
                ]
            )
        args.report_path.write_text(report, encoding="utf-8")

        print(report)
        logging.info("Saved model report: %s", args.report_path)


if __name__ == "__main__":
    main()
