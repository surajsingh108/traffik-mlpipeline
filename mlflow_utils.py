"""MLflow helpers for transit delay model training runs.

Wraps MLflow calls so training scripts don't import mlflow directly —
easier to swap tracking backends later.

Backend
-------
Local dev  : sqlite:///mlflow.db  (default; set MLFLOW_TRACKING_URI to override)
Azure prod : wasbs://mlflow@<account>.blob.core.windows.net/mlruns
"""
from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import mlflow


def setup_mlflow() -> None:
    """Configure MLflow from environment. Call once at start of training script.

    Defaults to a local SQLite DB (mlflow.db) if MLFLOW_TRACKING_URI is not set.
    MLflow 3.x dropped the file-store backend; SQLite is the simplest local alternative.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[mlflow] Tracking URI: {tracking_uri}")


def log_training_run(
    experiment_name: str,
    run_name: str,
    params: dict,
    metrics: dict,
    model_obj: object,
    model_filename: str,
    tags: dict | None = None,
) -> str:
    """Log one complete training run to MLflow.

    Parameters
    ----------
    experiment_name : str
        e.g. "delay_model"
    run_name : str
        e.g. "20260702_120000"
    params : dict
        Hyperparameters to log.
    metrics : dict
        Performance metrics (test_mae, test_rmse, etc.).
    model_obj : object
        Trained model or artifacts dict to pickle as an artifact.
    model_filename : str
        Filename for the pickle artifact, e.g. "delay_model.pkl".
    tags : dict | None
        Optional extra tags for the run.

    Returns
    -------
    str
        MLflow run ID.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / model_filename
            with open(tmp_path, "wb") as f:
                pickle.dump(model_obj, f)
            mlflow.log_artifact(str(tmp_path), artifact_path="model")

        run_id = run.info.run_id
        print(f"[mlflow] Logged run {run_id} for experiment '{experiment_name}'")
        return run_id


def get_latest_run_metrics(experiment_name: str) -> dict | None:
    """Fetch the metrics of the latest run for a given experiment."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return None
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    return runs[0].data.metrics
