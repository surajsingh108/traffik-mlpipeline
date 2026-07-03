"""
transit/ml.py — LightGBM delay regression with champion/challenger promotion

Trains a LightGBM regressor to predict delay_minutes.
Uses a rolling 7-day holdout for evaluation.
Promotes the new model only if it beats the current champion by ≥2% MAE.

Functions
---------
train(con)               -> dict   Train on all historical data; return metrics.
evaluate(model, X, y)    -> dict   Compute MAE, RMSE on a held-out set.
load_champion()          -> object | None   Load current champion pkl, or None.
save_champion(model, metrics, config)       Atomic write to model/.
champion_metrics()       -> dict | None     Read last saved metrics.json.

Usage
-----
  python -m transit.ml          # train + evaluate + promote if better
  python -m transit.ml --force  # always promote regardless of metrics
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODEL_DIR  = Path(os.getenv("MODEL_DIR", "model"))
MODEL_FILE = MODEL_DIR / "delay_model.pkl"
CONFIG_FILE = MODEL_DIR / "config.json"
METRICS_FILE = MODEL_DIR / "metrics.json"

# Champion is promoted only if new MAE is this much better
PROMOTION_THRESHOLD = 0.98   # new_mae < champion_mae * 0.98

HOLDOUT_DAYS = 7

_LGBM_PARAMS = {
    "objective":       "regression_l1",   # MAE
    "n_estimators":    400,
    "learning_rate":   0.05,
    "num_leaves":      63,
    "min_child_samples": 10,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "n_jobs":          -1,
    "verbose":         -1,
}


# ── lazy LightGBM import ───────────────────────────────────────────────────────

try:
    import lightgbm as lgb
except ModuleNotFoundError as _err:
    lgb = None  # type: ignore[assignment]
    _LGB_MISSING = _err


def _require_lgb() -> None:
    if lgb is None:
        raise ModuleNotFoundError("Install with: pip install lightgbm") from _LGB_MISSING


# ── atomic save ───────────────────────────────────────────────────────────────

def _atomic_write(path: Path, data: bytes) -> None:
    """Write bytes to path via a temp file to avoid partial writes."""
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


# ── champion I/O ───────────────────────────────────────────────────────────────

def load_champion() -> object | None:
    """Load the current champion model, or None if no model has been saved yet."""
    if not MODEL_FILE.exists():
        return None
    try:
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        log.warning("load_champion: failed to load %s: %s", MODEL_FILE, exc)
        return None


def champion_metrics() -> dict | None:
    """Return the saved metrics of the current champion, or None."""
    if not METRICS_FILE.exists():
        return None
    try:
        return json.loads(METRICS_FILE.read_text())
    except Exception:
        return None


def save_champion(model: object, metrics: dict, config: dict) -> None:
    """Atomically save model + metrics + encoding config to model/."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write(MODEL_FILE,  pickle.dumps(model))
    _atomic_write(METRICS_FILE, json.dumps(metrics, indent=2).encode())
    _atomic_write(CONFIG_FILE,  json.dumps(config,  indent=2).encode())
    log.info("Saved champion: MAE=%.3f → %s", metrics.get("test_mae", 0), MODEL_FILE)


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Compute MAE and RMSE for a fitted model on held-out data.

    Parameters
    ----------
    model : fitted LGBMRegressor
    X : pd.DataFrame  Feature matrix.
    y : pd.Series     True delay_minutes.

    Returns
    -------
    dict
        Keys: test_mae, test_rmse, n_test.
    """
    preds = model.predict(X)
    errors = y.values - preds
    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return {"test_mae": mae, "test_rmse": rmse, "n_test": int(len(y))}


# ── training ───────────────────────────────────────────────────────────────────

def train(con, force: bool = False) -> dict:
    """
    Train a LightGBM delay regressor and apply champion/challenger logic.

    Uses the last HOLDOUT_DAYS days as an evaluation set.
    Promotes the new model if it beats the champion by ≥2% on test MAE,
    or if no champion exists, or if force=True.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open connection to the DuckDB database.
    force : bool
        If True, always promote the new model regardless of metrics.

    Returns
    -------
    dict
        Training result: metrics, promoted (bool), champion_mae (float | None).
    """
    _require_lgb()

    import transit.features as _feat
    from transit.features import build_features, make_feature_cols

    log.info("Building features for training …")
    df = build_features(con, mode="train", min_rows=20)
    feat_cols = make_feature_cols(df)

    # Read maps from the module AFTER build_features() has repopulated them.
    # Importing by name above would snapshot the empty dicts at import time.
    TRANSPORT_MODE_MAP = _feat.TRANSPORT_MODE_MAP
    LINE_ID_MAP        = _feat.LINE_ID_MAP
    SITE_ID_MAP        = _feat.SITE_ID_MAP

    if not feat_cols:
        raise RuntimeError("No feature columns available — check features.py")

    # Train/test split: last HOLDOUT_DAYS are the holdout
    cutoff = pd.to_datetime(df["scheduled"], utc=True).max() - timedelta(days=HOLDOUT_DAYS)
    scheduled_utc = pd.to_datetime(df["scheduled"], utc=True)
    train_mask = scheduled_utc <= cutoff
    test_mask  = scheduled_utc >  cutoff

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        log.warning(
            "Time-based split degenerate (train=%d test=%d) — falling back to 80/20 row split",
            train_mask.sum(), test_mask.sum(),
        )
        split_idx  = max(1, int(len(df) * 0.8))
        train_mask = pd.Series([True]  * split_idx + [False] * (len(df) - split_idx), index=df.index)
        test_mask  = ~train_mask

    X_train = df.loc[train_mask, feat_cols]
    y_train = df.loc[train_mask, "delay_minutes"]
    X_test  = df.loc[test_mask,  feat_cols]
    y_test  = df.loc[test_mask,  "delay_minutes"]

    log.info(
        "Training on %d rows, evaluating on %d rows (%d features)",
        len(X_train), len(X_test), len(feat_cols),
    )

    model = lgb.LGBMRegressor(**_LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )

    metrics = evaluate(model, X_test, y_test)
    metrics["n_train"]    = int(train_mask.sum())
    metrics["n_features"] = len(feat_cols)
    log.info("New model: MAE=%.3f RMSE=%.3f", metrics["test_mae"], metrics["test_rmse"])

    # Champion/challenger decision
    champ_m = champion_metrics()
    champ_mae = champ_m["test_mae"] if champ_m else None

    if force:
        promoted = True
        reason = "force=True"
    elif champ_mae is None:
        promoted = True
        reason = "no existing champion"
    elif metrics["test_mae"] < champ_mae * PROMOTION_THRESHOLD:
        promoted = True
        reason = f"new MAE {metrics['test_mae']:.3f} < {champ_mae * PROMOTION_THRESHOLD:.3f} (champion * {PROMOTION_THRESHOLD})"
    else:
        promoted = False
        reason = f"new MAE {metrics['test_mae']:.3f} did not beat champion {champ_mae:.3f} * {PROMOTION_THRESHOLD}"

    log.info("Promotion decision: %s → %s", "PROMOTED" if promoted else "REJECTED", reason)

    if promoted:
        if not TRANSPORT_MODE_MAP or not LINE_ID_MAP or not SITE_ID_MAP:
            log.warning(
                "Skipping champion save — encoding maps are empty "
                "(TRANSPORT_MODE_MAP=%d, LINE_ID_MAP=%d, SITE_ID_MAP=%d). "
                "Re-run build_features(fit=True) against a populated DB.",
                len(TRANSPORT_MODE_MAP), len(LINE_ID_MAP), len(SITE_ID_MAP),
            )
            promoted = False
            reason = "encoding maps empty — champion not saved"
        else:
            config = {
                "feature_cols":        feat_cols,
                "transport_mode_map":  TRANSPORT_MODE_MAP,
                "line_id_map":         LINE_ID_MAP,
                "site_id_map":         {str(k): v for k, v in SITE_ID_MAP.items()},
                "lgbm_params":         _LGBM_PARAMS,
                "holdout_days":        HOLDOUT_DAYS,
            }
            save_champion(model, metrics, config)

    return {
        "metrics":      metrics,
        "promoted":     promoted,
        "reason":       reason,
        "champion_mae": champ_mae,
    }


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    try:
        from dotenv import load_dotenv; load_dotenv()
    except ImportError:
        pass
    import duckdb
    from metrics_logger import log_metrics
    from mlflow_utils import setup_mlflow, log_training_run

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Always promote new model")
    args = parser.parse_args()

    db_path = os.getenv("TRAFFIK_DB_PATH", "data/traffik.duckdb")
    con = duckdb.connect(db_path, read_only=True)

    setup_mlflow()
    result = train(con, force=args.force)
    con.close()

    m = result["metrics"]
    log_metrics("delay_model", {"test_mae": m["test_mae"], "test_rmse": m["test_rmse"]},
                extra={"n_train": m["n_train"], "n_test": m["n_test"], "promoted": result["promoted"]})

    if result["promoted"]:
        from datetime import datetime
        champ = load_champion()
        log_training_run(
            experiment_name="delay_model",
            run_name=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            params={"n_features": m["n_features"], "holdout_days": HOLDOUT_DAYS, **_LGBM_PARAMS},
            metrics={"test_mae": m["test_mae"], "test_rmse": m["test_rmse"]},
            model_obj=champ,
            model_filename="delay_model.pkl",
            tags={"promoted": "true", "reason": result["reason"]},
        )

    print(f"\n{'PROMOTED' if result['promoted'] else 'REJECTED'}: {result['reason']}")
    print(f"  MAE={m['test_mae']:.3f}  RMSE={m['test_rmse']:.3f}  n_test={m['n_test']}")
    sys.exit(0)
