"""Lightweight local metrics logger. Used alongside MLflow as an append-only audit trail."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

METRICS_FILE = Path("model_metrics.jsonl")


def log_metrics(model_name: str, metrics: dict, extra: dict | None = None) -> None:
    """Append one line to model_metrics.jsonl.

    Parameters
    ----------
    model_name : str
        e.g. "delay_model"
    metrics : dict
        Float values — MAE, RMSE, etc.
    extra : dict | None
        Optional metadata — n_rows, feature_count, data_date_range, etc.
    """
    entry = {
        "timestamp":  datetime.utcnow().isoformat(),
        "model_name": model_name,
        "metrics":    {k: float(v) for k, v in metrics.items()},
        "meta":       extra or {},
    }
    with open(METRICS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[metrics_logger] {model_name}: {metrics}")


def tail(n: int = 10) -> list[dict]:
    """Return the last n entries."""
    if not METRICS_FILE.exists():
        return []
    lines = METRICS_FILE.read_text().strip().splitlines()
    return [json.loads(line) for line in lines[-n:]]
