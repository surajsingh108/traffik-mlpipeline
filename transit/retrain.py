"""
transit/retrain.py — conditional refresh orchestrator

Mirrors the SE3 retrain.py step order:
  1. Record pre-sync row counts
  2. backup_all() → Azure Blob
  3. Sync data (pipeline.py)
  4. check_freshness() → exit 0 if stale
  5. Decide whether to retrain (new_rows > threshold)
  6. build_features() → train() → evaluate() → champion/challenger
  7. log_metrics() + log_training_run() (MLflow)
  8. Write retrain_log entry to DuckDB

Called by the Azure Function timer trigger every 3 hours.

Usage
-----
  python -m transit.retrain            # normal run
  python -m transit.retrain --force    # force retrain regardless of new row count
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import duckdb

log = logging.getLogger(__name__)

DB_PATH           = os.getenv("TRAFFIK_DB_PATH", "data/traffik.duckdb")
RETRAIN_MIN_ROWS  = int(os.getenv("RETRAIN_MIN_ROWS", "50"))   # new rows needed to trigger retrain


def _row_count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _write_retrain_log(
    con: duckdb.DuckDBPyConnection,
    new_rows: int,
    retrained: bool,
    mae: float | None,
) -> None:
    con.execute("""
        INSERT INTO retrain_log (run_at, new_rows, retrained, mae)
        VALUES (?, ?, ?, ?)
    """, [datetime.now(timezone.utc), new_rows, retrained, mae])


def _main(force: bool = False) -> None:
    from model_backup import backup_all
    from freshness_check import check_freshness
    from slack_alert import send_slack_alert
    from metrics_logger import log_metrics
    from mlflow_utils import setup_mlflow, log_training_run
    from transit.pipeline import run_pipeline, _row_count as pipeline_count
    from transit.ml import train, load_champion, champion_metrics, _LGBM_PARAMS, HOLDOUT_DAYS

    setup_mlflow()

    con = duckdb.connect(DB_PATH)

    # ── 1. Pre-sync counts ─────────────────────────────────────────────────────
    pre_delays  = _row_count(con, "delays")
    pre_weather = _row_count(con, "weather")
    log.info("Pre-sync: delays=%d  weather=%d", pre_delays, pre_weather)

    # ── 2. Model backup ────────────────────────────────────────────────────────
    try:
        backup_all()
    except Exception as exc:
        log.warning("Model backup failed (non-fatal): %s", exc)

    # ── 3. Sync data ───────────────────────────────────────────────────────────
    sync_result = run_pipeline()
    new_delays  = sync_result["delays"]
    log.info("Sync complete: +%d delays  +%d weather", new_delays, sync_result["weather"])

    # ── 4. Freshness check ─────────────────────────────────────────────────────
    # Close the write connection before freshness check opens its own read-only
    # connection — DuckDB does not allow concurrent connections with mixed modes.
    con.close()
    if not check_freshness(send_alert_fn=send_slack_alert):
        log.warning("Stale data detected — skipping retrain")
        raise SystemExit(0)
    con = duckdb.connect(DB_PATH)

    # ── 5. Retrain decision ────────────────────────────────────────────────────
    total_delays = _row_count(con, "delays")
    should_retrain = force or (new_delays >= RETRAIN_MIN_ROWS)

    if not should_retrain:
        log.info(
            "Skipping retrain: only %d new delay rows (threshold %d)",
            new_delays, RETRAIN_MIN_ROWS,
        )
        _write_retrain_log(con, new_delays, retrained=False, mae=None)
        con.close()
        return

    log.info(
        "Retraining: %d new rows (threshold %d), total delays=%d",
        new_delays, RETRAIN_MIN_ROWS, total_delays,
    )

    # ── 6. Train + champion/challenger ─────────────────────────────────────────
    result = train(con, force=force)
    m      = result["metrics"]
    mae    = m["test_mae"]

    # ── 7. Log metrics + MLflow ────────────────────────────────────────────────
    log_metrics(
        "delay_model",
        {"test_mae": mae, "test_rmse": m["test_rmse"]},
        extra={
            "n_train":   m["n_train"],
            "n_test":    m["n_test"],
            "promoted":  result["promoted"],
            "new_rows":  new_delays,
        },
    )

    if result["promoted"]:
        champion = load_champion()
        log_training_run(
            experiment_name="delay_model",
            run_name=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            params={
                "n_features":  m["n_features"],
                "holdout_days": HOLDOUT_DAYS,
                **_LGBM_PARAMS,
            },
            metrics={"test_mae": mae, "test_rmse": m["test_rmse"]},
            model_obj=champion,
            model_filename="delay_model.pkl",
            tags={"promoted": "true", "reason": result["reason"]},
        )

    # ── 8. Write retrain log ───────────────────────────────────────────────────
    _write_retrain_log(con, new_delays, retrained=True, mae=mae)
    con.close()

    log.info(
        "Retrain complete: MAE=%.3f  promoted=%s  reason=%s",
        mae, result["promoted"], result["reason"],
    )


def main(force: bool = False) -> None:
    """Entry point — wraps _main() with top-level exception alerting."""
    from slack_alert import send_slack_alert

    start = datetime.utcnow()
    try:
        _main(force=force)
    except SystemExit:
        raise
    except Exception as exc:
        elapsed = (datetime.utcnow() - start).seconds
        send_slack_alert(
            f"Traffik Retrain Job FAILED\n"
            f"Error: {str(exc)[:300]}\n"
            f"Runtime: {elapsed}s\n"
            f"Time (UTC): {start.strftime('%Y-%m-%d %H:%M')}"
        )
        raise


if __name__ == "__main__":
    import argparse, sys
    try:
        from dotenv import load_dotenv; load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force retrain even with few new rows")
    args = parser.parse_args()

    main(force=args.force)
    sys.exit(0)
