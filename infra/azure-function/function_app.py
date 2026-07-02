"""
Azure Function App — timer triggers for the transit pipeline.

Triggers
--------
retrain_trigger : runs every 3 hours  (cron: "0 0 */3 * * *")
    Calls transit/retrain.py — syncs data, conditionally retrains, logs to MLflow.

digest_trigger  : runs every Monday at 08:00 Stockholm time  (cron: "0 0 8 * * 1")
    Calls agent/digest.py — generates weekly LLM summary and stores it in DuckDB.

Deploy
------
  func azure functionapp publish traffik-ingest
  (requires Azure Functions Core Tools: npm install -g azure-functions-core-tools@4)

Environment variables (set in Azure Function App settings or local.settings.json)
----------------------------------------------------------------------------------
  TRAFFIK_DB_PATH
  TRAFIKLAB_API_KEY
  GROQ_API_KEY
  AZURE_STORAGE_CONNECTION_STRING
  MLFLOW_TRACKING_URI
  SLACK_WEBHOOK          (optional)
"""
from __future__ import annotations

import logging
import azure.functions as func

app = func.FunctionApp()

log = logging.getLogger(__name__)


@app.timer_trigger(
    schedule="0 0 */3 * * *",       # every 3 hours
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True,
)
def retrain_trigger(timer: func.TimerRequest) -> None:
    """Sync data + conditionally retrain the delay model."""
    if timer.past_due:
        log.warning("retrain_trigger: timer is past due — running now")

    log.info("retrain_trigger: starting pipeline + retrain cycle")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from transit.retrain import main as retrain_main
    retrain_main(force=False)
    log.info("retrain_trigger: complete")


@app.timer_trigger(
    schedule="0 0 8 * * 1",         # every Monday 08:00 UTC
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True,
)
def digest_trigger(timer: func.TimerRequest) -> None:
    """Generate and store the weekly LLM delay digest."""
    if timer.past_due:
        log.warning("digest_trigger: timer is past due — running now")

    log.info("digest_trigger: generating weekly digest")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    try:
        from agent.digest import generate_and_store
        generate_and_store()
        log.info("digest_trigger: complete")
    except Exception as exc:
        log.error("digest_trigger: failed: %s", exc)
        from slack_alert import send_slack_alert
        send_slack_alert(f"Weekly digest generation failed: {exc}")
