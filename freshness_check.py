"""Data freshness check for the transit pipeline.

Call check_freshness() before running any retrain or prediction. Returns True
if all sources are fresh, False if any are stale (and sends an alert if configured).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import duckdb

DB_PATH = os.getenv("TRAFFIK_DB_PATH", "data/traffik.duckdb")

# Transit data is denser than electricity prices — 2h threshold is appropriate.
THRESHOLDS = {
    "delays":  timedelta(hours=2),
    "weather": timedelta(hours=6),
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def check_freshness(send_alert_fn=None) -> bool:
    """Return True if all data sources are fresh, False if any are stale.

    Parameters
    ----------
    send_alert_fn : callable(str) | None
        Optional alert function (e.g. send_slack_alert). Called with a message
        string when stale data is detected.
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    now = _utcnow()
    stale: list[str] = []

    ts_cols = {
        "delays":  "fetched_at",
        "weather": "timestamp",
    }

    for table, threshold in THRESHOLDS.items():
        col = ts_cols.get(table, "timestamp")
        try:
            row = con.execute(
                f"SELECT MAX({col}) as latest FROM {table}"
            ).fetchone()
            latest = row[0]

            if latest is None:
                stale.append(f"  {table}: table is empty")
                continue

            if hasattr(latest, "tzinfo") and latest.tzinfo is not None:
                latest = latest.astimezone(timezone.utc).replace(tzinfo=None)

            lag = now - latest
            if lag > threshold:
                stale.append(
                    f"  {table}: {lag.total_seconds() / 3600:.1f}h stale "
                    f"(latest row: {latest.strftime('%Y-%m-%d %H:%M')} UTC)"
                )
            else:
                print(f"  {table}: fresh ({lag.total_seconds() / 3600:.1f}h lag)")

        except Exception as exc:
            stale.append(f"  {table}: query failed - {exc}")

    con.close()

    if stale:
        msg = "Traffik Pipeline Freshness Alert:\n" + "\n".join(stale)
        print(msg)
        if send_alert_fn:
            send_alert_fn(msg)
        return False

    print("  All data sources are fresh")
    return True


if __name__ == "__main__":
    ok = check_freshness()
    raise SystemExit(0 if ok else 1)
