"""
transit/pipeline.py — incremental data sync into DuckDB

Orchestrates fetching from all sources and upserting into a local DuckDB
database. Each table stores only new rows beyond the current MAX(timestamp).

Tables managed
--------------
delays   — SL real-time departure delays (polled continuously)
weather  — Open-Meteo hourly actuals (back-filled on first run, then daily)

Usage
-----
  python -m transit.pipeline          # one-shot sync
  python -m transit.pipeline --init   # force full re-sync from 90 days ago
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone

import duckdb
import pandas as pd

from transit.data_sources import (
    DEFAULT_SITE_IDS,
    fetch_sl_departures,
    fetch_weather_archive,
    fetch_weather_forecast,
)

log = logging.getLogger(__name__)

DB_PATH   = os.getenv("TRAFFIK_DB_PATH", "data/traffik.duckdb")
BACKFILL_DAYS = 90   # history window on first run


# ── schema ─────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS delays (
    fetched_at       TIMESTAMPTZ NOT NULL,
    site_id          INTEGER     NOT NULL,
    line_id          TEXT,
    line_name        TEXT,
    transport_mode   TEXT,
    direction        TEXT,
    destination      TEXT,
    scheduled        TIMESTAMPTZ,
    expected         TIMESTAMPTZ,
    delay_minutes    DOUBLE
);

CREATE TABLE IF NOT EXISTS weather (
    timestamp    TIMESTAMPTZ NOT NULL PRIMARY KEY,
    temperature  DOUBLE,
    wind_speed   DOUBLE,
    precipitation DOUBLE,
    snowfall     DOUBLE,
    cloud_cover  DOUBLE
);

CREATE TABLE IF NOT EXISTS predictions (
    timestamp      TIMESTAMPTZ NOT NULL,
    site_id        INTEGER,
    line_id        TEXT,
    pred_delay     DOUBLE,
    actual_delay   DOUBLE,
    model_version  TEXT
);

CREATE TABLE IF NOT EXISTS retrain_log (
    run_at      TIMESTAMPTZ NOT NULL,
    new_rows    INTEGER,
    retrained   BOOLEAN,
    mae         DOUBLE
);

-- Latest snapshot per departure — used by training to get true final delay.
-- With frequent polling, a departure appears many times; the last fetch before
-- it departs is the most accurate delay reading.
CREATE OR REPLACE VIEW latest_delays AS
SELECT DISTINCT ON (site_id, line_id, scheduled)
    fetched_at, site_id, line_id, line_name,
    transport_mode, direction, destination,
    scheduled, expected, delay_minutes
FROM delays
ORDER BY site_id, line_id, scheduled, fetched_at DESC;
"""


def _init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_DDL)


# ── helpers ────────────────────────────────────────────────────────────────────

def _max_ts(con: duckdb.DuckDBPyConnection, table: str, col: str = "timestamp") -> datetime | None:
    """Return the latest timestamp in *table*, or None if the table is empty."""
    try:
        row = con.execute(f"SELECT MAX({col}) FROM {table}").fetchone()
        ts = row[0]
        if ts is None:
            return None
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            return ts.astimezone(timezone.utc).replace(tzinfo=None)
        return ts
    except Exception:
        return None


def _row_count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# ── sync: delays ───────────────────────────────────────────────────────────────

def sync_delays(con: duckdb.DuckDBPyConnection) -> int:
    """
    Fetch a snapshot of SL departures and append to the delays table.

    Delays are append-only (each poll is a new snapshot row); we do not
    de-duplicate because the same departure may appear at different lag times.

    Returns
    -------
    int
        Number of rows inserted.
    """
    df = fetch_sl_departures(site_ids=DEFAULT_SITE_IDS)
    if df.empty:
        log.warning("sync_delays: no rows returned from SL API")
        return 0

    # DuckDB requires TIMESTAMPTZ — attach UTC zone info
    for col in ("scheduled", "expected", "fetched_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)

    before = _row_count(con, "delays")
    con.register("_delays_stage", df)
    con.execute("""
        INSERT INTO delays
        SELECT
            fetched_at, site_id, line_id, line_name,
            transport_mode, direction, destination,
            scheduled, expected, delay_minutes
        FROM _delays_stage
    """)
    con.unregister("_delays_stage")
    inserted = _row_count(con, "delays") - before
    log.info("sync_delays: inserted %d rows (total %d)", inserted, before + inserted)
    return inserted


# ── sync: weather ──────────────────────────────────────────────────────────────

def sync_weather(con: duckdb.DuckDBPyConnection, force_full: bool = False) -> int:
    """
    Upsert weather rows into the weather table.

    On first run (empty table) back-fills BACKFILL_DAYS of history.
    Subsequent calls fetch from the latest stored timestamp to today,
    then append the 48-hour forecast.

    Returns
    -------
    int
        Number of new rows written.
    """
    latest = None if force_full else _max_ts(con, "weather")

    if latest is None:
        start = date.today() - timedelta(days=BACKFILL_DAYS)
        log.info("sync_weather: back-filling from %s", start)
    else:
        # resume from latest stored row
        start = (latest + timedelta(hours=1)).date()
        log.info("sync_weather: incremental from %s", start)

    end_archive = date.today()
    df_arch = fetch_weather_archive(start, end_archive)
    df_fcst = fetch_weather_forecast()

    frames = [df for df in (df_arch, df_fcst) if not df.empty]
    if not frames:
        log.warning("sync_weather: no data from archive or forecast")
        return 0

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    before = _row_count(con, "weather")
    con.register("_weather_stage", df)
    con.execute("""
        INSERT OR REPLACE INTO weather
        SELECT timestamp, temperature, wind_speed, precipitation, snowfall, cloud_cover
        FROM _weather_stage
    """)
    con.unregister("_weather_stage")
    after = _row_count(con, "weather")
    inserted = after - before
    log.info("sync_weather: %d rows now in table (%+d)", after, inserted)
    return inserted


# ── main entry point ───────────────────────────────────────────────────────────

def run_pipeline(force_full: bool = False) -> dict[str, int]:
    """
    Run one full sync cycle: delays + weather.

    Parameters
    ----------
    force_full : bool
        If True, ignore existing data and re-fetch from scratch.

    Returns
    -------
    dict[str, int]
        Keys: "delays", "weather" — rows inserted per table.
    """
    import pathlib
    pathlib.Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(DB_PATH)
    _init_db(con)

    log.info("=== pipeline sync start ===")

    delays_new  = sync_delays(con)
    weather_new = sync_weather(con, force_full=force_full)

    con.close()
    log.info("=== pipeline sync done: delays=%d weather=%d ===", delays_new, weather_new)
    return {"delays": delays_new, "weather": weather_new}


if __name__ == "__main__":
    import argparse
    import sys

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Sync transit + weather data into DuckDB")
    parser.add_argument("--init", action="store_true", help="Force full re-sync")
    args = parser.parse_args()

    result = run_pipeline(force_full=args.init)
    print(f"\nSync complete: {result}")
    sys.exit(0)
