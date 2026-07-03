"""
transit/data_sources.py — SL real-time departures + Open-Meteo weather

Fetches and normalises data from two external sources:
  1. Trafiklab SL Transport v1 — real-time departures, computes delay_minutes
  2. Open-Meteo archive + forecast — weather actuals and near-term forecast

Functions
---------
fetch_sl_departures(site_ids, api_key) -> DataFrame
    Snapshot of current delays at the given SL stop site IDs.

fetch_weather_archive(start, end) -> DataFrame
    Historical hourly weather from Open-Meteo archive API.

fetch_weather_forecast() -> DataFrame
    48-hour Open-Meteo weather forecast for Stockholm.

Usage
-----
  python -m transit.data_sources
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Sequence

import pandas as pd
import requests

log = logging.getLogger(__name__)

# Stockholm city centre coordinates
WEATHER_LAT = 59.33
WEATHER_LON = 18.07

SL_BASE_URL = "https://transport.integration.sl.se/v1"

# Major SL sites to monitor (site IDs from the SL Transport API).
# Add more with: GET /v1/sites?expand=false&q=<name>
DEFAULT_SITE_IDS: list[int] = [
    9001,   # T-Centralen (metro hub)
    9180,   # Slussen
    9117,   # Fridhemsplan
    9192,   # Gullmarsplan
    9261,   # Odenplan
    9530,   # Liljeholmen
    9306,   # Solna centrum
    9325,   # Sundbyberg
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _to_date(d) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, (pd.Timestamp, datetime)):
        return d.date()
    return date.fromisoformat(str(d)[:10])


def _ok(name: str, df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    print(f"  OK  {name:<30}: {n:>6,} rows")
    return df


def _fail(name: str, exc: Exception) -> pd.DataFrame:
    log.warning("FAIL %s: %s: %s (returning empty)", name, type(exc).__name__, exc)
    return pd.DataFrame()


def _parse_iso(ts_str: str | None) -> datetime | None:
    """Parse an ISO-8601 string → tz-aware datetime, or None."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


# ── 1. SL real-time departures ─────────────────────────────────────────────────

def _fetch_site_departures(site_id: int, api_key: str, forecast: int = 60) -> pd.DataFrame:
    """
    Fetch upcoming departures for one SL site.

    Parameters
    ----------
    site_id : int
        SL Transport API site identifier.
    api_key : str
        Trafiklab API subscription key.
    forecast : int
        Look-ahead window in minutes (default 60).

    Returns
    -------
    pd.DataFrame
        One row per departure with columns: site_id, line_id, line_name,
        transport_mode, direction, destination, scheduled, expected,
        delay_minutes, timestamp.
    """
    r = requests.get(
        f"{SL_BASE_URL}/sites/{site_id}/departures",
        params={"forecast": forecast},
        headers={"Ocp-Apim-Subscription-Key": api_key},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()

    rows: list[dict] = []
    now_utc = datetime.now(timezone.utc)

    # API returns a flat "departures" list; transport_mode is on line.transport_mode
    for dep in data.get("departures", []):
        sched = _parse_iso(dep.get("scheduled"))
        exp   = _parse_iso(dep.get("expected"))

        if sched is None:
            continue

        # Use expected if available, else fall back to scheduled (= 0 delay)
        actual = exp if exp is not None else sched
        delay_min = (actual - sched).total_seconds() / 60.0

        line = dep.get("line", {})
        rows.append({
            "site_id":        site_id,
            "line_id":        str(line.get("id", "")),
            "line_name":      line.get("designation", ""),
            "transport_mode": line.get("transport_mode", ""),
            "direction":      dep.get("direction", ""),
            "destination":    dep.get("destination", ""),
            "scheduled":      sched.astimezone(timezone.utc).replace(tzinfo=None),
            "expected":       actual.astimezone(timezone.utc).replace(tzinfo=None),
            "delay_minutes":  round(delay_min, 2),
            "fetched_at":     now_utc.replace(tzinfo=None),
        })

    return pd.DataFrame(rows)


def fetch_sl_departures(
    site_ids: Sequence[int] | None = None,
    api_key: str | None = None,
    forecast: int = 60,
) -> pd.DataFrame:
    """
    Snapshot current real-time delays for a list of SL stop sites.

    Polls each site independently; failures per site are logged and skipped
    so a single bad stop does not abort the whole fetch.

    Parameters
    ----------
    site_ids : sequence of int | None
        SL site IDs to query. Defaults to DEFAULT_SITE_IDS.
    api_key : str | None
        Trafiklab subscription key. Reads TRAFIKLAB_API_KEY env var if None.
    forecast : int
        Look-ahead window in minutes passed to the SL API.

    Returns
    -------
    pd.DataFrame
        Columns: site_id, line_id, line_name, transport_mode, direction,
                 destination, scheduled, expected, delay_minutes, fetched_at.
        Index: RangeIndex.  Empty DataFrame on total failure.
    """
    key = api_key or os.getenv("TRAFIKLAB_API_KEY", "")
    if not key:
        log.error("TRAFIKLAB_API_KEY not set — skipping SL fetch")
        return pd.DataFrame()

    ids = list(site_ids) if site_ids is not None else DEFAULT_SITE_IDS
    chunks: list[pd.DataFrame] = []

    for site_id in ids:
        try:
            chunk = _fetch_site_departures(site_id, key, forecast)
            if not chunk.empty:
                chunks.append(chunk)
        except requests.HTTPError as exc:
            log.warning("SL site %s HTTP %s — skipped", site_id, exc.response.status_code)
        except Exception as exc:
            log.warning("SL site %s failed: %s — skipped", site_id, exc)

    if not chunks:
        return _fail("SL departures", RuntimeError("all sites failed or returned empty"))

    df = pd.concat(chunks, ignore_index=True)
    return _ok("SL departures", df)


# ── 2. Open-Meteo archive (historical weather actuals) ─────────────────────────

def fetch_weather_archive(start, end) -> pd.DataFrame:
    """
    Fetch hourly historical weather from Open-Meteo archive API.

    Parameters
    ----------
    start, end : date | str | pd.Timestamp
        Inclusive date range (archive lags ~2 days behind today).

    Returns
    -------
    pd.DataFrame
        Hourly index (UTC). Columns: temperature, wind_speed, precipitation,
        snowfall, cloud_cover.  Empty DataFrame on failure.
    """
    try:
        s = _to_date(start)
        e = min(_to_date(end), date.today() - timedelta(days=2))
        if s > e:
            log.warning("fetch_weather_archive: start %s > available end %s, skipping", s, e)
            return pd.DataFrame()

        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude":   WEATHER_LAT,
                "longitude":  WEATHER_LON,
                "start_date": s.isoformat(),
                "end_date":   e.isoformat(),
                "hourly":     "temperature_2m,wind_speed_10m,precipitation,snowfall,cloud_cover",
                "timezone":   "UTC",
            },
            timeout=120,
        )
        r.raise_for_status()
        h = r.json().get("hourly", {})

        df = pd.DataFrame({
            "timestamp":   pd.to_datetime(h["time"]).tz_localize("UTC"),
            "temperature": h.get("temperature_2m"),
            "wind_speed":  h.get("wind_speed_10m"),
            "precipitation": h.get("precipitation"),
            "snowfall":    h.get("snowfall"),
            "cloud_cover": h.get("cloud_cover"),
        })
        df = df.set_index("timestamp").sort_index()
        return _ok("Open-Meteo archive", df)

    except Exception as exc:
        return _fail("Open-Meteo archive", exc)


# ── 3. Open-Meteo forecast (next 48 h) ─────────────────────────────────────────

def fetch_weather_forecast() -> pd.DataFrame:
    """
    Fetch the next 48-hour weather forecast from Open-Meteo for Stockholm.

    Returns
    -------
    pd.DataFrame
        Hourly index (UTC). Columns: temperature, wind_speed, precipitation,
        snowfall, cloud_cover.  Empty DataFrame on failure.
    """
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":    WEATHER_LAT,
                "longitude":   WEATHER_LON,
                "hourly":      "temperature_2m,wind_speed_10m,precipitation,snowfall,cloud_cover",
                "forecast_days": 2,
                "timezone":    "UTC",
            },
            timeout=30,
        )
        r.raise_for_status()
        h = r.json().get("hourly", {})

        df = pd.DataFrame({
            "timestamp":   pd.to_datetime(h["time"]).tz_localize("UTC"),
            "temperature": h.get("temperature_2m"),
            "wind_speed":  h.get("wind_speed_10m"),
            "precipitation": h.get("precipitation"),
            "snowfall":    h.get("snowfall"),
            "cloud_cover": h.get("cloud_cover"),
        })
        df = df.set_index("timestamp").sort_index()
        return _ok("Open-Meteo forecast", df)

    except Exception as exc:
        return _fail("Open-Meteo forecast", exc)


# ── standalone smoke test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=" * 55)
    print("DATA SOURCE SMOKE TEST")
    print("=" * 55)

    df_deps = fetch_sl_departures()
    if not df_deps.empty:
        print(f"\n  Sample delays (minutes):\n{df_deps[['line_name','destination','delay_minutes']].head()}")

    today = date.today()
    df_wx = fetch_weather_archive(today - timedelta(days=7), today)
    df_fcst = fetch_weather_forecast()

    print("=" * 55)
    if df_deps.empty and df_wx.empty:
        print("FAILED: no data returned from any source")
        sys.exit(1)
    print("Smoke test PASSED")
