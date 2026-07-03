"""
transit/features.py — unified feature builder for delay prediction

Reads raw delays + weather from DuckDB and produces a model-ready DataFrame.
Supports two modes:
  "train"    — full historical join, used in ml.py
  "predict"  — single-row inference, used in api.py

Feature groups
--------------
Calendar    : hour, day_of_week, is_weekend, is_holiday, month, sin/cos hour
Route       : transport_mode, site_id, line_id (label-encoded)
Weather     : temperature, wind_speed, precipitation, snowfall, cloud_cover
Peaks       : morning_peak (7-9h), evening_peak (16-18h)
Lags        : per-site rolling delay stats (requires historical data in DuckDB)

Usage
-----
  from transit.features import build_features, make_feature_cols
  df = build_features(con, mode="train")
  X  = df[make_feature_cols(df)]
  y  = df["delay_minutes"]
"""
from __future__ import annotations

import logging
from typing import Literal

import holidays
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_SE_HOLIDAYS = holidays.Sweden()

# ── label-encoding maps (built from training data, reused at inference) ────────
# Populated by build_features(); saved in model/config.json by ml.py

TRANSPORT_MODE_MAP: dict[str, int] = {}
LINE_ID_MAP:        dict[str, int] = {}
SITE_ID_MAP:        dict[int, int] = {}


# ── feature column lists ────────────────────────────────────────────────────────

_CALENDAR_COLS = [
    "hour", "hour_sin", "hour_cos",
    "day_of_week", "is_weekend", "is_holiday",
    "month", "week_of_year",
    "morning_peak", "evening_peak",
]

_ROUTE_COLS = [
    "transport_mode_enc", "site_id_enc", "line_id_enc",
]

_WEATHER_COLS = [
    "temperature", "wind_speed", "precipitation", "snowfall", "cloud_cover",
]

_LAG_COLS = [
    "lag_1h_mean", "lag_24h_mean", "rolling_6h_mean",
]


def make_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns present in df (graceful if some are missing).

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_features().

    Returns
    -------
    list[str]
        Ordered feature column names ready for model input.
    """
    candidates = _CALENDAR_COLS + _ROUTE_COLS + _WEATHER_COLS + _LAG_COLS
    return [c for c in candidates if c in df.columns]


# ── calendar features ──────────────────────────────────────────────────────────

def _add_calendar(df: pd.DataFrame, ts_col: str = "scheduled") -> pd.DataFrame:
    """Add calendar features derived from the scheduled departure timestamp."""
    ts = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert("Europe/Stockholm")

    df = df.copy()
    df["hour"]         = ts.dt.hour
    df["hour_sin"]     = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]     = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week"]  = ts.dt.dayofweek          # 0=Mon, 6=Sun
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"]   = ts.dt.date.map(lambda d: int(d in _SE_HOLIDAYS))
    df["month"]        = ts.dt.month
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["morning_peak"] = df["hour"].between(7, 9).astype(int)
    df["evening_peak"] = df["hour"].between(16, 18).astype(int)
    return df


# ── route encoding ─────────────────────────────────────────────────────────────

def _encode_routes(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """Label-encode transport_mode, site_id, line_id.

    Parameters
    ----------
    fit : bool
        If True, rebuild the global encoding maps from df (training mode).
        If False, apply existing maps and encode unknowns as -1 (inference mode).
    """
    global TRANSPORT_MODE_MAP, LINE_ID_MAP, SITE_ID_MAP

    df = df.copy()

    if fit:
        modes = sorted(df["transport_mode"].dropna().unique())
        TRANSPORT_MODE_MAP = {m: i for i, m in enumerate(modes)}

        lines = sorted(df["line_id"].astype(str).dropna().unique())
        LINE_ID_MAP = {l: i for i, l in enumerate(lines)}

        sites = sorted(df["site_id"].dropna().unique())
        SITE_ID_MAP = {int(s): i for i, s in enumerate(sites)}

    df["transport_mode_enc"] = df["transport_mode"].map(TRANSPORT_MODE_MAP).fillna(-1).astype(int)
    df["line_id_enc"]        = df["line_id"].astype(str).map(LINE_ID_MAP).fillna(-1).astype(int)
    df["site_id_enc"]        = df["site_id"].map(SITE_ID_MAP).fillna(-1).astype(int)
    return df


# ── weather join ───────────────────────────────────────────────────────────────

def _join_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Left-join weather onto delays by rounding scheduled time to the nearest hour."""
    if weather.empty:
        log.warning("_join_weather: weather DataFrame is empty — weather features will be NaN")
        for col in _WEATHER_COLS:
            df[col] = np.nan
        return df

    df = df.copy()
    # Round scheduled to hour and convert to UTC for join key
    ts = pd.to_datetime(df["scheduled"], utc=True)
    df["_weather_key"] = ts.dt.floor("h")

    wx = weather.copy()
    wx.index = pd.to_datetime(wx.index, utc=True)
    wx.index.name = "_weather_key"
    wx = wx.reset_index()

    merged = df.merge(wx, on="_weather_key", how="left")
    merged = merged.drop(columns=["_weather_key"])
    return merged


# ── lag features ───────────────────────────────────────────────────────────────

def _add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-site rolling delay statistics as lag features.

    Computes over the historical delays already in df (sorted by scheduled).
    With few rows these will be NaN; LightGBM handles NaN natively.
    """
    df = df.copy().sort_values("scheduled")

    df["lag_1h_mean"]    = np.nan
    df["lag_24h_mean"]   = np.nan
    df["rolling_6h_mean"] = np.nan

    for site_id, grp in df.groupby("site_id"):
        idx = grp.index
        ts  = pd.to_datetime(grp["scheduled"], utc=True)

        lag_1h   = []
        lag_24h  = []
        roll_6h  = []

        for i, (row_idx, row_ts) in enumerate(zip(idx, ts)):
            hist = grp.loc[idx[:i]]
            hist_ts = ts.iloc[:i]

            # Mean delay in the last 1 hour at this site
            mask_1h  = hist_ts >= (row_ts - pd.Timedelta(hours=1))
            lag_1h.append(hist.loc[mask_1h.values, "delay_minutes"].mean() if mask_1h.any() else np.nan)

            # Mean delay at same hour ±1h window 24h ago
            same_hour_window = (
                (hist_ts >= row_ts - pd.Timedelta(hours=25)) &
                (hist_ts <= row_ts - pd.Timedelta(hours=23))
            )
            lag_24h.append(hist.loc[same_hour_window.values, "delay_minutes"].mean() if same_hour_window.any() else np.nan)

            # Rolling 6h mean
            mask_6h = hist_ts >= (row_ts - pd.Timedelta(hours=6))
            roll_6h.append(hist.loc[mask_6h.values, "delay_minutes"].mean() if mask_6h.any() else np.nan)

        df.loc[idx, "lag_1h_mean"]     = lag_1h
        df.loc[idx, "lag_24h_mean"]    = lag_24h
        df.loc[idx, "rolling_6h_mean"] = roll_6h

    return df


# ── main builder ───────────────────────────────────────────────────────────────

def build_features(
    con,
    weather: pd.DataFrame | None = None,
    mode: Literal["train", "predict"] = "train",
    min_rows: int = 20,
) -> pd.DataFrame:
    """
    Build the model-ready feature DataFrame from DuckDB.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection.
    weather : pd.DataFrame | None
        Pre-fetched weather DataFrame (timestamp index, UTC).
        If None, loads from the weather table in DuckDB.
    mode : {"train", "predict"}
        "train" fits encoding maps from data; "predict" applies existing maps.
    min_rows : int
        Minimum rows required; raises ValueError if fewer rows found.

    Returns
    -------
    pd.DataFrame
        One row per departure with all feature columns + delay_minutes target.
    """
    # Load delays
    df = con.execute("""
        SELECT
            fetched_at, site_id, line_id, line_name,
            transport_mode, direction, destination,
            scheduled, expected, delay_minutes
        FROM latest_delays
        WHERE delay_minutes IS NOT NULL
        ORDER BY scheduled
    """).fetchdf()

    if len(df) < min_rows:
        raise ValueError(
            f"build_features: only {len(df)} rows in delays table "
            f"(need at least {min_rows}). Run pipeline.py to collect more data."
        )

    log.info("build_features: loaded %d delay rows", len(df))

    # Load weather if not provided
    if weather is None:
        wx_df = con.execute(
            "SELECT * FROM weather ORDER BY timestamp"
        ).fetchdf()
        wx_df["timestamp"] = pd.to_datetime(wx_df["timestamp"], utc=True)
        wx_df = wx_df.set_index("timestamp")
    else:
        wx_df = weather

    # Build features
    df = _add_calendar(df, ts_col="scheduled")
    df = _encode_routes(df, fit=(mode == "train"))
    df = _join_weather(df, wx_df)
    df = _add_lags(df)

    # Drop rows where target is missing or extreme (>120 min is data error)
    df = df[df["delay_minutes"].between(-5, 120)].copy()

    log.info(
        "build_features: %d rows, %d feature cols ready",
        len(df), len(make_feature_cols(df)),
    )
    return df


# ── standalone smoke test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    try:
        from dotenv import load_dotenv; load_dotenv()
    except ImportError:
        pass
    import duckdb
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    db_path = os.getenv("TRAFFIK_DB_PATH", "data/traffik.duckdb")
    con = duckdb.connect(db_path, read_only=True)

    df = build_features(con, mode="train", min_rows=5)
    feat_cols = make_feature_cols(df)

    print(f"\nFeature matrix: {df.shape}")
    print(f"Feature cols ({len(feat_cols)}): {feat_cols}")
    print(f"\nTarget stats:\n{df['delay_minutes'].describe()}")
    print(f"\nNull counts:\n{df[feat_cols].isnull().sum()[df[feat_cols].isnull().sum() > 0]}")
    con.close()
    print("\nfeatures.py smoke test PASSED")
