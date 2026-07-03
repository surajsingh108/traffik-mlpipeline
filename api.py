"""
api.py — traffik-mlpipeline REST API

Serves delay predictions from the champion LightGBM model and exposes
recent delay + weather data from DuckDB.  Mirrors SE3 api.py style.

Endpoints
---------
GET  /health        Liveness check + model readiness flag
GET  /model/info    Champion metrics and feature config
GET  /delays        Recent delays (last 50 rows)
GET  /weather       Recent weather (last 24 hours)
POST /predict       Single-stop delay prediction
POST /retrain       Trigger champion/challenger retrain subprocess
GET  /config        Public runtime config (Groq key for client-side NL parsing)
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

log = logging.getLogger(__name__)

DB_PATH    = os.environ.get("TRAFFIK_DB_PATH", "data/traffik.duckdb")
MODEL_DIR  = Path(os.environ.get("MODEL_DIR", "model"))
MODEL_FILE = MODEL_DIR / "delay_model.pkl"
CONFIG_FILE = MODEL_DIR / "config.json"
METRICS_FILE = MODEL_DIR / "metrics.json"

_SE_HOLIDAYS = holidays.Sweden()

app = FastAPI(title="Traffik ML API", description="Stockholm transit delay prediction")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── model + config loaded once at startup ──────────────────────────────────────

_model: object | None = None
_config: dict | None = None
_metrics: dict | None = None


def _load_artifacts() -> None:
    global _model, _config, _metrics
    if MODEL_FILE.exists():
        try:
            with open(MODEL_FILE, "rb") as f:
                _model = pickle.load(f)
        except Exception as exc:
            log.warning("Could not load model: %s", exc)
    if CONFIG_FILE.exists():
        try:
            _config = json.loads(CONFIG_FILE.read_text())
        except Exception as exc:
            log.warning("Could not load config: %s", exc)
    if METRICS_FILE.exists():
        try:
            _metrics = json.loads(METRICS_FILE.read_text())
        except Exception as exc:
            log.warning("Could not load metrics: %s", exc)


_load_artifacts()


# ── DB helper ──────────────────────────────────────────────────────────────────

def _get_db(read_only: bool = True):
    import duckdb
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    return duckdb.connect(DB_PATH, read_only=read_only)


# ── inference helper ───────────────────────────────────────────────────────────

def _build_row(
    site_id: int,
    line_id: str,
    transport_mode: str,
    scheduled: str,
    temperature: float | None,
    wind_speed: float | None,
    precipitation: float | None,
    snowfall: float | None,
    cloud_cover: float | None,
    config: dict,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for inference.

    Applies calendar transforms and encoding maps from config.
    Weather and lag features fall back to NaN when not supplied
    (LightGBM handles NaN natively).

    Parameters
    ----------
    site_id, line_id, transport_mode : route identifiers
    scheduled : ISO 8601 datetime string for the departure
    temperature … cloud_cover : optional weather values (°C, km/h, mm, mm, %)
    config : dict loaded from model/config.json

    Returns
    -------
    pd.DataFrame
        One row with all 21 feature columns.
    """
    ts = pd.Timestamp(scheduled, tz="UTC").tz_convert("Europe/Stockholm")

    hour        = ts.hour
    day_of_week = ts.dayofweek
    month       = ts.month
    week_of_year = ts.isocalendar()[1]

    transport_mode_map: dict[str, int] = config.get("transport_mode_map", {})
    line_id_map: dict[str, int]        = config.get("line_id_map", {})
    site_id_map: dict[str, int]        = {str(k): v for k, v in config.get("site_id_map", {}).items()}

    row = {
        # calendar
        "hour":          hour,
        "hour_sin":      float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos":      float(np.cos(2 * np.pi * hour / 24)),
        "day_of_week":   day_of_week,
        "is_weekend":    int(day_of_week >= 5),
        "is_holiday":    int(ts.date() in _SE_HOLIDAYS),
        "month":         month,
        "week_of_year":  week_of_year,
        "morning_peak":  int(7 <= hour <= 9),
        "evening_peak":  int(16 <= hour <= 18),
        # route (unknown → -1)
        "transport_mode_enc": transport_mode_map.get(transport_mode, -1),
        "site_id_enc":        site_id_map.get(str(site_id), -1),
        "line_id_enc":        line_id_map.get(str(line_id), -1),
        # weather
        "temperature":   temperature,
        "wind_speed":    wind_speed,
        "precipitation": precipitation,
        "snowfall":      snowfall,
        "cloud_cover":   cloud_cover,
        # lag features — NaN at inference (LightGBM handles)
        "lag_1h_mean":    np.nan,
        "lag_24h_mean":   np.nan,
        "rolling_6h_mean": np.nan,
    }

    feat_cols: list[str] = config.get("feature_cols", list(row.keys()))
    df = pd.DataFrame([row])
    # Return only columns the model was trained on, in training order
    available = [c for c in feat_cols if c in df.columns]
    return df[available]


# ── request / response models ──────────────────────────────────────────────────

class PredictRequest(BaseModel):
    site_id: int
    line_id: str
    transport_mode: str
    scheduled: str                  # ISO 8601 UTC datetime, e.g. "2025-07-02T08:15:00Z"
    temperature: float | None = None
    wind_speed: float | None = None
    precipitation: float | None = None
    snowfall: float | None = None
    cloud_cover: float | None = None


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Liveness check. Returns model readiness and artifact timestamps."""
    model_ok  = _model is not None
    config_ok = _config is not None
    return {
        "status":     "ok" if model_ok else "degraded",
        "model_ready": model_ok,
        "config_ready": config_ok,
        "model_file":  str(MODEL_FILE),
        "model_mtime": (
            datetime.fromtimestamp(MODEL_FILE.stat().st_mtime, tz=timezone.utc).isoformat()
            if MODEL_FILE.exists() else None
        ),
    }


@app.get("/model/info")
async def model_info():
    """Return champion model metrics and feature configuration."""
    if _metrics is None and _config is None:
        return {"error": "no model artifacts found", "model_dir": str(MODEL_DIR)}
    return {
        "metrics": _metrics,
        "feature_cols": _config.get("feature_cols") if _config else None,
        "lgbm_params":  _config.get("lgbm_params")  if _config else None,
        "holdout_days": _config.get("holdout_days")  if _config else None,
    }


@app.get("/delays")
async def get_delays():
    """Return the 50 most recent delay rows from DuckDB."""
    try:
        conn = _get_db()
        df = conn.execute("""
            SELECT fetched_at, site_id, line_id, line_name,
                   transport_mode, scheduled, expected, delay_minutes
            FROM delays
            ORDER BY scheduled DESC
            LIMIT 50
        """).df()
        conn.close()

        if df.empty:
            return {"delays": [], "error": "no data"}

        delays = [
            {
                "fetched_at":     str(row["fetched_at"]),
                "site_id":        int(row["site_id"]),
                "line_id":        str(row["line_id"]),
                "line_name":      str(row["line_name"]),
                "transport_mode": str(row["transport_mode"]),
                "scheduled":      str(row["scheduled"]),
                "expected":       str(row["expected"]),
                "delay_minutes":  float(row["delay_minutes"]) if pd.notna(row["delay_minutes"]) else None,
            }
            for _, row in df.iterrows()
        ]
        delays.reverse()
        return {"delays": delays, "count": len(delays)}
    except Exception as exc:
        return {"delays": [], "error": str(exc)}


@app.get("/weather")
async def get_weather():
    """Return the 24 most recent weather rows from DuckDB."""
    try:
        conn = _get_db()
        df = conn.execute("""
            SELECT timestamp, temperature, wind_speed, precipitation,
                   snowfall, cloud_cover
            FROM weather
            ORDER BY timestamp DESC
            LIMIT 24
        """).df()
        conn.close()

        if df.empty:
            return {"weather": [], "error": "no data"}

        weather = [
            {
                "timestamp":    str(row["timestamp"]),
                "temperature":  float(row["temperature"])  if pd.notna(row["temperature"])  else None,
                "wind_speed":   float(row["wind_speed"])   if pd.notna(row["wind_speed"])   else None,
                "precipitation": float(row["precipitation"]) if pd.notna(row["precipitation"]) else None,
                "snowfall":     float(row["snowfall"])     if pd.notna(row["snowfall"])     else None,
                "cloud_cover":  float(row["cloud_cover"])  if pd.notna(row["cloud_cover"])  else None,
            }
            for _, row in df.iterrows()
        ]
        weather.reverse()
        return {"weather": weather}
    except Exception as exc:
        return {"weather": [], "error": str(exc)}


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    Predict delay_minutes for a single departure.

    If weather fields are omitted, the latest matching weather row is fetched
    from DuckDB. Lag features default to NaN (LightGBM handles gracefully).
    """
    if _model is None or _config is None:
        return {"error": "model not loaded — check /health"}

    # Fill missing weather from DB if possible
    temperature  = req.temperature
    wind_speed   = req.wind_speed
    precipitation = req.precipitation
    snowfall     = req.snowfall
    cloud_cover  = req.cloud_cover

    if any(v is None for v in [temperature, wind_speed, precipitation, snowfall, cloud_cover]):
        try:
            conn = _get_db()
            wx = conn.execute("""
                SELECT temperature, wind_speed, precipitation, snowfall, cloud_cover
                FROM weather
                ORDER BY timestamp DESC
                LIMIT 1
            """).df()
            conn.close()
            if not wx.empty:
                row = wx.iloc[0]
                if temperature  is None: temperature  = float(row["temperature"])  if pd.notna(row["temperature"])  else None
                if wind_speed   is None: wind_speed   = float(row["wind_speed"])   if pd.notna(row["wind_speed"])   else None
                if precipitation is None: precipitation = float(row["precipitation"]) if pd.notna(row["precipitation"]) else None
                if snowfall     is None: snowfall     = float(row["snowfall"])     if pd.notna(row["snowfall"])     else None
                if cloud_cover  is None: cloud_cover  = float(row["cloud_cover"])  if pd.notna(row["cloud_cover"])  else None
        except Exception:
            pass  # proceed with NaN weather — model handles it

    try:
        X = _build_row(
            site_id=req.site_id,
            line_id=req.line_id,
            transport_mode=req.transport_mode,
            scheduled=req.scheduled,
            temperature=temperature,
            wind_speed=wind_speed,
            precipitation=precipitation,
            snowfall=snowfall,
            cloud_cover=cloud_cover,
            config=_config,
        )
        pred = float(_model.predict(X)[0])
    except Exception as exc:
        return {"error": f"prediction failed: {exc}"}

    return {
        "delay_minutes": round(pred, 2),
        "site_id":        req.site_id,
        "line_id":        req.line_id,
        "transport_mode": req.transport_mode,
        "scheduled":      req.scheduled,
        "weather_used": {
            "temperature":   temperature,
            "wind_speed":    wind_speed,
            "precipitation": precipitation,
            "snowfall":      snowfall,
            "cloud_cover":   cloud_cover,
        },
    }


@app.post("/retrain")
async def retrain():
    """
    Trigger a champion/challenger retrain via transit.retrain.

    Runs as a blocking subprocess (timeout 10 min).
    Returns the promotion decision and new metrics on success.
    """
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "transit.retrain"],
            capture_output=True, text=True, timeout=600,
        )
        duration = round(time.time() - t0, 1)
        success  = result.returncode == 0

        # Reload artifacts so /predict and /model/info reflect the new champion
        if success:
            _load_artifacts()

        return {
            "status":           "ok" if success else "error",
            "returncode":       result.returncode,
            "duration_seconds": duration,
            "stdout":           result.stdout[-500:].strip() if result.stdout else None,
            "stderr":           result.stderr[-300:].strip() if not success and result.stderr else None,
            "metrics":          _metrics,
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "duration_seconds": 600}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@app.get("/config")
async def get_config():
    """
    Return public runtime config for the dashboard.

    The Groq key is served here so the browser can call Groq directly,
    avoiding Cloudflare datacenter blocks that affect server-side calls.
    The key is stored as a Container App env var — never committed to git.
    """
    return {"groq_key": os.environ.get("GROQ_API_KEY", "")}


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=8000)
