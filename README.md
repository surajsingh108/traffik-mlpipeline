# Traffik ML — Stockholm Transit Delay Prediction

An end-to-end MLOps pipeline that predicts SL transit delays in Stockholm using real-time departure data, weather, and a LightGBM champion/challenger model — deployed to Azure Container Apps.

The dashboard lets anyone check if their bus, metro or train will be on time. Type a journey in plain English ("Bus 65 from Odenplan tomorrow at 8am") and Groq parses it into a prediction request in real time.

**[Live Dashboard](https://surajsingh108.github.io/traffik-mlpipeline/) · [API Docs](https://traffik-api.ambitiousflower-45d3cfc8.swedencentral.azurecontainerapps.io/docs)**

---

## Architecture

```
  GitHub Pages dashboard
        │                               ┌─────────────────┐
        │  /predict  /delays  /weather  │   Browser calls  │
        │  /config (Groq key)           │   Groq directly  │
        ▼                               │   (bypasses CDN  │
┌─────────────────────────────────────┐ │    datacenter    │
│  Azure Container Apps               │ │    block)        │
│                                     │ └────────┬────────┘
│  ┌──────────────────┐    ┌────────────────────┐│           │
│  │  FastAPI (8000)  │    │  Data Poller        ││           │
│  │  /predict        │    │  (every 15 min)     ││           │
│  │  /delays         │    │  SL Departures →    ││           │
│  │  /weather        │    │  DuckDB             ││           │
│  │  /retrain        │    │  Open-Meteo →       ││           │
│  │  /config         │    │  DuckDB             │           │
│  └──────────────────┘    └─────────────────────┘           │
│           │                                                 │
│    DuckDB (data/traffik.duckdb)                             │
└─────────────────────────────────────────────────────────────┘
         │                        │              │
   Trafiklab SL API          Open-Meteo API   Groq API
   (departures)              (weather)        (NL parsing,
                                               called from
                                               browser)

Registry: GitHub Container Registry (free)
CI/CD: GitHub Actions → GHCR → Azure Container Apps
```

## Features

**21 model features across 4 groups:**

| Group | Features |
|-------|----------|
| Calendar | hour, hour_sin/cos, day_of_week, is_weekend, is_holiday, month, morning_peak, evening_peak |
| Route | transport_mode_enc, site_id_enc, line_id_enc (label-encoded) |
| Weather | temperature, wind_speed, precipitation, snowfall, cloud_cover |
| Lags | lag_1h_mean, lag_24h_mean, rolling_6h_mean (per-site rolling delay stats) |

## Model Performance

| Metric | Value |
|--------|-------|
| Test MAE | 0.035 min |
| Test RMSE | 0.203 min |
| Algorithm | LightGBM (MAE objective) |
| Holdout | Last 7 days |
| Champion rule | New MAE < Champion × 0.98 |

## API

Base URL: `https://traffik-api.ambitiousflower-45d3cfc8.swedencentral.azurecontainerapps.io`

```bash
# Health check
curl /health

# Predict delay for a departure
curl -X POST /predict \
  -H "Content-Type: application/json" \
  -d '{
    "site_id": 9001,
    "line_id": "65",
    "transport_mode": "BUS",
    "scheduled": "2025-07-03T08:15:00Z"
  }'

# Get public config (Groq key for client-side NL parsing)
curl /config

# Recent delays
curl /delays

# Current weather
curl /weather

# Trigger retrain
curl -X POST /retrain
```

## Stack

| Layer | Technology |
|-------|------------|
| Data ingestion | Python, Trafiklab SL API, Open-Meteo |
| Storage | DuckDB |
| ML | LightGBM, scikit-learn, pandas |
| Experiment tracking | MLflow (SQLite) |
| Serving | FastAPI, uvicorn |
| Container | Docker, GitHub Container Registry |
| Hosting | Azure Container Apps (scale-to-zero, free tier) |
| Dashboard | GitHub Pages, Tailwind CSS, Chart.js |
| NL parsing | Groq (llama-3.1-8b-instant), called directly from browser |
| CI/CD | GitHub Actions |

## Project Structure

```
transit/
├── data_sources.py   SL + Open-Meteo API clients
├── pipeline.py       Incremental DuckDB sync
├── features.py       Feature engineering (21 features)
├── ml.py             LightGBM training + champion/challenger
└── retrain.py        Orchestrator (8 steps)

infra/
├── poller.sh         Continuous data collection loop
└── azure-function/
    └── function_app.py  Timer triggers (retrain every 3h)

api.py                FastAPI serving layer
Dockerfile            Container definition
deploy.sh             Manual deploy script
docs/index.html       GitHub Pages dashboard
.github/workflows/
└── deploy.yml        CI/CD: push → build → deploy
```

## Local Setup

```bash
git clone https://github.com/surajsingh108/traffik-mlpipeline
cd traffik-mlpipeline
pip install -e .
cp .env.example .env   # add SL_API_KEY

# Collect data
python -m transit.pipeline

# Train model
python -m transit.ml --force

# Run API
uvicorn api:app --reload
```

## Deployment

```bash
export GITHUB_TOKEN="ghp_..."   # write:packages + read:packages + repo
export SL_API_KEY="..."
./deploy.sh
```

Or push to `master` — GitHub Actions builds and deploys automatically.

## Cost

**~$0/month** — Azure Container Apps scale-to-zero free tier, GitHub Container Registry free tier, GitHub Pages free.
