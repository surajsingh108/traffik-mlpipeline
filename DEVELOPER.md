# Developer Reference — Traffik ML Pipeline

Stockholm transit delay prediction. End-to-end MLOps: SL real-time API → DuckDB → LightGBM → FastAPI → Azure Container Apps → GitHub Pages dashboard.

---

## Architecture

```
Browser (GitHub Pages)
  │
  ├── GET /config          → Groq API key (browser calls Groq directly)
  ├── GET /upcoming        → Next 30 min departures + ML predictions
  ├── GET /delays          → Last 50 measured delays (from DuckDB)
  ├── GET /weather         → Last 24h weather (from DuckDB)
  ├── GET /health          → Liveness + model_ready flag
  └── GET /model/info      → Champion metrics + feature config

FastAPI (Azure Container Apps — swedencentral)
  │
  ├── DuckDB (data/traffik.duckdb — baked into container)
  │     ├── delays table   ← SL Trafiklab API (every 15 min)
  │     └── weather table  ← Open-Meteo API (every 15 min)
  │
  └── model/ directory (baked into container)
        ├── delay_model.pkl   LightGBM champion
        ├── config.json       label maps + feature list
        └── metrics.json      test MAE/RMSE/n_train/n_features
```

---

## Key URLs

| Resource | URL |
|----------|-----|
| Live dashboard | https://surajsingh108.github.io/traffik-mlpipeline/ |
| API base | https://traffik-api.ambitiousflower-45d3cfc8.swedencentral.azurecontainerapps.io |
| API docs (Swagger) | `<API base>/docs` |
| GHCR package | ghcr.io/surajsingh108/traffik-api |
| Azure resource group | rg-traffik-ml (swedencentral) |
| Container App name | traffik-api |

---

## Environment Variables

All set as Container App env vars (not secretrefs — plain env vars so they survive restarts without secret propagation delays).

| Variable | Where set | Purpose |
|----------|-----------|---------|
| `TRAFIKLAB_API_KEY` | Container App | SL departure API |
| `GROQ_API_KEY` | Container App | Served via GET /config to browser |
| `TRAFFIK_DB_PATH` | Container App | `/app/data/traffik.duckdb` |
| `MODEL_DIR` | Container App | `/app/model` |

Local dev: copy `.env.example` → `.env`, fill in keys. `.env` is gitignored.

---

## Model

**Algorithm:** LightGBM (MAE objective)  
**Target:** `delay_minutes` (continuous regression)  
**Holdout:** last 7 days of collected data  
**Promotion rule:** new model MAE < champion × 0.98 (must beat by ≥ 2%)

### 21 Features

| Group | Features |
|-------|----------|
| Calendar | hour, hour_sin, hour_cos, day_of_week, is_weekend, is_holiday, month, week_of_year, morning_peak, evening_peak |
| Route | transport_mode_enc, site_id_enc, line_id_enc (label-encoded from config.json) |
| Weather | temperature, wind_speed, precipitation, snowfall, cloud_cover |
| Lags | lag_1h_mean, lag_24h_mean, rolling_6h_mean |

**Unknown routes:** `line_id_enc = -1` when a line wasn't seen in training. Model still predicts but the API returns `known_route: false` and the dashboard shows a warning.

---

## Data Pipeline

**Poller** runs inside the container via `supervisord` (defined in `infra/poller.sh`):

```
every 15 min:
  fetch_sl_departures(DEFAULT_SITE_IDS, forecast=60)  → delays table
  fetch_weather()                                       → weather table
```

**Stations polled** (`DEFAULT_SITE_IDS` in `transit/data_sources.py`):

| Station | site_id |
|---------|---------|
| T-Centralen | 9001 |
| Slussen | 9180 |
| Fridhemsplan | 9117 |
| Gullmarsplan | 9192 |
| Odenplan | 9261 |
| Liljeholmen | 9530 |
| Solna centrum | 9306 |
| Sundbyberg | 9325 |

Solna centrum and Sundbyberg were added 2026-07-03. Their model encodings will be `-1` (unknown) until a retrain includes their data.

---

## Docker Build & Deploy

The model and data files are **not in git** — they're baked into the image during a local build.

```bash
# 1. Build locally (includes model/ and data/)
docker build --platform linux/amd64 -t ghcr.io/surajsingh108/traffik-api:YYYYMMDD .

# 2. Push to GHCR (PAT needs write:packages scope)
echo $GITHUB_TOKEN | docker login ghcr.io -u surajsingh108 --password-stdin
docker push ghcr.io/surajsingh108/traffik-api:YYYYMMDD

# 3. Deploy to Container Apps
az containerapp update \
  --name traffik-api \
  --resource-group rg-traffik-ml \
  --image ghcr.io/surajsingh108/traffik-api:YYYYMMDD
```

**Critical:** always use a new tag (dated). Using `:latest` causes Container Apps to serve the cached old image even after a push.

---

## CI/CD (GitHub Actions)

`.github/workflows/deploy.yml` builds and pushes to GHCR on every push to `master`. The Azure deploy step requires `AZURE_CREDENTIALS` secret (service principal JSON) — not yet configured. Without it, the workflow builds+pushes but skips deploy. Manual `az containerapp update` is the current deploy method.

To fix CI/CD deploy permanently:
```bash
az ad sp create-for-rbac \
  --name "traffik-github-actions" \
  --role contributor \
  --scopes "/subscriptions/<SUB_ID>/resourceGroups/rg-traffik-ml" \
  --sdk-auth
# paste JSON output as AZURE_CREDENTIALS GitHub secret
```

---

## Groq NL Parsing — Architecture Decision

**Problem:** Azure Container Apps use datacenter IPs blocked by Cloudflare fronting `api.groq.com` (error 1010).

**Solution:** `GET /config` returns the Groq API key. The browser (residential IP) calls Groq directly. Key is never committed to git — only lives in the Container App env var.

**Rate limit:** 5 calls per 30 minutes enforced client-side in `groqCallLog` array.

**Key rotation:** If GitHub secret scanning detects the key in a commit, Groq auto-revokes it. Generate a new key at console.groq.com and update the Container App env var:
```bash
az containerapp update \
  --name traffik-api \
  --resource-group rg-traffik-ml \
  --set-env-vars "GROQ_API_KEY=<new-key>"
```

---

## Adding a New Station

1. Look up the site_id: `curl "https://transport.integration.sl.se/v1/sites?q=<name>"`
2. Add to `DEFAULT_SITE_IDS` in `transit/data_sources.py`
3. Add `<option>` to the station dropdown in `docs/index.html`
4. Add to `STATIONS` dict in the dashboard JS
5. Add to the Groq system prompt station list in the dashboard JS
6. Commit and push — Pages deploys automatically
7. Rebuild Docker image and deploy (model will show `known_route: false` for the new station until retrain)

---

## Retrain

```bash
# Via API (triggers transit.retrain subprocess inside container)
curl -X POST <API base>/retrain

# Locally
python -m transit.retrain
```

The retrain module runs champion/challenger: trains a new model, compares MAE against the current `model/delay_model.pkl`. If new MAE < champion × 0.98, promotes new model (overwrites pkl + metrics + config). After retrain via API, `/predict` and `/model/info` use the new model immediately (in-memory reload).

---

## Known Issues & Solutions

| Issue | Cause | Fix |
|-------|-------|-----|
| Groq 403 from container | Cloudflare blocks Azure datacenter IPs | Browser calls Groq directly via /config key |
| model_ready: false | model/ not in git, GitHub Actions image lacks it | Local Docker build (includes model/) |
| Same tag not re-pulled | Container Apps caches image by digest | Use dated tag on every deploy |
| GitHub Pages "deployment failed" | Transient GitHub infrastructure | `gh run rerun <run-id>` |
| `urllib.error` AttributeError | urllib.error is a separate submodule | Import explicitly: `import urllib.error` |
| Groq key blocked by GitHub | Key committed to HTML | Key moved to Container App env var, served via /config |
| `known_route: false` on new station/line | Line not in training label map | Expected — collect data, then retrain |
