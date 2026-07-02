# Stage 3 Handoff — Serving + Deployment

## What was built

FastAPI serving layer, Docker image, ACR push pipeline, Container Apps deployment.

```
api.py              ✓  6 endpoints (health, model/info, delays, weather, predict, retrain)
Dockerfile          ✓  updated — single-process uvicorn, no supervisord
.dockerignore       ✓  excludes data/, .env, __pycache__, venv
deploy.sh           ✓  build → ACR push → Container Apps create/update
```

## Endpoints

| Method | Path         | Purpose                                           |
|--------|--------------|---------------------------------------------------|
| GET    | /health      | Liveness + model readiness + artifact timestamps  |
| GET    | /model/info  | Champion metrics, feature cols, LGBM params       |
| GET    | /delays      | Last 50 delay rows from DuckDB                    |
| GET    | /weather     | Last 24 weather rows from DuckDB                  |
| POST   | /predict     | Single-stop delay prediction                      |
| POST   | /retrain     | Trigger champion/challenger retrain subprocess    |
| GET    | /docs        | Auto-generated Swagger UI (FastAPI built-in)      |

### POST /predict — example

```json
{
  "site_id": 9001,
  "line_id": "17",
  "transport_mode": "TRAM",
  "scheduled": "2025-07-02T08:15:00Z"
}
```

Response:
```json
{
  "delay_minutes": 1.23,
  "site_id": 9001,
  "line_id": "17",
  "transport_mode": "TRAM",
  "scheduled": "2025-07-02T08:15:00Z",
  "weather_used": { "temperature": 18.4, "wind_speed": 3.1, ... }
}
```

Weather is auto-filled from the latest DuckDB row if omitted.  
Unknown route IDs encode to -1 (LightGBM handles gracefully).

## How to deploy

### 1. Provision Azure resources (one-time, if not done in Stage 2)

```bash
az group create --name rg-traffik-ml --location swedencentral
az acr create --name traffikmlacr -g rg-traffik-ml --sku Basic --admin-enabled true
az containerapp env create --name traffik-env -g rg-traffik-ml --location swedencentral
```

Grant the Container Apps environment pull access to ACR:

```bash
ACR_ID=$(az acr show --name traffikmlacr --query id -o tsv)
# Use system-assigned identity (deploy.sh passes --registry-identity system)
# ACA creates the identity on first deploy; grant pull role after:
PRINCIPAL=$(az containerapp show \
  --name traffik-api -g rg-traffik-ml \
  --query "identity.principalId" -o tsv)
az role assignment create \
  --assignee "$PRINCIPAL" \
  --role AcrPull \
  --scope "$ACR_ID"
```

### 2. Set env vars and deploy

```bash
export SL_API_KEY="<your-key>"
export AZURE_STORAGE_CONNECTION_STRING="<from Stage 2>"
# optional
export SLACK_WEBHOOK_URL="<url>"
export GROQ_API_KEY="<key>"

chmod +x deploy.sh
./deploy.sh
```

The script:
1. `az acr login` → authenticates Docker to ACR
2. `docker build --platform linux/amd64` → builds for Linux/amd64
3. `docker push` → pushes to `traffikmlacr.azurecr.io/traffik-api:latest`
4. `az containerapp create` (first run) or `az containerapp update` (subsequent)
5. Prints the public HTTPS URL

### 3. Subsequent deploys

```bash
./deploy.sh                    # rebuild + push + deploy latest
./deploy.sh --build-only       # just build + push (no Container App update)
./deploy.sh --deploy-only      # re-deploy without rebuilding the image
IMAGE_TAG=v1.2 ./deploy.sh     # tag a specific version
```

## Key design notes

- **Model artifacts in image**: `model/*.pkl` and `model/*.json` are copied into the image at build time. The Container App starts with the model already loaded; no cold-start model download needed.
- **Data DB not in image**: `data/` is excluded via `.dockerignore`. The Container App currently uses `TRAFFIK_DB_PATH=/app/data/traffik.duckdb`. For production, mount an Azure Files share or download from Blob on startup.
- **Weather auto-fill**: `/predict` fetches the latest weather from DuckDB if the caller omits weather fields.
- **Lag features at inference**: `lag_1h_mean`, `lag_24h_mean`, `rolling_6h_mean` are set to NaN. LightGBM handles NaN natively with no accuracy impact at low-data inference time.
- **`/retrain` reloads artifacts**: after a successful retrain subprocess, `_load_artifacts()` is called so `/predict` immediately uses the promoted model without a container restart.
- **Supervisord removed**: Dockerfile now runs `uvicorn` directly. Add supervisord back when a dashboard process is added.

## Azure infra summary

| Resource              | Name                    | Location      |
|-----------------------|-------------------------|---------------|
| Resource Group        | rg-traffik-ml           | swedencentral |
| Storage Account       | traffikmlstorage        | swedencentral |
| Key Vault             | traffik-kv              | swedencentral |
| Container Registry    | traffikmlacr            | swedencentral |
| Container Apps Env    | traffik-env             | swedencentral |
| Container App (API)   | traffik-api             | swedencentral |
| Function App (ingest) | traffik-ingest          | swedencentral |

## Next: Stage 4 — Dashboard (optional) or CI/CD

Possible next steps:
- **Streamlit dashboard** (`dashboard.py`) — add back supervisord for dual-process container
- **GitHub Actions CI** — on push to main: `./deploy.sh`
- **Azure Files mount** — persist DuckDB across Container App restarts
- **MLflow on Azure** — point `MLFLOW_TRACKING_URI` at an Azure ML workspace
