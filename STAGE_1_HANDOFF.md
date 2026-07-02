# Stage 1 Handoff — Foundation

## What was built

Repo scaffold + data ingestion layer. All files in this list are complete:

```
traffik-mlpipeline/
├── transit/
│   ├── __init__.py
│   ├── data_sources.py    ✓  Trafiklab SL departures + Open-Meteo archive/forecast
│   └── pipeline.py        ✓  DuckDB incremental sync (delays + weather tables)
├── metrics_logger.py      ✓  Ported from SE3 — unchanged
├── slack_alert.py         ✓  Ported from SE3 — unchanged
├── freshness_check.py     ✓  Ported from SE3 — thresholds: delays 2h, weather 6h
├── mlflow_utils.py        ✓  Ported from SE3 — prod URI = Azure Blob wasbs://
├── model_backup.py        ✓  Ported from SE3 — GCS → azure-storage-blob
├── pyproject.toml         ✓
├── Dockerfile             ✓  Python 3.11-slim + supervisord
├── supervisord.conf       ✓  api:8000 + dashboard:8080
├── .env.example           ✓
└── .gitignore             ✓
```

## Env vars needed (copy .env.example → .env and fill in)

| Var | Purpose | Status |
|-----|---------|--------|
| `TRAFIKLAB_API_KEY` | SL Transport API | User has key |
| `TRAFFIK_DB_PATH` | DuckDB file path | Default: `data/traffik.duckdb` |
| `GROQ_API_KEY` | Groq LLM digest | Needed in Stage 4 |
| `AZURE_STORAGE_CONNECTION_STRING` | model_backup.py | Needed for prod |
| `MLFLOW_TRACKING_URI` | Azure Blob prod | Default: sqlite local |
| `SLACK_WEBHOOK` | Optional alerts | Optional |

## Smoke test (run these before starting Stage 2)

```bash
# Install deps
pip install -e ".[dev]"

# Verify data flows into DuckDB
python -m transit.pipeline --init   # back-fills 90 days of weather, grabs current delays

# Check freshness
python freshness_check.py           # should print "All data sources are fresh" if --init ran

# Verify data in DuckDB
python -c "
import duckdb
con = duckdb.connect('data/traffik.duckdb', read_only=True)
print(con.execute('SELECT COUNT(*) FROM delays').fetchone())
print(con.execute('SELECT COUNT(*) FROM weather').fetchone())
print(con.execute('SELECT MIN(timestamp), MAX(timestamp) FROM weather').fetchone())
"
```

## DuckDB schema

```sql
delays   (fetched_at, site_id, line_id, line_name, transport_mode, direction, destination, scheduled, expected, delay_minutes)
weather  (timestamp PK, temperature, wind_speed, precipitation, snowfall, cloud_cover)
predictions (timestamp, site_id, line_id, pred_delay, actual_delay, model_version)
retrain_log (run_at, new_rows, retrained, mae)
```

## Azure provisioning (do before Stage 3)

These can be provisioned any time; not required for Stage 2 local ML work:
1. `az group create -n rg-traffik-ml -l swedencentral`
2. `az storage account create -n traffikmlstorage -g rg-traffik-ml --sku Standard_LRS`
3. `az storage container create -n model-backups --account-name traffikmlstorage`
4. `az storage container create -n mlflow --account-name traffikmlstorage`
5. `az keyvault create -n traffik-kv -g rg-traffik-ml -l swedencentral`
6. `az acr create -n traffikmlacr -g rg-traffik-ml --sku Basic`
7. `az containerapp env create -n traffik-env -g rg-traffik-ml -l swedencentral`

## Next: Stage 2 — ML Loop

Start the next conversation with this prompt:

> I'm building the Stockholm transit delay prediction pipeline (traffik-mlpipeline).
> Stage 1 (data foundation) is complete — see STAGE_1_HANDOFF.md.
> Now build Stage 2: transit/features.py, transit/ml.py, transit/retrain.py,
> and the Azure Function timer trigger. Mirror the SE3 project at
> C:\Users\suraj\GitHub\Sthlm-electricity-usage\ for code style.
