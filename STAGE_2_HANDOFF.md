# Stage 2 Handoff — ML Loop

## What was built

Full ML pipeline — features, training, orchestration, Azure Function trigger.

```
transit/
├── features.py    ✓  21 features (calendar, route, weather, lags)
├── ml.py          ✓  LightGBM MAE regression + champion/challenger (±2% threshold)
└── retrain.py     ✓  8-step orchestrator (mirrors SE3 retrain.py)

infra/
└── azure-function/
    └── function_app.py  ✓  Timer triggers: retrain every 3h, digest every Monday 08:00
```

## Verified working locally

```
python -m transit.features      → 21 feature cols, 324 rows
python -m transit.ml --force    → MAE=0.035, model/delay_model.pkl saved, MLflow run logged
```

MLflow run at: sqlite:///mlflow.db  (run `mlflow ui` to browse)

## Model artifacts (model/)

| File | Contents |
|------|----------|
| `delay_model.pkl` | Champion LightGBM model |
| `metrics.json` | `{test_mae, test_rmse, n_test, n_train, n_features}` |
| `config.json` | Feature cols + label encoding maps + LGBM params |

## Key design notes

- **Champion/challenger**: new model promoted only if `new_mae < champion_mae * 0.98`
- **Retrain threshold**: `RETRAIN_MIN_ROWS=50` new delay rows (set in .env to override)
- **Fallback split**: with <7 days of data, falls back to 80/20 row split (will auto-resolve as data accumulates)
- **Lag features**: `lag_24h_mean` is NaN until 24h+ of polling — LightGBM handles NaN natively, no action needed

## Azure (still pending — needed for Stage 3)

Open a terminal and run:
```bash
# After az login (fresh shell needed for CLI to work):
az group create --name rg-traffik-ml --location swedencentral
az storage account create --name traffikmlstorage -g rg-traffik-ml --sku Standard_LRS
az storage container create --name model-backups --account-name traffikmlstorage --auth-mode login
az storage container create --name mlflow --account-name traffikmlstorage --auth-mode login
az keyvault create --name traffik-kv -g rg-traffik-ml --location swedencentral
az acr create --name traffikmlacr -g rg-traffik-ml --sku Basic --admin-enabled true
az containerapp env create --name traffik-env -g rg-traffik-ml --location swedencentral
az functionapp create --name traffik-ingest -g rg-traffik-ml \
  --consumption-plan-location swedencentral --runtime python --runtime-version 3.11 \
  --functions-version 4 --os-type linux --storage-account traffikmlstorage

# Then get storage connection string → add to .env as AZURE_STORAGE_CONNECTION_STRING
az storage account show-connection-string --name traffikmlstorage -g rg-traffik-ml --query connectionString -o tsv
```

## Next: Stage 3 — Serving + Deployment

Start the next conversation with:

> I'm building the Stockholm transit delay prediction pipeline (traffik-mlpipeline).
> Stages 1 + 2 are complete — see STAGE_2_HANDOFF.md.
> Now build Stage 3: api.py (FastAPI), Docker build, push to Azure Container Registry,
> deploy to Azure Container Apps. Mirror SE3 api.py style.
> SE3 reference: C:\Users\suraj\GitHub\Sthlm-electricity-usage\api.py
