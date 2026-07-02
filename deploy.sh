#!/usr/bin/env bash
# deploy.sh — Build, push to GitHub Container Registry, deploy to Azure Container Apps
#
# Usage:
#   ./deploy.sh              # build + push + deploy (create or update)
#   ./deploy.sh --build-only # build + push, skip deploy
#   ./deploy.sh --deploy-only # skip build, just re-deploy latest image
#
# Prerequisites:
#   az login              (done once per session)
#   docker running        (Docker Desktop)
#   GITHUB_TOKEN env var  (PAT with write:packages + read:packages scope)
#
# Env vars that override defaults:
#   GITHUB_USER, GITHUB_TOKEN, CONTAINER_APP_NAME, CONTAINER_APP_ENV,
#   RESOURCE_GROUP, LOCATION, IMAGE_TAG,
#   TRAFFIK_DB_PATH, MODEL_DIR, SL_API_KEY, AZURE_STORAGE_CONNECTION_STRING

set -euo pipefail

# ── config ─────────────────────────────────────────────────────────────────────
GITHUB_USER="${GITHUB_USER:-surajsingh108}"
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-traffik-ml}"
CONTAINER_APP_NAME="${CONTAINER_APP_NAME:-traffik-api}"
CONTAINER_APP_ENV="${CONTAINER_APP_ENV:-traffik-env}"
LOCATION="${LOCATION:-swedencentral}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

REGISTRY="ghcr.io"
IMAGE_NAME="traffik-api"
FULL_IMAGE="${REGISTRY}/${GITHUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

BUILD_ONLY=false
DEPLOY_ONLY=false

for arg in "$@"; do
  case $arg in
    --build-only)  BUILD_ONLY=true  ;;
    --deploy-only) DEPLOY_ONLY=true ;;
  esac
done

echo "=== traffik-mlpipeline Stage 3 deploy ==="
echo "  Registry:        ${REGISTRY}"
echo "  Image:           ${FULL_IMAGE}"
echo "  Container App:   ${CONTAINER_APP_NAME}"
echo "  Environment:     ${CONTAINER_APP_ENV}"
echo "  Resource Group:  ${RESOURCE_GROUP}"
echo ""

# ── validate GITHUB_TOKEN ──────────────────────────────────────────────────────

if [ "$DEPLOY_ONLY" = false ]; then
  if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "ERROR: GITHUB_TOKEN is not set."
    echo "  Create a PAT at https://github.com/settings/tokens with scopes:"
    echo "    write:packages   read:packages"
    echo "  Then: export GITHUB_TOKEN=<your-token>"
    exit 1
  fi
fi

# ── build + push ───────────────────────────────────────────────────────────────

if [ "$DEPLOY_ONLY" = false ]; then
  echo "▶ Logging in to ghcr.io …"
  echo "${GITHUB_TOKEN}" | docker login ghcr.io -u "${GITHUB_USER}" --password-stdin

  echo "▶ Building Docker image …"
  docker build \
    --platform linux/amd64 \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -t "${FULL_IMAGE}" \
    .

  echo "▶ Pushing to ghcr.io …"
  docker push "${FULL_IMAGE}"
  echo "✓ Image pushed: ${FULL_IMAGE}"
fi

if [ "$BUILD_ONLY" = true ]; then
  echo "✓ Build-only mode — skipping deploy."
  exit 0
fi

# ── validate GITHUB_TOKEN for registry pull ────────────────────────────────────

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "ERROR: GITHUB_TOKEN is required for Container Apps to pull from ghcr.io"
  exit 1
fi

# ── collect app env vars ───────────────────────────────────────────────────────

SL_API_KEY="${SL_API_KEY:-}"
AZURE_STORAGE_CONNECTION_STRING="${AZURE_STORAGE_CONNECTION_STRING:-}"
OPEN_METEO_KEY="${OPEN_METEO_KEY:-}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
GROQ_API_KEY="${GROQ_API_KEY:-}"

[ -z "$SL_API_KEY" ]                      && echo "⚠  SL_API_KEY not set — departure fetch will fail in container"
[ -z "$AZURE_STORAGE_CONNECTION_STRING" ] && echo "⚠  AZURE_STORAGE_CONNECTION_STRING not set — model backups will fail"

ENV_VARS=(
  "TRAFFIK_DB_PATH=${TRAFFIK_DB_PATH:-/app/data/traffik.duckdb}"
  "MODEL_DIR=${MODEL_DIR:-/app/model}"
)
[ -n "$SL_API_KEY" ]                       && ENV_VARS+=("SL_API_KEY=${SL_API_KEY}")
[ -n "$AZURE_STORAGE_CONNECTION_STRING" ]  && ENV_VARS+=("AZURE_STORAGE_CONNECTION_STRING=${AZURE_STORAGE_CONNECTION_STRING}")
[ -n "$OPEN_METEO_KEY" ]                   && ENV_VARS+=("OPEN_METEO_KEY=${OPEN_METEO_KEY}")
[ -n "$SLACK_WEBHOOK_URL" ]                && ENV_VARS+=("SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL}")
[ -n "$GROQ_API_KEY" ]                     && ENV_VARS+=("GROQ_API_KEY=${GROQ_API_KEY}")

# ── deploy to Container Apps ───────────────────────────────────────────────────

EXISTING=$(az containerapp show \
  --name "${CONTAINER_APP_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --query "name" -o tsv 2>/dev/null || true)

if [ -z "$EXISTING" ]; then
  echo "▶ Creating Container App '${CONTAINER_APP_NAME}' …"
  az containerapp create \
    --name "${CONTAINER_APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --environment "${CONTAINER_APP_ENV}" \
    --image "${FULL_IMAGE}" \
    --registry-server "${REGISTRY}" \
    --registry-username "${GITHUB_USER}" \
    --registry-password "${GITHUB_TOKEN}" \
    --target-port 8000 \
    --ingress external \
    --min-replicas 0 \
    --max-replicas 2 \
    --cpu 0.5 \
    --memory 1.0Gi \
    --set-env-vars "${ENV_VARS[@]}"
else
  echo "▶ Updating Container App '${CONTAINER_APP_NAME}' to ${IMAGE_TAG} …"
  az containerapp update \
    --name "${CONTAINER_APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --image "${FULL_IMAGE}" \
    --set-env-vars "${ENV_VARS[@]}"
fi

# ── print public URL ───────────────────────────────────────────────────────────

echo ""
FQDN=$(az containerapp show \
  --name "${CONTAINER_APP_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --query "properties.configuration.ingress.fqdn" -o tsv)
echo "✓ Deployed: https://${FQDN}"
echo "  Health:   https://${FQDN}/health"
echo "  Docs:     https://${FQDN}/docs"
