#!/usr/bin/env bash
# Cloud Run deployment script.
# Usage: bash scripts/deploy.sh
# Prerequisites: gcloud CLI installed and authenticated (see M6 in README).

set -euo pipefail

# ── configuration ────────────────────────────────────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID env var}"
GITHUB_USER="${GITHUB_USER:?Set GITHUB_USER env var}"
REPO_NAME="${REPO_NAME:-pdm-lite}"
REGION="europe-west1"
SERVICE_NAME="pdm-lite"
IMAGE="ghcr.io/${GITHUB_USER}/${REPO_NAME}:latest"

# ── deploy ────────────────────────────────────────────────────────────────────
echo "Deploying ${IMAGE} → Cloud Run (${REGION})"

gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --port 8000 \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 2 \
  --timeout 30

echo ""
echo "Live URL:"
gcloud run services describe "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --format "value(status.url)"
