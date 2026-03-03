#!/usr/bin/env bash

set -euo pipefail

# Fix GitHub Actions -> GCP Workload Identity Federation IAM permissions.
#
# This script is safe to run multiple times (idempotent).
#
# Usage:
#   bash infra/gcp/scripts/fix_wif_iam_permissions.sh \
#     --project-id miniflow-489011 \
#     --repo KaranjotVendal/MiniFlow
#
# Optional flags:
#   --service-account      Service account email (default: github-actions@<project>.iam.gserviceaccount.com)
#   --pool                 Workload Identity Pool name (default: github-pool)
#   --provider             Workload Identity Provider name (default: github-provider)

PROJECT_ID=""
REPO=""
SERVICE_ACCOUNT=""
POOL="github-pool"
PROVIDER="github-provider"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-id)
      PROJECT_ID="$2"
      shift 2
      ;;
    --repo)
      REPO="$2"
      shift 2
      ;;
    --service-account)
      SERVICE_ACCOUNT="$2"
      shift 2
      ;;
    --pool)
      POOL="$2"
      shift 2
      ;;
    --provider)
      PROVIDER="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,60p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$PROJECT_ID" || -z "$REPO" ]]; then
  echo "Error: --project-id and --repo are required"
  echo "Example:"
  echo "  bash infra/gcp/scripts/fix_wif_iam_permissions.sh --project-id miniflow-489011 --repo KaranjotVendal/MiniFlow"
  exit 1
fi

if [[ -z "$SERVICE_ACCOUNT" ]]; then
  SERVICE_ACCOUNT="github-actions@${PROJECT_ID}.iam.gserviceaccount.com"
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "Error: gcloud CLI is not installed or not in PATH"
  exit 1
fi

echo "Checking active gcloud account..."
gcloud auth list --filter=status:ACTIVE --format='value(account)'

echo "Checking project exists: ${PROJECT_ID}"
gcloud projects describe "$PROJECT_ID" >/dev/null

PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')

echo "Checking service account exists: ${SERVICE_ACCOUNT}"
gcloud iam service-accounts describe "$SERVICE_ACCOUNT" --project="$PROJECT_ID" >/dev/null

echo "Checking Workload Identity Pool exists: ${POOL}"
gcloud iam workload-identity-pools describe "$POOL" \
  --location="global" \
  --project="$PROJECT_ID" >/dev/null

echo "Checking Workload Identity Provider exists: ${PROVIDER}"
gcloud iam workload-identity-pools providers describe "$PROVIDER" \
  --location="global" \
  --workload-identity-pool="$POOL" \
  --project="$PROJECT_ID" >/dev/null

MEMBER="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${REPO}"

add_sa_binding() {
  local role="$1"
  echo "Granting service-account-level role ${role} to ${MEMBER}"
  gcloud iam service-accounts add-iam-policy-binding "$SERVICE_ACCOUNT" \
    --project="$PROJECT_ID" \
    --role="$role" \
    --member="$MEMBER" >/dev/null
}

add_project_binding_to_sa() {
  local role="$1"
  echo "Granting project-level role ${role} to serviceAccount:${SERVICE_ACCOUNT}"
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="$role" >/dev/null
}

# Required for GitHub OIDC principal to impersonate the target service account.
add_sa_binding "roles/iam.workloadIdentityUser"
add_sa_binding "roles/iam.serviceAccountTokenCreator"

# Required by this repository's Terraform deployment workflow.
add_project_binding_to_sa "roles/serviceusage.serviceUsageAdmin"
add_project_binding_to_sa "roles/run.admin"
add_project_binding_to_sa "roles/compute.admin"
add_project_binding_to_sa "roles/iam.serviceAccountUser"

echo ""
echo "Done. Current service account IAM policy:"
gcloud iam service-accounts get-iam-policy "$SERVICE_ACCOUNT" --project="$PROJECT_ID"

echo ""
echo "Workflow secrets should be:"
echo "  GCP_PROJECT_ID=${PROJECT_ID}"
echo "  GCP_SERVICE_ACCOUNT=${SERVICE_ACCOUNT}"
echo "  GCP_WORKLOAD_IDENTITY_PROVIDER=projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}"

echo ""
echo "Next step: re-run Deploy Staging (GCP) workflow."
