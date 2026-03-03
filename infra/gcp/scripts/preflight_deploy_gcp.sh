#!/usr/bin/env bash

set -euo pipefail

# Local preflight for Deploy Staging (GCP)
#
# Purpose:
# - Validate workflow/script syntax
# - Verify GCP auth and project access
# - Verify GHCR image exists
# - Mirror GHCR image to GAR
# - Run Terraform init/validate/plan with GAR image
#
# Usage:
#   bash infra/gcp/scripts/preflight_deploy_gcp.sh \
#     --project-id miniflow-489011 \
#     --region us-central1 \
#     --image-tag sha-18dcfc9
#
# Optional:
#   --owner <github_owner>        (default: karanjotvendal)
#   --gar-repository <repo_name>  (default: miniflow)
#   --deployment-type <cpu|gpu>   (default: cpu)
#   --gpu-zone <zone>             (required when deployment-type=gpu)
#   --gpu-type <gpu_type>         (default: nvidia-tesla-p100)
#
# Requirements:
# - gcloud, docker, terraform, python3, jq installed
# - gcloud authenticated (gcloud auth login)
# - GITHUB_TOKEN exported for GHCR pull

PROJECT_ID=""
REGION="us-central1"
IMAGE_TAG=""
OWNER="karanjotvendal"
GAR_REPOSITORY="miniflow"
DEPLOYMENT_TYPE="cpu"
GPU_ZONE=""
GPU_TYPE="nvidia-tesla-p100"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-id)
      PROJECT_ID="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --image-tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --owner)
      OWNER="$2"
      shift 2
      ;;
    --gar-repository)
      GAR_REPOSITORY="$2"
      shift 2
      ;;
    --deployment-type)
      DEPLOYMENT_TYPE="$2"
      shift 2
      ;;
    --gpu-zone)
      GPU_ZONE="$2"
      shift 2
      ;;
    --gpu-type)
      GPU_TYPE="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$PROJECT_ID" || -z "$IMAGE_TAG" ]]; then
  echo "Error: --project-id and --image-tag are required"
  exit 1
fi

if [[ "$DEPLOYMENT_TYPE" == "gpu" && -z "$GPU_ZONE" ]]; then
  echo "Error: --gpu-zone is required when --deployment-type=gpu"
  exit 1
fi

for cmd in gcloud docker terraform python3 jq; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: missing required command: $cmd"
    exit 1
  fi
done

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Error: GITHUB_TOKEN is not set"
  echo "Export a GitHub token with package read access, then retry"
  exit 1
fi

GHCR_IMAGE="ghcr.io/${OWNER}/miniflow:${IMAGE_TAG}"
GAR_HOST="${REGION}-docker.pkg.dev"
GAR_IMAGE="${GAR_HOST}/${PROJECT_ID}/${GAR_REPOSITORY}/miniflow:${IMAGE_TAG}"
TF_DIR="infra/gcp/terraform"

log() {
  echo "[preflight] $*"
}

log "Workflow YAML parse check"
python3 - <<'PY'
import yaml
with open('.github/workflows/deploy-staging-gcp.yml', 'r', encoding='utf-8') as f:
    yaml.safe_load(f)
print('workflow yaml OK')
PY

log "IAM fix script syntax check"
bash -n infra/gcp/scripts/fix_wif_iam_permissions.sh

log "gcloud account check"
gcloud auth list --filter=status:ACTIVE --format='value(account)'

log "Set gcloud project: ${PROJECT_ID}"
gcloud config set project "${PROJECT_ID}" >/dev/null

log "Enable required APIs"
gcloud services enable \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  compute.googleapis.com \
  serviceusage.googleapis.com \
  --project="${PROJECT_ID}" >/dev/null

log "Ensure GAR repository exists: ${GAR_REPOSITORY}"
if ! gcloud artifacts repositories describe "${GAR_REPOSITORY}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${GAR_REPOSITORY}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="MiniFlow staging container images" \
    --project="${PROJECT_ID}" >/dev/null
fi

log "Pull image from GHCR: ${GHCR_IMAGE}"
echo "${GITHUB_TOKEN}" | docker login ghcr.io -u "${OWNER}" --password-stdin >/dev/null

docker pull "${GHCR_IMAGE}" >/dev/null

log "Configure docker auth for GAR: ${GAR_HOST}"
gcloud auth configure-docker "${GAR_HOST}" --quiet >/dev/null

log "Mirror image to GAR: ${GAR_IMAGE}"
docker tag "${GHCR_IMAGE}" "${GAR_IMAGE}"
docker push "${GAR_IMAGE}" >/dev/null

log "Terraform init/validate/plan"
pushd "${TF_DIR}" >/dev/null
terraform init -input=false >/dev/null
terraform validate >/dev/null

if [[ "$DEPLOYMENT_TYPE" == "cpu" ]]; then
  terraform plan \
    -var "project_id=${PROJECT_ID}" \
    -var "region=${REGION}" \
    -var "deployment_type=cpu" \
    -var "container_image=${GAR_IMAGE}" \
    -var "miniflow_device=cpu" >/dev/null
else
  terraform plan \
    -var "project_id=${PROJECT_ID}" \
    -var "region=${REGION}" \
    -var "deployment_type=gpu" \
    -var "gpu_zone=${GPU_ZONE}" \
    -var "gpu_type=${GPU_TYPE}" \
    -var "container_image=${GAR_IMAGE}" \
    -var "miniflow_device=cuda" >/dev/null
fi
popd >/dev/null

log "Preflight success"
log "Deployment image ready in GAR: ${GAR_IMAGE}"
log "You can now trigger Deploy Staging (GCP) with image_tag=${IMAGE_TAG}"
