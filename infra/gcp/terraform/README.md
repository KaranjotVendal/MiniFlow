# GCP Infrastructure for MiniFlow

This directory contains Terraform configurations for deploying MiniFlow on Google Cloud Platform.

## Architecture

Two deployment options are available:

### 1. Cloud Run (CPU)
- **Best for**: Learning, testing, development
- **Pros**: Instant deployment, scales to zero, very low cost
- **Cons**: 10x slower inference, no GPU acceleration
- **Cost**: ~$0-5/month (scales to zero)

### 2. Compute Engine with GPU
- **Best for**: Production inference, demos, benchmarks
- **Pros**: GPU acceleration, full control, persistent storage
- **Cons**: More expensive, requires GPU availability
- **Cost**: ~$0.50-0.80/hour (P100) or ~$0.35/hour with preemptible

## Quick Start

### Find Available GPU Zone

Since GPU availability varies by zone and time, use the helper script:

```bash
# Find zone with P100 (recommended)
./find-gpu-zone.sh

# Find zone with T4
./find-gpu-zone.sh miniflow-489011 n1-standard-4 nvidia-tesla-t4 1

# Find zone with V100
./find-gpu-zone.sh miniflow-489011 n1-standard-8 nvidia-tesla-v100 1
```

### Deploy Cloud Run (CPU)

```bash
cd infra/gcp/terraform
terraform init
terraform plan \
  -var="project_id=miniflow-489011" \
  -var="deployment_type=cpu" \
  -var="container_image=ghcr.io/youruser/miniflow:main"

terraform apply \
  -var="project_id=miniflow-489011" \
  -var="deployment_type=cpu" \
  -var="container_image=ghcr.io/youruser/miniflow:main"
```

### Deploy with GPU

```bash
cd infra/gcp/terraform

# First, find available zone
ZONE=$(./find-gpu-zone.sh | grep "export GPU_ZONE" | cut -d'"' -f2)

# Deploy with GPU (only provisions GPU instance, not Cloud Run)
terraform plan \
  -var="project_id=miniflow-489011" \
  -var="deployment_type=gpu" \
  -var="gpu_zone=$ZONE" \
  -var="gpu_type=nvidia-tesla-p100" \
  -var="container_image=ghcr.io/youruser/miniflow:main" \
  -var="miniflow_device=cuda"

terraform apply \
  -var="project_id=miniflow-489011" \
  -var="deployment_type=gpu" \
  -var="gpu_zone=$ZONE" \
  -var="gpu_type=nvidia-tesla-p100" \
  -var="container_image=ghcr.io/youruser/miniflow:main" \
  -var="miniflow_device=cuda"
```

### Conditional Deployment

Terraform now only provisions the resources you need:

| deployment_type | Provisions |
|----------------|------------|
| `cpu` | Cloud Run only |
| `gpu` | Compute Engine GPU only |

Note: The GitHub Actions workflow supports an `auto` mode that automatically
selects GPU if available, otherwise falls back to CPU. Terraform receives
either `cpu` or `gpu` from the workflow (never `auto`).

This ensures:
- No orphaned resources
- Faster deployment
- Lower cost (no unnecessary resources)

## GitHub Actions Deployment

The workflow supports three deployment modes:

### Manual Trigger Options

| Option | Description |
|--------|-------------|
| **cpu** | Deploy to Cloud Run (always works) |
| **gpu** | Deploy to Compute Engine with GPU (requires available zone) |
| **auto** | Try GPU first, fallback to CPU if unavailable |

### Usage

1. Go to **Actions** → **Deploy Staging (GCP)**
2. Click **Run workflow**
3. Select deployment type:
   - `cpu` - Guaranteed to work
   - `gpu` - Needs available GPU zone
   - `auto` - Best of both worlds
4. Enter image tag (e.g., `sha-abc1234` or `main`)
5. (Optional) Select GPU type for GPU deployments
6. Click **Run workflow**

## GPU Types Comparison

| GPU | Memory | Relative Speed | Cost/hour | Availability |
|-----|--------|----------------|-----------|--------------|
| **P100** | 16GB | Baseline | ~$0.50 | ⭐⭐⭐⭐⭐ |
| **T4** | 16GB | 1.3x | ~$0.54 | ⭐⭐⭐ |
| **V100** | 16GB | 2.0x | ~$1.20 | ⭐⭐⭐ |
| **A100** | 40GB | 4.0x | ~$2.50 | ⭐⭐ |

**Recommendation**: P100 offers the best balance of performance, cost, and availability for MiniFlow.

## Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `project_id` | GCP project ID | Required |
| `region` | GCP region | `us-central1` |
| `deployment_type` | cpu, gpu, or auto | `cpu` |
| `container_image` | Container image URL | Required |
| `gpu_zone` | Zone with GPU availability | `us-central1-b` |
| `gpu_type` | GPU accelerator type | `nvidia-tesla-p100` |
| `gpu_count` | Number of GPUs | `1` |
| `miniflow_device` | Device (cpu/cuda) | `cpu` |
| `miniflow_config` | Config file path | `configs/3_TTS-to-vibevoice.yml` |

## Outputs

| Output | Description |
|--------|-------------|
| `service_url` | Cloud Run service URL |
| `gpu_instance_url` | GPU instance URL |
| `gpu_instance_public_ip` | GPU instance IP |
| `gpu_zone_used` | Zone where GPU was deployed |

## Troubleshooting

### ZONE_RESOURCE_POOL_EXHAUSTED
GPU capacity is full in that zone. The zone finder will automatically try other zones.

### QUOTA_EXCEEDED
Request quota increase at: https://console.cloud.google.com/iam-admin/quotas

### GPU not supported in zone
Some zones don't support certain GPU types. The zone finder handles this.

## Costs

### Cloud Run (CPU)
- Free tier: 2M requests/month, 360,000 vCPU-seconds, 180,000 GiB-seconds
- Beyond free: ~$0.00002400/vCPU-second, ~$0.00000250/GiB-second

### Compute Engine with GPU
- P100 preemptible: ~$0.35/hour (~$8.40/day)
- P100 standard: ~$0.50/hour (~$12/day)
- Includes: 4 vCPU, 15GB RAM, 50GB SSD

**With $300 credit:**
- GPU demo for 3-4 days: ~$25-50
- CPU-only for months: ~$0-20
