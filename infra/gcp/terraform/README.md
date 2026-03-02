# MiniFlow GCP Staging (Terraform)

Terraform configuration for deploying MiniFlow to Google Cloud Platform using Cloud Run.

---

## Overview

This module deploys MiniFlow as a Cloud Run service on GCP. Cloud Run provides:
- Serverless container hosting
- Automatic scaling (including to zero)
- Built-in load balancing and SSL
- Pay-per-use pricing

---

## Prerequisites

1. **GCP Project**: Create a project at https://console.cloud.google.com/
2. **Billing**: Enable billing (required even for free tier)
3. **Container Image**: Image must be pushed to GHCR or GCR

---

## Quick Start

### 1. Authenticate GCP

```bash
# Install gcloud CLI: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth application-default login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### 2. Deploy

```bash
cd infra/gcp/terraform

# Initialize
terraform init

# Plan
terraform plan \
  -var "project_id=YOUR_PROJECT_ID" \
  -var "container_image=ghcr.io/karanjotvendal/miniflow:sha-XXXXXXX"

# Apply
terraform apply \
  -var "project_id=YOUR_PROJECT_ID" \
  -var "container_image=ghcr.io/karanjotvendal/miniflow:sha-XXXXXXX"
```

### 3. Test Deployment

```bash
# Get the service URL
export SERVICE_URL=$(terraform output -raw service_url)

echo "Service URL: $SERVICE_URL"

# Test endpoints
curl $SERVICE_URL/health
curl $SERVICE_URL/ready
```

### 4. Cleanup

```bash
# Destroy all resources (stops billing)
terraform destroy
```

---

## Configuration

### Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `project_id` | GCP project ID | - | Yes |
| `region` | GCP region | `us-central1` | No |
| `container_image` | Container image URL | - | Yes |
| `service_name` | Cloud Run service name | `miniflow` | No |
| `cpu` | CPU allocation | `2` | No |
| `memory` | Memory allocation | `4Gi` | No |
| `miniflow_config` | Config file path | `configs/3_TTS-to-vibevoice.yml` | No |
| `miniflow_device` | Device (cpu/cuda) | `cpu` | No |
| `release_id` | Release identifier | `gcp-staging` | No |

### Example: terraform.tfvars

```hcl
project_id = "my-project-123456"
region     = "us-central1"

container_image = "ghcr.io/karanjotvendal/miniflow:sha-abc1234"

cpu    = "2"
memory = "4Gi"

miniflow_device = "cpu"
release_id      = "gcp-staging-v1"
```

---

## Architecture

```
Internet
    │
    ▼
┌─────────────────────────────────────┐
│  Cloud Run                          │
│  ┌───────────────────────────────┐  │
│  │  Container: miniflow          │  │
│  │  ├─ Port: 8000               │  │
│  │  ├─ CPU: 2                   │  │
│  │  ├─ Memory: 4Gi              │  │
│  │  └─ Auto-scaling: 0-5        │  │
│  └───────────────────────────────┘  │
│                                     │
│  Features:                          │
│  ├─ HTTPS endpoint (auto-SSL)      │
│  ├─ Load balancing                 │
│  ├─ Auto-scaling                   │
│  └─ Cloud Logging                  │
└─────────────────────────────────────┘
```

---

## Cost Estimation

### Cloud Run Pricing (approximate)

| Resource | Free Tier | Paid Tier |
|----------|-----------|-----------|
| **CPU** | 180,000 vCPU-seconds/month | $0.00002400/vCPU-second |
| **Memory** | 360,000 GiB-seconds/month | $0.00000250/GiB-second |
| **Requests** | 2 million/month | $0.40/million |

### MiniFlow Staging Estimate

Assuming:
- 1 container running continuously
- 2 vCPU, 4Gi memory
- Light traffic (testing only)

**Cost: $0-10/month**

- Mostly covered by free tier
- If exceeding: ~$30-50/month for 24/7
- Scale to zero when not testing: ~$0

---

## Scaling Configuration

### Auto-scaling Behavior

```hcl
# In main.tf
metadata {
  annotations = {
    "autoscaling.knative.dev/minScale" = "0"  # Scale to zero
    "autoscaling.knative.dev/maxScale" = "5"  # Max 5 instances
  }
}
```

| Setting | Behavior |
|---------|----------|
| `minScale = 0` | Scales to zero when no traffic (saves money) |
| `minScale = 1` | Always keeps 1 instance warm (faster response) |
| `maxScale = 5` | Never exceeds 5 instances (cost control) |

### Container Concurrency

```hcl
# In main.tf
spec {
  container_concurrency = 10  # 10 requests per container
}
```

- Each container handles up to 10 concurrent requests
- Additional requests spin up new containers
- Adjust based on your app's concurrency capabilities

---

## Monitoring

### View Logs

```bash
# Stream logs
gcloud logging tail "run.googleapis.com%2Fminiflow" --format="value(textPayload)"

# Or in Console: Cloud Logging → Logs Explorer
```

### View Metrics

```bash
# Console: Cloud Monitoring → Metrics Explorer
# Metrics: run.googleapis.com/container/cpu/utilizations
```

---

## Troubleshooting

### Issue: Service not deploying

```bash
# Check logs
terraform apply  # Will show errors

# Check Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=miniflow"
```

### Issue: Image not found

```bash
# Verify image exists
docker pull ghcr.io/karanjotvendal/miniflow:sha-XXXXXXX

# Or use GCR (Google Container Registry)
# gcr.io/PROJECT_ID/miniflow:tag
```

### Issue: Out of memory

```bash
# Increase memory
terraform apply -var "memory=8Gi"
```

### Issue: Slow cold start

```bash
# Set minScale to 1 to keep instance warm
# Edit main.tf:
# "autoscaling.knative.dev/minScale" = "1"
```

---

## GitHub Actions Deployment

### Setup

1. **Configure Workload Identity Federation**:
   ```bash
   # Create service account
   gcloud iam service-accounts create github-actions \
     --display-name="GitHub Actions"

   # Grant permissions
   gcloud projects add-iam-policy-binding PROJECT_ID \
     --member="serviceAccount:github-actions@PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/run.admin"

   gcloud projects add-iam-policy-binding PROJECT_ID \
     --member="serviceAccount:github-actions@PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/iam.serviceAccountUser"
   ```

2. **Configure Workload Identity Pool** (for OIDC):
   ```bash
   # Follow: https://github.com/google-github-actions/auth#setup
   ```

3. **Add GitHub Secrets**:
   - `GCP_PROJECT_ID`: Your project ID
   - `GCP_WORKLOAD_IDENTITY_PROVIDER`: Workload identity provider resource name
   - `GCP_SERVICE_ACCOUNT`: Service account email

### Trigger Deployment

```bash
# Via GitHub UI
# Actions → Deploy Staging (GCP) → Run workflow

# Via CLI
gh workflow run deploy-staging-gcp.yml -f image_tag=sha-abc1234
```

---

## Comparison with AWS

| Feature | AWS ECS Fargate | GCP Cloud Run |
|---------|-----------------|---------------|
| **Setup complexity** | High (15+ resources) | Low (3 resources) |
| **Load balancer** | Separate (ALB) | Built-in |
| **SSL** | Manual (ACM) | Automatic |
| **Scaling** | Configurable | Automatic |
| **Scale to zero** | No | Yes |
| **Pricing** | Per task/hour | Per request + compute time |
| **GPU support** | Yes (ECS on EC2) | No (use GCE for GPU) |

---

## Next Steps

1. **Test locally**: `docker-compose up`
2. **Push image**: Let CI/CD push to GHCR
3. **Deploy to GCP**: Run GitHub Actions workflow
4. **Test endpoints**: Verify `/health` and `/ready`
5. **Monitor**: Check Cloud Logging
6. **Iterate**: Deploy new versions with new image tags

---

## Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud Run Pricing](https://cloud.google.com/run/pricing)
- [Terraform Google Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [GitHub Actions for GCP](https://github.com/google-github-actions)
