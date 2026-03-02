# Infrastructure Changes: AWS to GCP Migration

A complete guide to what code and infrastructure changes are needed to move MiniFlow from AWS to GCP.

---

## Table of Contents

1. [High-Level Changes Overview](#high-level-changes-overview)
2. [GitHub Actions Workflow Changes](#github-actions-workflow-changes)
3. [Terraform Changes](#terraform-changes)
4. [Configuration Changes](#configuration-changes)
5. [File-by-File Comparison](#file-by-file-comparison)
6. [Effort Estimation](#effort-estimation)
7. [Maintaining Both Infrastructures](#maintaining-both-infrastructures)

---

## High-Level Changes Overview

### What Stays the Same (No Changes)

| Component | Status | Why |
|-----------|--------|-----|
| **Dockerfile** | ✅ No change | Container is cloud-agnostic |
| **Application code** | ✅ No change | MiniFlow doesn't know where it runs |
| **docker-compose.yml** | ✅ No change | Local development unchanged |
| **CI workflow (lint/test)** | ✅ No change | GitHub Actions is platform-agnostic |
| **CD workflow (build/push)** | ✅ No change | Still pushes to GHCR |
| **Pre-commit hooks** | ✅ No change | Code quality checks |
| **Documentation** | ⚠️ Update paths | Minor updates only |

### What Changes

| Component | Change Type | Effort |
|-----------|-------------|--------|
| **Deploy workflow** | New file | 1 hour |
| **Terraform directory** | New structure | 2-3 hours |
| **Terraform provider** | AWS → Google | 30 minutes |
| **Networking resources** | Different names | 1 hour |
| **Compute resources** | ECS → Cloud Run/GCE | 1-2 hours |
| **Secrets** | AWS → GCP credentials | 30 minutes |
| **Documentation** | Add GCP paths | 1 hour |

---

## GitHub Actions Workflow Changes

### Current: AWS Deploy Workflow

```yaml
# .github/workflows/deploy-staging.yml (AWS)
name: Deploy Staging

on:
  workflow_dispatch:
    inputs:
      image_tag:
        description: "Image tag to deploy"
        required: true

env:
  AWS_REGION: us-east-1
  TF_DIR: infra/aws/terraform

permissions:
  contents: read
  id-token: write  # For AWS OIDC

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-region: ${{ env.AWS_REGION }}
          role-to-assume: ${{ secrets.AWS_DEPLOY_ROLE_ARN }}

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3

      - name: Terraform init
        working-directory: ${{ env.TF_DIR }}
        run: terraform init

      - name: Terraform apply
        working-directory: ${{ env.TF_DIR }}
        run: |
          terraform apply -auto-approve \
            -var "container_image=ghcr.io/${{ github.repository_owner }}/miniflow:${{ github.event.inputs.image_tag }}"

      - name: Health checks
        run: |
          ALB_DNS=$(terraform output -raw alb_dns_name)
          curl --fail http://${ALB_DNS}/health
```

### New: GCP Deploy Workflow

```yaml
# .github/workflows/deploy-staging-gcp.yml (NEW FILE)
name: Deploy Staging (GCP)

on:
  workflow_dispatch:
    inputs:
      image_tag:
        description: "Image tag to deploy"
        required: true

env:
  GCP_PROJECT_ID: your-project-id
  GCP_REGION: us-central1
  TF_DIR: infra/gcp/terraform

permissions:
  contents: read
  id-token: write  # For GCP OIDC

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
          service_account: ${{ secrets.GCP_SERVICE_ACCOUNT }}

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3

      - name: Terraform init
        working-directory: ${{ env.TF_DIR }}
        run: terraform init

      - name: Terraform apply
        working-directory: ${{ env.TF_DIR }}
        run: |
          terraform apply -auto-approve \
            -var "project_id=${{ env.GCP_PROJECT_ID }}" \
            -var "region=${{ env.GCP_REGION }}" \
            -var "container_image=ghcr.io/${{ github.repository_owner }}/miniflow:${{ github.event.inputs.image_tag }}"

      - name: Get endpoint
        id: get-url
        working-directory: ${{ env.TF_DIR }}
        run: |
          echo "url=$(terraform output -raw service_url)" >> $GITHUB_OUTPUT

      - name: Health checks
        run: |
          curl --fail "${{ steps.get-url.outputs.url }}/health"
```

**Key Differences:**

| Aspect | AWS | GCP |
|--------|-----|-----|
| **Authentication** | `aws-actions/configure-aws-credentials` | `google-github-actions/auth` |
| **Identity** | IAM Role ARN | Workload Identity Provider + Service Account |
| **Terraform vars** | `aws_region`, `vpc_id`, `subnet_ids` | `project_id`, `region`, `zone` |
| **Output** | `alb_dns_name` | `service_url` or `instance_ip` |
| **Health check** | HTTP (no SSL) | HTTPS (Cloud Run auto-SSL) |

---

## Terraform Changes

### Directory Structure Changes

```
Current AWS Structure:
infra/
└── aws/
    └── terraform/
        ├── main.tf
        ├── variables.tf
        ├── outputs.tf
        ├── providers.tf
        └── versions.tf

New GCP Structure:
infra/
├── aws/                    # Keep existing
│   └── terraform/
└── gcp/                    # NEW
    └── terraform/
        ├── main.tf
        ├── variables.tf
        ├── outputs.tf
        ├── providers.tf
        └── versions.tf
```

### Option 1: GCP Cloud Run (Simplest)

```hcl
# infra/gcp/terraform/providers.tf
terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
```

```hcl
# infra/gcp/terraform/variables.tf
variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "container_image" {
  description = "Container image URL"
  type        = string
}

variable "cpu" {
  description = "CPU allocation"
  type        = string
  default     = "2"
}

variable "memory" {
  description = "Memory allocation"
  type        = string
  default     = "4Gi"
}
```

```hcl
# infra/gcp/terraform/main.tf
# Enable required APIs
resource "google_project_service" "run" {
  service = "run.googleapis.com"
}

resource "google_project_service" "container_registry" {
  service = "containerregistry.googleapis.com"
}

# Deploy Cloud Run service
resource "google_cloud_run_service" "miniflow" {
  name     = "miniflow"
  location = var.region

  template {
    spec {
      containers {
        image = var.container_image
        
        resources {
          limits = {
            cpu    = var.cpu
            memory = var.memory
          }
        }

        ports {
          container_port = 8000
        }

        env {
          name  = "MINIFLOW_DEVICE"
          value = "cpu"
        }
        env {
          name  = "COQUI_TOS_AGREED"
          value = "1"
        }
        env {
          name  = "MINIFLOW_CONFIG"
          value = "configs/3_TTS-to-vibevoice.yml"
        }
        env {
          name  = "RELEASE_ID"
          value = "gcp-staging"
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [google_project_service.run]
}

# Allow public access
resource "google_cloud_run_service_iam_member" "public" {
  service  = google_cloud_run_service.miniflow.name
  location = google_cloud_run_service.miniflow.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
```

```hcl
# infra/gcp/terraform/outputs.tf
output "service_url" {
  description = "URL of the deployed service"
  value       = google_cloud_run_service.miniflow.status[0].url
}
```

**Comparison: AWS (15 resources) vs GCP Cloud Run (3 resources)**

| Resource | AWS | GCP Cloud Run |
|----------|-----|---------------|
| VPC | ✅ Required | ❌ Not needed |
| Subnets | ✅ 2+ required | ❌ Not needed |
| Internet Gateway | ✅ Required | ❌ Not needed |
| Security Groups | ✅ 2 required | ❌ Built-in |
| ALB | ✅ Required | ❌ Built-in |
| Target Group | ✅ Required | ❌ Built-in |
| ECS Cluster | ✅ Required | ❌ Not needed |
| ECS Task Definition | ✅ Required | ❌ Built-in |
| ECS Service | ✅ Required | ✅ Cloud Run service |
| IAM Roles | ✅ Required | ✅ Auto-created |
| CloudWatch Logs | ✅ Required | ✅ Built-in (Cloud Logging) |
| **Total resources** | **~15** | **~3** |

### Option 2: GCP Compute Engine (GPU - Similar to AWS EC2)

```hcl
# infra/gcp/terraform/main.tf (GPU version)

# Firewall rule (like AWS Security Group)
resource "google_compute_firewall" "allow_miniflow" {
  name    = "allow-miniflow"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8000", "22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["miniflow"]
}

# VM instance with GPU
resource "google_compute_instance" "miniflow" {
  name         = "miniflow-gpu"
  machine_type = "n1-standard-4"
  zone         = "us-central1-a"
  tags         = ["miniflow"]

  boot_disk {
    initialize_params {
      image = "cos-cloud/cos-stable"  # Container-Optimized OS
      size  = 50
    }
  }

  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }

  network_interface {
    network = "default"
    access_config {}  # Ephemeral public IP
  }

  metadata = {
    gce-container-declaration = <<EOF
spec:
  containers:
    - name: miniflow
      image: ${var.container_image}
      env:
        - name: MINIFLOW_DEVICE
          value: "cuda"
        - name: COQUI_TOS_AGREED
          value: "1"
        - name: MINIFLOW_CONFIG
          value: "configs/3_TTS-to-vibevoice.yml"
        - name: RELEASE_ID
          value: "gcp-gpu"
      resources:
        limits:
          nvidia.com/gpu: 1
  restartPolicy: Always
EOF
  }

  scheduling {
    on_host_maintenance = "TERMINATE"  # Required for GPU
    preemptible         = true          # 70% cheaper (Spot equivalent)
  }
}
```

---

## Configuration Changes

### GitHub Secrets

**Current AWS Secrets:**
```
AWS_DEPLOY_ROLE_ARN          # IAM role for OIDC
STAGING_VPC_ID              # VPC identifier
STAGING_PUBLIC_SUBNET_IDS   # Subnet list
BUDGET_ALERT_EMAIL          # Notification email
```

**New GCP Secrets:**
```
GCP_WORKLOAD_IDENTITY_PROVIDER  # Format: projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/POOL/providers/PROVIDER
GCP_SERVICE_ACCOUNT            # Format: miniflow-deploy@PROJECT_ID.iam.gserviceaccount.com
GCP_PROJECT_ID                 # Your project ID
```

### Terraform Backend (Optional)

**Current (local state):**
```hcl
# No backend configuration (local terraform.tfstate)
```

**Optional - GCS backend for GCP:**
```hcl
# infra/gcp/terraform/backend.tf (optional)
terraform {
  backend "gcs" {
    bucket = "miniflow-terraform-state"
    prefix = "staging"
  }
}
```

---

## File-by-File Comparison

### Complete Side-by-Side

| File | AWS Path | GCP Path | Changes |
|------|----------|----------|---------|
| **Deploy workflow** | `.github/workflows/deploy-staging.yml` | `.github/workflows/deploy-staging-gcp.yml` | New file |
| **Terraform main** | `infra/aws/terraform/main.tf` | `infra/gcp/terraform/main.tf` | Complete rewrite |
| **Terraform variables** | `infra/aws/terraform/variables.tf` | `infra/gcp/terraform/variables.tf` | Different vars |
| **Terraform outputs** | `infra/aws/terraform/outputs.tf` | `infra/gcp/terraform/outputs.tf` | Different outputs |
| **Terraform providers** | `infra/aws/terraform/providers.tf` | `infra/gcp/terraform/providers.tf` | AWS → Google provider |
| **Terraform versions** | `infra/aws/terraform/versions.tf` | `infra/gcp/terraform/versions.tf` | AWS → Google constraints |
| **Terraform README** | `infra/aws/terraform/README.md` | `infra/gcp/terraform/README.md` | GCP-specific instructions |
| **Docs** | `docs/learning/aws-foundations.md` | (existing) | Reference both |

### Code Diff: Resource Declaration

**AWS ECS Task Definition:**
```hcl
resource "aws_ecs_task_definition" "miniflow" {
  family                   = "miniflow"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "2048"
  memory                   = "4096"
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn

  container_definitions = jsonencode([{
    name  = "api"
    image = var.container_image
    portMappings = [{ containerPort = 8000 }]
    # ... more config
  }])
}
```

**GCP Cloud Run Service:**
```hcl
resource "google_cloud_run_service" "miniflow" {
  name     = "miniflow"
  location = var.region

  template {
    spec {
      containers {
        image = var.container_image
        ports { container_port = 8000 }
        resources {
          limits = { cpu = "2", memory = "4Gi" }
        }
        # ... env vars
      }
    }
  }
}
```

**Key Syntax Differences:**

| Aspect | AWS | GCP |
|--------|-----|-----|
| **Resource type** | `aws_ecs_task_definition` | `google_cloud_run_service` |
| **Container image** | Inside JSON | Direct property |
| **Ports** | `portMappings` list | `ports` block |
| **Resources** | `cpu`, `memory` as strings | `limits` map |
| **JSON encoding** | Required | Not needed |

---

## Effort Estimation

### Creating GCP Infrastructure from Scratch

| Task | Time | Complexity |
|------|------|------------|
| Create GitHub workflow | 1 hour | Low |
| Create Terraform structure | 30 min | Low |
| Write GCP Terraform (Cloud Run) | 1 hour | Medium |
| Test deployment | 2 hours | Medium |
| Debug issues | 1-2 hours | Medium |
| Update documentation | 1 hour | Low |
| **Total** | **6-8 hours** | **Medium** |

### If You've Already Done AWS

| Task | Time | Notes |
|------|------|-------|
| Copy AWS workflow | 15 min | Modify auth and paths |
| Copy Terraform structure | 15 min | Change provider |
| Translate resources | 2-3 hours | AWS resource → GCP equivalent |
| Test | 1 hour | Should be similar |
| **Total** | **4-5 hours** | **Faster with experience** |

---

## Maintaining Both Infrastructures

### Recommended: Keep Both

```
infra/
├── aws/
│   └── terraform/          # Keep for AWS deployment
├── gcp/
│   └── terraform/          # Add for GCP deployment
└── shared/
    └── modules/            # Optional: reusable modules
```

### Workflow Strategy

```yaml
# .github/workflows/
├── cd.yml                  # Build and push image (agnostic)
├── deploy-aws.yml          # Deploy to AWS (when needed)
└── deploy-gcp.yml          # Deploy to GCP (default for learning)
```

### Conditional Deployment

```yaml
# Example: Deploy to both for testing
name: Deploy to Both Clouds

on:
  workflow_dispatch:

jobs:
  deploy-aws:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to AWS
        run: echo "AWS deployment"
        # ... AWS steps

  deploy-gcp:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to GCP
        run: echo "GCP deployment"
        # ... GCP steps
```

---

## Migration Checklist

### To Add GCP Support

- [ ] Create `.github/workflows/deploy-staging-gcp.yml`
- [ ] Create `infra/gcp/terraform/` directory
- [ ] Create `infra/gcp/terraform/providers.tf` (google provider)
- [ ] Create `infra/gcp/terraform/variables.tf`
- [ ] Create `infra/gcp/terraform/main.tf` (Cloud Run or GCE)
- [ ] Create `infra/gcp/terraform/outputs.tf`
- [ ] Create `infra/gcp/terraform/README.md`
- [ ] Set up GCP project
- [ ] Configure GitHub secrets for GCP
- [ ] Test deployment
- [ ] Update documentation

### Can Keep (No Changes)

- [x] `.github/workflows/cd.yml` (build/push)
- [x] `.github/workflows/linting_formatting.yml`
- [x] `.github/workflows/unit-tests.yml`
- [x] `Dockerfile`
- [x] `docker-compose.yml`
- [x] Application code in `src/`
- [x] Tests

---

## Summary

### What You Need to Create

**New Files (~6 files):**
1. `.github/workflows/deploy-staging-gcp.yml`
2. `infra/gcp/terraform/providers.tf`
3. `infra/gcp/terraform/variables.tf`
4. `infra/gcp/terraform/main.tf`
5. `infra/gcp/terraform/outputs.tf`
6. `infra/gcp/terraform/README.md`

**Effort: 6-8 hours for first-time, 4-5 hours if experienced**

### What Stays the Same

**Unchanged (~90% of codebase):**
- All application code
- All tests
- Dockerfile
- docker-compose
- CI workflows (lint, test, build)
- AWS infrastructure (if keeping)

---

## Bottom Line

**The good news:** Your application code doesn't change at all. Only the infrastructure code changes.

**The effort:** Creating 6 new files, ~6-8 hours of work.

**The benefit:** You can deploy to both clouds, choose based on cost/situation, and learn both platforms.

**Recommendation:** Create the GCP infrastructure alongside AWS. Keep both. Use GCP for learning (free), AWS for resume (industry standard).

**Want me to create the actual GCP Terraform files and GitHub workflow for you?**