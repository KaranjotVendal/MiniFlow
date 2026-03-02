# GCP Setup Explained: Line by Line

A detailed breakdown of every command and configuration we used to set up GCP for MiniFlow deployment.

---

## Table of Contents

1. [What We Built: Big Picture](#what-we-built-big-picture)
2. [The Setup Script Explained](#the-setup-script-explained)
3. [Workload Identity Federation Deep Dive](#workload-identity-federation-deep-dive)
4. [GitHub Actions Workflow Explained](#github-actions-workflow-explained)
5. [Terraform Configuration Explained](#terraform-configuration-explained)
6. [Security Model](#security-model)
7. [Common Issues and Solutions](#common-issues-and-solutions)

---

## What We Built: Big Picture

### Architecture Overview

```
Your Laptop (You)
    │
    │ gcloud commands
    ▼
GCP Project: miniflow-489011
│
├── APIs Enabled
│   ├── Cloud Run (for deployment)
│   ├── Cloud Logging (for logs)
│   └── IAM Credentials (for authentication)
│
├── Service Account: github-actions@...
│   ├── Permissions: Can deploy to Cloud Run
│   └── Permissions: Can act as itself
│
├── Workload Identity Pool: github-pool
│   └── Provider: github-provider
│       ├── Trusts: GitHub Actions (token.actions.githubusercontent.com)
│       ├── Allows: Only KaranjotVendal/MiniFlow repo
│       └── Issues: Short-lived tokens
│
└── (Ready for) Cloud Run Service: miniflow
    └── Will run: Your container image
```

### Authentication Flow

```
GitHub Actions Workflow Runs
        │
        │ 1. Requests OIDC token from GitHub
        ▼
GitHub Issues Token (valid for 5 minutes)
        │
        │ 2. Presents token to GCP
        ▼
GCP Workload Identity Provider
        │
        │ 3. Verifies:
        │    - Token from GitHub? ✓
        │    - Correct repository? ✓
        │    - Not expired? ✓
        ▼
GCP Issues Access Token (valid for 1 hour)
        │
        │ 4. GitHub Actions uses token
        ▼
Deploy to Cloud Run ✓
```

---

## The Setup Script Explained

### Full Script with Line-by-Line Comments

```bash
#!/bin/bash

# These are variables - they make the script reusable
# If you change projects, just change these values
PROJECT_ID="miniflow-489011"                    # Your GCP project
REPO="KaranjotVendal/MiniFlow"                  # Your GitHub repo

echo "=== Setting up GCP for MiniFlow ==="
```

---

### Step 1: Enable APIs

```bash
# Enable APIs...
gcloud services enable run.googleapis.com --project=$PROJECT_ID
gcloud services enable logging.googleapis.com --project=$PROJECT_ID
gcloud services enable iamcredentials.googleapis.com --project=$PROJECT_ID
```

**What is this?**
- GCP services are disabled by default (security)
- You must explicitly enable what you need

**Line by line:**

```bash
gcloud services enable run.googleapis.com --project=$PROJECT_ID
```
- `gcloud services enable`: Command to turn on a service
- `run.googleapis.com`: The Cloud Run API
- `--project=$PROJECT_ID`: Do this in your specific project

**Why needed?**
- Without Cloud Run API enabled, you cannot deploy to Cloud Run
- The error would be: "Cloud Run API has not been used in project 123456789 before or it is disabled"

```bash
gcloud services enable logging.googleapis.com --project=$PROJECT_ID
```
- Enables Cloud Logging API
- Allows your app to write logs
- Without this: No logs visible in console

```bash
gcloud services enable iamcredentials.googleapis.com --project=$PROJECT_ID
```
- Enables IAM Credentials API
- Required for Workload Identity Federation
- Without this: GitHub Actions cannot authenticate

---

### Step 2: Create Service Account

```bash
# Create service account...
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions" \
  --project=$PROJECT_ID

# The email is automatically generated:
# github-actions@miniflow-489011.iam.gserviceaccount.com
```

**What is a Service Account?**

Think of it like a robot user:

```
Regular User (You)
├── Email: karanjotgharu60@gmail.com
├── Logs in with password
└── Can do things in GCP

Service Account (Robot)
├── Email: github-actions@project.iam.gserviceaccount.com
├── Logs in with keys/tokens
└── Can do things in GCP (automated)
```

**Why not use your personal account?**
- Security: If GitHub is compromised, only service account is at risk
- Principle of least privilege: Service account has only needed permissions
- Audit trail: Clear separation of human vs automated actions

**The command explained:**
```bash
gcloud iam service-accounts create github-actions
```
- `gcloud iam service-accounts create`: Creates a new service account
- `github-actions`: The ID (name) of the service account

```bash
--display-name="GitHub Actions"
```
- Human-readable name shown in console
- Optional but recommended

```bash
--project=$PROJECT_ID
```
- Creates it in your specific project

**The error you saw:**
```
ERROR: Service account github-actions already exists
```

This is fine! It means you already ran this step. The script continues with `|| echo "Service account may already exist"`

---

### Step 3: Grant Permissions

```bash
# Grant permissions...
SERVICE_ACCOUNT="github-actions@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/iam.serviceAccountUser"
```

**What are IAM Roles?**

IAM = Identity and Access Management

Think of roles like job descriptions:

```
Role: roles/run.admin (Cloud Run Admin)
├── Can deploy services
├── Can update services
├── Can delete services
└── Can view logs

Role: roles/viewer (Project Viewer)
├── Can view resources
└── Cannot modify anything

Role: roles/owner (Project Owner)
├── Can do everything
└── Includes billing
```

**Line by line:**

```bash
gcloud projects add-iam-policy-binding $PROJECT_ID
```
- `gcloud projects add-iam-policy-binding`: Adds a permission to the project
- `$PROJECT_ID`: Which project to modify

```bash
--member="serviceAccount:$SERVICE_ACCOUNT"
```
- Who gets the permission
- `serviceAccount:` indicates it's a service account (not a person)
- `$SERVICE_ACCOUNT` is the email we defined earlier

```bash
--role="roles/run.admin"
```
- What permission they get
- `roles/run.admin` = Can do everything with Cloud Run

**Why `roles/run.admin`?**
- Deploy new services
- Update existing services
- Configure service settings
- View service details

**Why `roles/iam.serviceAccountUser`?**
- Required for Workload Identity Federation
- Allows the service account to "act as itself"
- Without this: "Error: Permission denied on resource project"

---

### Step 4: Workload Identity Pool

```bash
# Create Workload Identity Pool...
gcloud iam workload-identity-pools create github-pool \
  --location="global" \
  --display-name="GitHub Actions Pool" \
  --project=$PROJECT_ID
```

**What is Workload Identity Federation?**

Before this existed, you had to use **Service Account Keys** (JSON files):

```
OLD WAY (Bad for security):
GitHub Actions
    │
    │ Uses: service-account-key.json
    │       (long-lived credentials)
    ▼
Can impersonate service account
    │
    ▼
Deploy to GCP

PROBLEM: If JSON key leaks, attacker has permanent access!
```

```
NEW WAY (Workload Identity Federation):
GitHub Actions
    │
    │ Requests: OIDC token from GitHub
    │            (short-lived, 5 minutes)
    ▼
Exchanges token at GCP
    │
    │ GCP verifies: Is this really GitHub?
    ▼
Gets temporary access token
    │       (valid for 1 hour)
    ▼
Deploy to GCP

BENEFIT: No long-lived credentials! Much more secure.
```

**The command explained:**

```bash
gcloud iam workload-identity-pools create github-pool
```
- Creates a "pool" that trusts external identity providers
- `github-pool`: The name we chose

```bash
--location="global"
```
- Workload Identity Pools are global resources
- Not tied to a specific region

**The error you saw:**
```
ERROR: ALREADY_EXISTS: Requested entity already exists
```

Fine! Pool was already created. Continuing...

---

### Step 5: Workload Identity Provider

```bash
# Create Provider...
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Actions Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='${REPO}'" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --project=$PROJECT_ID
```

**This is the most complex part!** Let's break it down:

**What is an OIDC Provider?**

```
Think of it like a bouncer at a club:

Club: GCP Project
Bouncer: Workload Identity Provider
ID Check: OIDC Token from GitHub

Bouncer checks:
1. Is this ID from a trusted issuer? (GitHub)
2. Is this person on the guest list? (Your repo)
3. Is the ID still valid? (Not expired)

If yes → Issue temporary access pass
```

**Line by line:**

```bash
gcloud iam workload-identity-pools providers create-oidc github-provider
```
- Creates an OIDC (OpenID Connect) provider
- OIDC is a standard for identity tokens
- `github-provider`: Name we chose

```bash
--workload-identity-pool="github-pool"
```
- Puts this provider in the pool we created earlier

```bash
--issuer-uri="https://token.actions.githubusercontent.com"
```
- **CRITICAL**: Who issues the tokens?
- This is GitHub's OIDC endpoint
- GCP will verify tokens came from this issuer

```bash
--attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository"
```
- Maps GitHub's token claims to GCP's format
- `assertion.sub` = The subject (GitHub Actions)
- `assertion.repository` = The repository name
- This lets us filter by repository later

```bash
--attribute-condition="assertion.repository=='${REPO}'"
```
- **SECURITY CRITICAL**: Only allow your specific repository!
- Without this: ANY GitHub repo could authenticate to your GCP project
- With this: Only `KaranjotVendal/MiniFlow` is allowed

**The error you saw initially:**
```
INVALID_ARGUMENT: The attribute condition must reference one of the provider's claims
```

This was because of a quoting issue. The condition needs exact syntax:
```bash
# WRONG:
--attribute-condition="assertion.repository="KaranjotVendal/MiniFlow"

# CORRECT:
--attribute-condition="assertion.repository=='KaranjotVendal/MiniFlow'"
```

Note: Double equals `==` and single quotes around the repo name!

---

### Step 6: Service Account IAM Binding

```bash
# Allow GitHub to impersonate service account
gcloud iam service-accounts add-iam-policy-binding \
  $SERVICE_ACCOUNT \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/${REPO}" \
  --project=$PROJECT_ID
```

**What does this do?**

This is the crucial link! It says:
> "When Workload Identity Pool verifies a token from GitHub repo `KaranjotVendal/MiniFlow`, allow it to use service account `github-actions@...`"

**The member string explained:**

```
principalSet://iam.googleapis.com/projects/448858650085/locations/global/workloadIdentityPools/github-pool/attribute.repository/KaranjotVendal/MiniFlow
│           │                                               │                           │              │                    │
│           │                                               │                           │              │                    └── Your specific repo
│           │                                               │                           │              └── Filter by repository attribute
│           │                                               │                           └── The pool name
│           │                                               └── Resource location
│           └── Resource type (set of principals)
└── Protocol (Workload Identity Federation)
```

**Why so complex?**
- GCP needs to know: WHO is allowed to impersonate the service account
- This string uniquely identifies: "Any authenticated request from GitHub repo X in pool Y"

**The error you saw:**
```
argument --member: Must be specified.
```

This was a line continuation issue in the script. The string got cut off. When you run commands manually or use the fixed script, it works.

---

## Workload Identity Federation Deep Dive

### Why Is This Better Than Service Account Keys?

| Aspect | Service Account Key | Workload Identity Federation |
|--------|---------------------|------------------------------|
| **Credential type** | JSON file (long-lived) | OIDC token (5 min) |
| **Rotation** | Manual | Automatic |
| **If leaked** | Attacker has permanent access | Token expires quickly |
| **Storage** | In GitHub Secrets (risky) | No secrets stored |
| **Setup complexity** | Simple | More complex |
| **Security** | ⚠️ Less secure | ✅ More secure |

### The Token Exchange Process

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   GitHub Actions │     │  GitHub OIDC    │     │   GCP Project   │
│    (Your Repo)   │────▶│    Issuer       │────▶│  (Your Cloud)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                                               │
        │ 1. Request OIDC token                        │
        │    (GitHub Action: `id-token: write`)       │
        ▼                                               │
┌─────────────────┐                                    │
│   OIDC Token    │                                    │
│   {             │                                    │
│     "sub": "repo:KaranjotVendal/MiniFlow:ref:refs/heads/main", │
│     "aud": "https://iam.googleapis.com/projects/...",          │
│     "exp": 1234567890,                                           │
│     ...                                                           │
│   }             │                                    │
└─────────────────┘                                    │
        │                                               │
        │ 2. Present token to GCP                      │
        │    (GitHub Action: `google-github-actions/auth`)         │
        ▼                                               │
        │──────────────────────────────────────────────▶│
        │              3. GCP validates:                │
        │                 - Signature valid?            │
        │                 - Issuer trusted?             │
        │                 - Repo matches condition?     │
        │                 - Not expired?                │
        │                                               │
        │◀──────────────────────────────────────────────│
        │              4. Return GCP access token       │
        │                 (valid for 1 hour)            │
        │                                               │
        │ 5. Use token to deploy                        │
        │    (Terraform, gcloud, etc.)                  │
        ▼                                               ▼
   ┌─────────────────┐                          ┌─────────────────┐
   │ Deploy to       │                          │ Cloud Run       │
   │ Cloud Run ✓     │                          │ Service Running │
   └─────────────────┘                          └─────────────────┘
```

---

## GitHub Actions Workflow Explained

### Full Workflow with Comments

```yaml
# .github/workflows/deploy-staging-gcp.yml

name: Deploy Staging (GCP)
# The name shown in GitHub Actions tab

on:
  workflow_dispatch:
    # Only run when manually triggered (not on every push)
    inputs:
      image_tag:
        description: "Image tag to deploy (e.g., sha-abc1234)"
        required: true
        default: "main"
        # User must provide which image to deploy

env:
  GCP_PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  # Reads from GitHub Secrets (secure, not hardcoded)
  
  GCP_REGION: us-central1
  # Region where we'll deploy
  
  TF_DIR: infra/gcp/terraform
  # Directory containing Terraform code

permissions:
  contents: read
  # Can read repository contents (to checkout code)
  
  id-token: write
  # CRITICAL: Can request OIDC token from GitHub!
  # Without this, Workload Identity Federation won't work

jobs:
  deploy:
    runs-on: ubuntu-latest
    # Use GitHub-hosted runner (fresh VM each time)

    steps:
      # Step 1: Get the code
      - name: Checkout
        uses: actions/checkout@v4

      # Step 2: Authenticate to GCP
      - name: Authenticate to GCP
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
          # The provider resource name we set up
          # Format: projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/POOL/providers/PROVIDER
          
          service_account: ${{ secrets.GCP_SERVICE_ACCOUNT }}
          # The service account to impersonate
          # Format: github-actions@PROJECT_ID.iam.gserviceaccount.com

      # Step 3: Install Terraform
      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: "1.6.0"

      # Step 4: Initialize Terraform
      - name: Terraform Init
        working-directory: ${{ env.TF_DIR }}
        # Run in the Terraform directory
        run: terraform init
        # Downloads Google provider plugin

      # Step 5: Validate Terraform syntax
      - name: Terraform Validate
        working-directory: ${{ env.TF_DIR }}
        run: terraform validate
        # Checks for syntax errors

      # Step 6: Preview changes
      - name: Terraform Plan
        working-directory: ${{ env.TF_DIR }}
        run: |
          terraform plan \
            -var "project_id=${{ env.GCP_PROJECT_ID }}" \
            -var "region=${{ env.GCP_REGION }}" \
            -var "container_image=ghcr.io/${{ github.repository_owner }}/miniflow:${{ github.event.inputs.image_tag }}"
        # Shows what will be created/changed/destroyed
        # Does NOT make actual changes

      # Step 7: Apply changes
      - name: Terraform Apply
        working-directory: ${{ env.TF_DIR }}
        run: |
          terraform apply -auto-approve \
            -var "project_id=${{ env.GCP_PROJECT_ID }}" \
            -var "region=${{ env.GCP_REGION }}" \
            -var "container_image=ghcr.io/${{ github.repository_owner }}/miniflow:${{ github.event.inputs.image_tag }}"
        # -auto-approve: Don't ask for confirmation
        # This actually creates/updates resources

      # Step 8: Get the service URL
      - name: Get Service URL
        id: get-url
        working-directory: ${{ env.TF_DIR }}
        run: |
          SERVICE_URL=$(terraform output -raw service_url)
          echo "url=${SERVICE_URL}" >> $GITHUB_OUTPUT
          echo "Service URL: ${SERVICE_URL}"
        # Saves URL for later steps

      # Step 9: Verify deployment works
      - name: Health Checks
        run: |
          echo "Waiting for service to be ready..."
          sleep 30
          
          echo "Checking /health endpoint..."
          curl --fail --retry 5 --retry-delay 10 \
            "${{ steps.get-url.outputs.url }}/health"
          
          echo "Checking /ready endpoint..."
          curl --fail --retry 5 --retry-delay 10 \
            "${{ steps.get-url.outputs.url }}/ready"
          
          echo "All health checks passed!"
        # --fail: Exit with error if HTTP status >= 400
        # --retry 5: Try 5 times before giving up

      # Step 10: Show summary
      - name: Output Deployment Info
        run: |
          echo "## Deployment Summary" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "- **Service URL**: ${{ steps.get-url.outputs.url }}" >> $GITHUB_STEP_SUMMARY
          echo "- **Image**: ghcr.io/${{ github.repository_owner }}/miniflow:${{ github.event.inputs.image_tag }}" >> $GITHUB_STEP_SUMMARY
          echo "- **Region**: ${{ env.GCP_REGION }}" >> $GITHUB_STEP_SUMMARY
        # Shows nice summary in GitHub Actions UI
```

---

## Terraform Configuration Explained

### Provider Configuration

```hcl
# providers.tf

terraform {
  required_version = ">= 1.6.0"
  # Must use Terraform 1.6.0 or newer

  required_providers {
    google = {
      source  = "hashicorp/google"
      # Download from HashiCorp's registry
      
      version = "~> 5.0"
      # Use version 5.x (latest minor version)
      # ~> means: 5.0 or higher, but not 6.0
    }
  }
}

provider "google" {
  project = var.project_id
  # Use the project ID passed as variable
  
  region  = var.region
  # Default region for resources
}
```

### Main Resources

```hcl
# Enable required APIs
resource "google_project_service" "run" {
  service = "run.googleapis.com"
  # The API identifier
  
  disable_on_destroy = false
  # Keep API enabled even if we delete this resource
}
```

**Why `disable_on_destroy = false`?**
- If you run `terraform destroy`, this controls whether to disable the API
- Setting to `false` prevents accidentally breaking other services
- Usually want to keep APIs enabled

```hcl
# Cloud Run service
resource "google_cloud_run_service" "miniflow" {
  name     = local.service_name
  # Name shown in GCP Console
  
  location = var.region
  # Where to deploy (us-central1, etc.)

  template {
    spec {
      containers {
        image = var.container_image
        # Docker image to run
        
        resources {
          limits = {
            cpu    = var.cpu
            # CPU cores ("1", "2", "4", etc.)
            
            memory = var.memory
            # Memory ("512Mi", "1Gi", "4Gi", etc.)
          }
        }

        ports {
          container_port = 8000
          # Port your app listens on
        }

        env {
          # Environment variables passed to container
          name  = "MINIFLOW_DEVICE"
          value = var.miniflow_device
        }
        # ... more env vars
      }

      container_concurrency = 10
      # How many requests ONE container handles at once
      # 10 means: if 11 requests come, Cloud Run starts a 2nd container

      timeout_seconds = 900
      # Max 15 minutes per request
      # After this, request is killed
    }

    metadata {
      labels = local.common_labels
      # Tags for organization
      
      annotations = {
        "autoscaling.knative.dev/minScale" = "0"
        # Scale to 0 when no traffic (saves money)
        
        "autoscaling.knative.dev/maxScale" = "5"
        # Never more than 5 containers
        # Protects against runaway costs
      }
    }
  }

  traffic {
    percent         = 100
    # Send 100% of traffic to this revision
    
    latest_revision = true
    # Use the latest deployment
  }
}
```

### IAM Configuration

```hcl
# Allow public access
resource "google_cloud_run_service_iam_member" "public" {
  service  = google_cloud_run_service.miniflow.name
  location = google_cloud_run_service.miniflow.location
  role     = "roles/run.invoker"
  # Built-in role: can invoke (call) the service
  
  member   = "allUsers"
  # Special value: anyone on the internet
  # Without this: Service is private, requires authentication
}
```

---

## Security Model

### Defense in Depth

```
Layer 1: Workload Identity Federation
├── Only GitHub repo X can authenticate
├── Tokens expire in 5 minutes
└── No long-lived credentials

Layer 2: Service Account Permissions
├── Can only deploy to Cloud Run
├── Cannot access other services
├── Cannot modify IAM

Layer 3: Cloud Run IAM
├── Service is public (allUsers)
├── But only for invoking (not admin)

Layer 4: Auto-scaling Limits
├── Max 5 containers
├── Prevents cost runaway

Layer 5: Request Timeouts
├── 15 minute max per request
├── Prevents hanging processes
```

### What If...

**GitHub repo is compromised?**
- Attacker can deploy to your Cloud Run
- Cannot access other GCP services
- Cannot escalate permissions
- You can revoke access immediately

**GCP service account is leaked?**
- No JSON keys exist (Workload Identity)
- Nothing to leak!
- Even if token stolen, expires in 1 hour

**Cloud Run service is attacked?**
- Container is isolated
- Max 5 instances (limits blast radius)
- No persistent data (stateless)
- Logs available for forensics

---

## Common Issues and Solutions

### Issue 1: "Cloud Run API has not been used"

**Error:**
```
Error: Error creating Service: googleapi: Error 403: Cloud Run API has not been used in project 123456789 before or it is disabled.
```

**Cause:** API not enabled

**Solution:**
```bash
gcloud services enable run.googleapis.com --project=miniflow-489011
```

---

### Issue 2: "Permission denied on resource project"

**Error:**
```
Error 403: Permission denied on resource project
```

**Cause:** Service account lacks permissions

**Solution:**
```bash
gcloud projects add-iam-policy-binding miniflow-489011 \
  --member="serviceAccount:github-actions@miniflow-489011.iam.gserviceaccount.com" \
  --role="roles/run.admin"
```

---

### Issue 3: "Failed to generate credentials"

**Error:**
```
Error: Failed to generate credentials for Workload Identity Federation
```

**Cause:** Workload Identity Federation not configured correctly

**Solution:** Check:
1. Workload Identity Pool exists
2. Provider exists with correct issuer
3. Attribute condition matches your repo
4. Service account IAM binding is correct

---

### Issue 4: "Image not found"

**Error:**
```
Error creating Service: googleapi: Error 400: spec.template.spec.containers[0].image
```

**Cause:** Container image doesn't exist or is private

**Solution:**
```bash
# Verify image exists
docker pull ghcr.io/karanjotvendal/miniflow:main

# If private, ensure GCP can access it
# OR use Artifact Registry instead
```

---

## Summary: What We Built

### Infrastructure
- ✅ GCP Project with billing
- ✅ Required APIs enabled
- ✅ Service account with minimal permissions
- ✅ Workload Identity Federation for secure auth
- ✅ Terraform configuration for Cloud Run

### CI/CD
- ✅ GitHub Actions workflow
- ✅ Automated deployment on manual trigger
- ✅ Health checks after deployment
- ✅ No secrets stored in code

### Security
- ✅ No long-lived credentials
- ✅ Short-lived OIDC tokens
- ✅ Repository-scoped access
- ✅ Auto-scaling limits
- ✅ Public service but controlled access

### Next Steps
1. Trigger deployment with: `gh workflow run deploy-staging-gcp.yml`
2. Verify with: `curl $(terraform output -raw service_url)/health`
3. Iterate and improve!

---

Ready to deploy? Run the GitHub Actions workflow and watch it work!
