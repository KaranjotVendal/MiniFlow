# GCP Setup Guide for MiniFlow

Step-by-step guide to configure Google Cloud Platform for MiniFlow deployment.

---

## Table of Contents

1. [Create GCP Project](#1-create-gcp-project)
2. [Enable Billing](#2-enable-billing)
3. [Enable APIs](#3-enable-apis)
4. [Configure Authentication](#4-configure-authentication)
5. [Set Up GitHub Secrets](#5-set-up-github-secrets)
6. [Test Deployment](#6-test-deployment)

---

## 1. Create GCP Project

### Step 1: Go to GCP Console

1. Visit: https://console.cloud.google.com/
2. Sign in with your Google account
3. Click "Select a project" → "New Project"

### Step 2: Create New Project

```
Project name: miniflow-staging
Location: No organization
Project ID: (auto-generated, or choose custom)
```

**Note the Project ID** - you'll need it later (e.g., `miniflow-staging-123456`)

---

## 2. Enable Billing

**Required**: Even with $300 credit, you must enable billing.

### Step 1: Go to Billing

1. In GCP Console, click the hamburger menu (☰)
2. Go to "Billing"
3. Click "Link a billing account"

### Step 2: Set Up Billing

1. Choose "Create billing account"
2. Enter your information
3. Add payment method (credit card required, but won't be charged until credit expires)
4. **You'll get $300 free credit**

### Step 3: Link to Project

1. Go to "Billing" → "My projects"
2. Find your `miniflow-staging` project
3. Link it to your billing account

---

## 3. Enable APIs

### Required APIs

Enable these APIs for your project:

1. **Cloud Run API** - For deploying containers
2. **Cloud Build API** - For building containers (optional)
3. **Cloud Logging API** - For logs
4. **Cloud Monitoring API** - For metrics

### Enable via Console

1. Go to: https://console.cloud.google.com/apis/library
2. Search for each API and click "Enable"

### Enable via gcloud CLI

```bash
# Install gcloud: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable APIs
gcloud services enable run.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable monitoring.googleapis.com
```

---

## 4. Configure Authentication

### Option A: Local Development (gcloud CLI)

For running Terraform locally:

```bash
# Authenticate
gcloud auth application-default login

# This creates credentials at:
# ~/.config/gcloud/application_default_credentials.json
```

### Option B: GitHub Actions (Workload Identity Federation)

For CI/CD deployment (recommended):

#### Step 1: Create Service Account

```bash
# Create service account for GitHub Actions
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions"

# Get service account email
export SERVICE_ACCOUNT="github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com"
```

#### Step 2: Grant Permissions

```bash
# Grant Cloud Run Admin (to deploy services)
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/run.admin"

# Grant IAM Service Account User (to act as the service account)
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/iam.serviceAccountUser"

# Grant Service Account Token Creator (for OIDC)
gcloud iam service-accounts add-iam-policy-binding $SERVICE_ACCOUNT \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/YOUR_PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/YOUR_GITHUB_USERNAME/MiniFlow"
```

**Replace:**
- `YOUR_PROJECT_ID` - Your GCP project ID
- `YOUR_PROJECT_NUMBER` - Your GCP project number (found in Console)
- `YOUR_GITHUB_USERNAME` - Your GitHub username

#### Step 3: Create Workload Identity Pool

```bash
# Create Workload Identity Pool
gcloud iam workload-identity-pools create github-pool \
  --location="global" \
  --display-name="GitHub Actions Pool"

# Create Workload Identity Provider
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Actions Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='YOUR_GITHUB_USERNAME/MiniFlow'" \
  --issuer-uri="https://token.actions.githubusercontent.com"
```

#### Step 4: Get Workload Identity Provider Resource Name

```bash
# Get the provider resource name
gcloud iam workload-identity-pools providers describe github-provider \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --format="value(name)"

# Output looks like:
# projects/123456789/locations/global/workloadIdentityPools/github-pool/providers/github-provider
```

**Save this** - you'll need it for GitHub secrets.

---

## 5. Set Up GitHub Secrets

### Step 1: Go to GitHub Repository

1. Go to: https://github.com/YOUR_USERNAME/MiniFlow
2. Click "Settings" → "Secrets and variables" → "Actions"
3. Click "New repository secret"

### Step 2: Add Secrets

Add these three secrets:

| Secret Name | Value | How to Get |
|-------------|-------|------------|
| `GCP_PROJECT_ID` | Your project ID | Console → Project selector |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Full provider path | From Step 4 above |
| `GCP_SERVICE_ACCOUNT` | Service account email | `github-actions@PROJECT_ID.iam.gserviceaccount.com` |

**Example values:**
```
GCP_PROJECT_ID=miniflow-staging-123456
GCP_WORKLOAD_IDENTITY_PROVIDER=projects/123456789/locations/global/workloadIdentityPools/github-pool/providers/github-provider
GCP_SERVICE_ACCOUNT=github-actions@miniflow-staging-123456.iam.gserviceaccount.com
```

---

## 6. Test Deployment

### Step 1: Verify GitHub Actions

1. Go to: https://github.com/YOUR_USERNAME/MiniFlow/actions
2. You should see "Deploy Staging (GCP)" workflow
3. Click on it

### Step 2: Trigger Deployment

```bash
# Via GitHub UI:
# 1. Click "Run workflow"
# 2. Enter image tag (e.g., "main" or "sha-abc1234")
# 3. Click "Run workflow"

# Via CLI:
gh workflow run deploy-staging-gcp.yml -f image_tag=main
```

### Step 3: Monitor Deployment

1. Watch the workflow run in GitHub Actions
2. Look for "Terraform Apply" step
3. Check for "Health Checks" passing

### Step 4: Verify Deployment

```bash
# Get the service URL from workflow output
# Or run locally:
cd infra/gcp/terraform
export SERVICE_URL=$(terraform output -raw service_url)

# Test endpoints
curl $SERVICE_URL/health
curl $SERVICE_URL/ready
```

---

## Quick Reference

### Useful gcloud Commands

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# List services
gcloud run services list

# View logs
gcloud logging tail "run.googleapis.com%2Fminiflow"

# Describe service
gcloud run services describe miniflow --region=us-central1

# Delete service
gcloud run services delete miniflow --region=us-central1
```

### Useful Terraform Commands

```bash
# Initialize
cd infra/gcp/terraform
terraform init

# Plan
terraform plan -var "project_id=YOUR_PROJECT_ID" -var "container_image=IMAGE_URL"

# Apply
terraform apply -var "project_id=YOUR_PROJECT_ID" -var "container_image=IMAGE_URL"

# Destroy (stop billing)
terraform destroy

# Output
terraform output service_url
```

---

## Troubleshooting

### "API not enabled" Error

```bash
# Enable the API
gcloud services enable SERVICE_NAME.googleapis.com
```

### "Permission denied" Error

1. Check service account has correct roles
2. Verify Workload Identity Federation is configured
3. Ensure GitHub secrets are correct

### "Image not found" Error

```bash
# Verify image exists
docker pull ghcr.io/karanjotvendal/miniflow:TAG

# If using GCR instead:
# gcr.io/PROJECT_ID/miniflow:TAG
```

### "Out of quota" Error

New GCP projects have default quotas. If you hit limits:

1. Go to: IAM & Admin → Quotas
2. Request increase for specific quota
3. Or check if you're using too many resources

---

## Next Steps

1. ✅ GCP project created
2. ✅ Billing enabled ($300 credit)
3. ✅ APIs enabled
4. ✅ Authentication configured
5. ✅ GitHub secrets added
6. ✅ Test deployment successful

**Now you can:**
- Deploy MiniFlow anytime with GitHub Actions
- Scale to zero when not using (cost: ~$0)
- Monitor via Cloud Logging
- Iterate on your ML deployment!

---

## Resources

- [GCP Free Tier](https://cloud.google.com/free)
- [Cloud Run Quickstart](https://cloud.google.com/run/docs/quickstarts)
- [Workload Identity Federation](https://cloud.google.com/iam/docs/workload-identity-federation)
- [GitHub Actions for GCP](https://github.com/google-github-actions/setup-gcloud)
