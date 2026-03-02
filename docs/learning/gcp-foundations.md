# GCP Foundations for ML Deployment

A beginner's guide to understanding Google Cloud Platform for deploying machine learning models.

---

## Table of Contents

1. [Why Google Cloud?](#why-google-cloud)
2. [GCP Account Setup](#gcp-account-setup)
3. [Core GCP Concepts](#core-gcp-concepts)
4. [GCP Services for ML Deployment](#gcp-services-for-ml-deployment)
5. [GCP vs AWS: Key Differences](#gcp-vs-aws-key-differences)
6. [GCP Console Walkthrough](#gcp-console-walkthrough)
7. [Cost Management](#cost-management)
8. [Security Basics](#security-basics)
9. [Next Steps](#next-steps)

---

## Why Google Cloud?

### Google's DNA: Data and AI

Google built GCP on their expertise running:
- **Google Search**: Billions of queries daily
- **YouTube**: 2+ billion users, ML for recommendations
- **Gmail**: Spam detection, smart reply
- **Google Photos**: Image recognition at scale

**Result**: GCP is designed for data-intensive and ML workloads from the ground up.

---

### GCP vs AWS: Philosophy

| Aspect | AWS | GCP |
|--------|-----|-----|
| **Philosophy** | "Everything is a service" | "Batteries included" |
| **Approach** | Granular, maximum control | Opinionated, faster to production |
| **Learning curve** | Steeper (more services to learn) | Gentler (sensible defaults) |
| **ML/AI focus** | Broad platform | Deep AI/ML integration |
| **Pricing** | Complex, fine-grained | Simpler, sustained use discounts |

**Choose GCP when**:
- ✅ You want to get to production faster
- ✅ You're doing heavy ML/AI work
- ✅ You prefer simplicity over maximum control
- ✅ You want Kubernetes (GKE is best-in-class)
- ✅ You're cost-conscious (sustained use discounts)

---

## GCP Account Setup

### Step 1: Create Google Account

You probably already have one (Gmail). If not:
1. Go to https://accounts.google.com/signup
2. Create account

### Step 2: Create GCP Project

1. Go to https://console.cloud.google.com/
2. Click "Select a project" → "New Project"
3. Enter:
   - **Project name**: `miniflow-learning`
   - **Organization**: None (or your organization)
4. Click "Create"

**Important**: Note your **Project ID** (e.g., `miniflow-learning-123456`)

### Step 3: Enable Billing

**Required even for free tier!**

1. Go to "Billing" in the left menu
2. Click "Link a billing account"
3. Create billing account with your credit card
4. **You'll receive $300 free credit for 90 days**

**Cost protection**:
- Free trial won't auto-charge when credit expires
- You must manually upgrade to paid account
- Set budget alerts (we'll cover this)

### Step 4: Install gcloud CLI

```bash
# macOS (with Homebrew)
brew install --cask google-cloud-sdk

# Ubuntu/Debian
curl https://sdk.cloud.google.com | bash

# Windows
# Download installer: https://cloud.google.com/sdk/docs/install

# Initialize
gcloud init

# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Verify
gcloud info
```

---

## Core GCP Concepts

### 1. Projects

**Project**: The top-level organizing unit in GCP

```
Your Google Account
├── Project: miniflow-learning
│   ├── Resources: Compute, Storage, etc.
│   ├── Billing: $300 credit
│   └── APIs: Enabled services
│
├── Project: personal-website
│   ├── Different resources
│   └── Different billing
│
└── Project: experiment-ml
    └── Yet another isolated environment
```

**Key points**:
- Resources belong to exactly one project
- Billing is per-project
- You can have multiple projects
- Projects are completely isolated

**Project ID vs Project Name**:
- **Project ID**: Unique across all of GCP (e.g., `miniflow-123456`)
- **Project Name**: Human-readable (e.g., "MiniFlow Learning")

---

### 2. Regions and Zones

```
Google Global Infrastructure
│
├── Region: us-central1 (Iowa)
│   ├── Zone: us-central1-a
│   ├── Zone: us-central1-b
│   ├── Zone: us-central1-c
│   └── Zone: us-central1-f
│
├── Region: europe-west1 (Belgium)
│   ├── Zone: europe-west1-b
│   ├── Zone: europe-west1-c
│   └── Zone: europe-west1-d
│
└── Region: asia-east1 (Taiwan)
    └── ...
```

**Region**: Geographic area (e.g., `us-central1`)
**Zone**: Isolated location within region (e.g., `us-central1-a`)

**Best practices**:
- Choose region closest to your users
- Spread resources across zones for high availability
- `us-central1` is usually cheapest and has most services

---

### 3. VPC Networks

**VPC (Virtual Private Cloud)**: Your isolated network

```
VPC Network (auto-created by default)
│
├── Subnets (auto-created in each region)
│   ├── Subnet: us-central1 (10.128.0.0/20)
│   ├── Subnet: europe-west1 (10.132.0.0/20)
│   └── Subnet: asia-east1 (10.140.0.0/20)
│
├── Firewall Rules
│   ├── Default: allow internal
│   └── Custom: your rules
│
└── Routes
    └── Default route to internet
```

**GCP vs AWS VPC**:
- GCP: VPC is global (spans all regions automatically)
- AWS: VPC is regional (one per region)
- GCP: Auto-created default network
- AWS: You must create VPC manually

**For MiniFlow**: Use default network (simpler), or create custom for security.

---

### 4. IAM (Identity and Access Management)

```
GCP IAM Hierarchy
│
├── Organization (optional)
│   └── Folder (optional)
│       └── Project
│           ├── Resources
│           └── IAM Policies
│
├── Members
│   ├── User: your-email@gmail.com
│   ├── Service Account: github-actions@project.iam.gserviceaccount.com
│   └── Group: data-science-team@company.com
│
└── Roles
    ├── Primitive
    │   ├── Owner (full control)
    │   ├── Editor (can modify)
    │   └── Viewer (read-only)
    └── Predefined
        ├── roles/run.admin (Cloud Run admin)
        ├── roles/storage.objectViewer
        └── roles/compute.instanceAdmin
```

**Key concept**: Grant roles to members on resources

**Example**:
```bash
# Grant Cloud Run admin to service account
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:github-actions@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"
```

---

## GCP Services for ML Deployment

### Service Comparison

| Service | Purpose | AWS Equivalent | Best For |
|---------|---------|----------------|----------|
| **Cloud Run** | Serverless containers | Fargate (sort of) | APIs, websites, batch jobs |
| **Compute Engine** | Virtual machines | EC2 | Full control, GPU workloads |
| **GKE** | Kubernetes | EKS | Container orchestration at scale |
| **Cloud Functions** | Serverless functions | Lambda | Event-driven, simple functions |
| **Cloud Storage** | Object storage | S3 | Storing files, models, data |
| **Artifact Registry** | Container registry | ECR | Docker images |
| **Vertex AI** | ML platform | SageMaker | End-to-end ML lifecycle |
| **Cloud Monitoring** | Monitoring | CloudWatch | Metrics, dashboards, alerts |
| **Cloud Logging** | Logging | CloudWatch Logs | Centralized logging |

---

### Cloud Run (Serverless Containers)

**What**: Run containers without managing servers

```
Your Container
       │
       ▼
┌─────────────────────┐
│    Cloud Run        │
│  ┌───────────────┐  │
│  │  Container 1  │  │  ← Running
│  │  (handles     │  │
│  │   requests)   │  │
│  └───────────────┘  │
│                     │
│  Auto-scaling:      │
│  • 0 instances when │
│    no traffic       │
│  • Up to N when     │
│    busy             │
│                     │
│  Built-in:          │
│  • HTTPS endpoint   │
│  • Load balancing   │
│  • SSL certificates │
└─────────────────────┘
```

**Why use for MiniFlow**:
- ✅ No server management
- ✅ Scales to zero (saves money)
- ✅ Built-in HTTPS/SSL
- ✅ Pay only for what you use
- ✅ Fast deployment (2-3 minutes)

**Limitations**:
- ⚠️ Max 32 vCPU, 32GB per container
- ⚠️ No GPU support (use Compute Engine for GPU)
- ⚠️ Request timeout: 60 minutes (3600 seconds)

---

### Compute Engine (VMs)

**What**: Virtual machines with full control

```
Compute Engine Instance
├── Machine type: n1-standard-4 (4 vCPU, 15GB)
├── Boot disk: Container-Optimized OS
├── GPU: NVIDIA T4 (optional)
├── Public IP: Auto-assigned
└── Startup script: Pull and run container
```

**Use when**:
- You need GPU (Cloud Run doesn't support GPU)
- You need persistent storage attached
- You need specific OS or kernel
- You want to SSH into the machine

**For MiniFlow GPU**:
```bash
# Create instance with GPU
gcloud compute instances create miniflow-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=cos-stable \
  --image-project=cos-cloud
```

---

### Cloud Storage

**What**: Object storage for files

```
Bucket: miniflow-data
├── models/
│   ├── v1.0/model.pkl
│   └── v2.0/model.pkl
├── datasets/
│   ├── training/
│   └── validation/
└── outputs/
    └── predictions/
```

**Storage classes**:
| Class | Use Case | Cost |
|-------|----------|------|
| **Standard** | Frequently accessed | Highest |
| **Nearline** | Monthly access | Lower |
| **Coldline** | Quarterly access | Low |
| **Archive** | Yearly access | Lowest |

**For MiniFlow**: Use Standard for active models/data.

---

### Artifact Registry

**What**: Store and manage Docker images

```
Artifact Registry
└── Repository: miniflow
    ├── us-docker.pkg.dev/project/miniflow/api:v1.0
    ├── us-docker.pkg.dev/project/miniflow/api:v2.0
    └── us-docker.pkg.dev/project/miniflow/api:latest
```

**Alternative**: Use GitHub Container Registry (GHCR) like we're doing

---

### Vertex AI (Optional)

**What**: Google's managed ML platform

```
Vertex AI
├── Training
│   ├── Custom training
│   └── AutoML
├── Models
│   └── Model registry
├── Endpoints
│   └── Online prediction
└── MLOps
    ├── Pipelines
    └── Feature store
```

**Use if**: You're doing full MLOps (training, versioning, serving)
**Skip if**: You just want to deploy a container (use Cloud Run)

---

## GCP vs AWS: Key Differences

### Networking

| Feature | AWS | GCP |
|---------|-----|-----|
| **VPC scope** | Regional | Global |
| **Default VPC** | Must create | Auto-created |
| **Subnets** | Manual per AZ | Auto-created per region |
| **Firewall** | Security Groups (stateful) | Firewall Rules (stateful) |
| **Load balancing** | ALB, NLB (separate resources) | Built into Cloud Run, or single LB |

### Compute

| Feature | AWS | GCP |
|---------|-----|-----|
| **Serverless containers** | Fargate (complex) | Cloud Run (simple) |
| **VMs** | EC2 | Compute Engine |
| **Kubernetes** | EKS | GKE (arguably better) |
| **GPU** | Many options | Fewer but simpler |
| **Spot instances** | Spot | Preemptible VMs |

### Pricing

| Aspect | AWS | GCP |
|--------|-----|-----|
| **Model** | Complex, service-specific | Simpler |
| **Discounts** | Reserved instances (commitment) | Sustained use (automatic) |
| **Free tier** | 12 months, limited services | $300 credit, some always-free |
| **Billing granularity** | Per second (most services) | Per second |

---

## GCP Console Walkthrough

### Navigation

```
Console (console.cloud.google.com)
│
├── ☰ Hamburger Menu
│   ├── Compute Engine
│   │   ├── VM instances
│   │   └── Instance templates
│   ├── Cloud Run
│   │   └── Services
│   ├── VPC Network
│   │   └── VPC networks
│   ├── IAM & Admin
│   │   └── IAM
│   └── Monitoring
│
├── Project selector (top)
│   └── Switch between projects
│
├── Cloud Shell (top right)
│   └── Terminal in browser
│
└── Notifications & Help
```

### Key Pages for MiniFlow

1. **Cloud Run**: Deploy and manage services
2. **Compute Engine**: For GPU instances (if needed)
3. **Cloud Logging**: View logs from your app
4. **Cloud Monitoring**: Metrics and dashboards
5. **IAM**: Manage permissions

### Cloud Shell

**What**: Browser-based terminal with gcloud CLI pre-installed

```bash
# Activate Cloud Shell (top right of console)
# You get:
# - 5 GB persistent storage
# - gcloud CLI pre-configured
# - Pre-authenticated to your project

# Example: Deploy from Cloud Shell
gcloud run deploy miniflow \
  --image=gcr.io/project/miniflow:latest \
  --region=us-central1
```

**Useful for**: Quick commands without installing gcloud locally

---

## Cost Management

### GCP Free Tier

**$300 Credit** (first 90 days):
- Use for any GCP service
- Valid for 90 days from signup
- Must enable billing (credit card)
- Won't auto-charge when credit expires

**Always Free** (limited, forever):
| Service | Limit |
|---------|-------|
| Compute Engine (f1-micro) | 1 instance per month (US regions) |
| Cloud Storage | 5 GB per month |
| Cloud Run | 2 million requests per month |
| BigQuery | 1 TB query processing per month |

### Cost Estimation for MiniFlow

**Cloud Run (CPU-only)**:
```
Configuration:
- 2 vCPU, 4Gi memory
- Running 24/7 for 3 days

Cost:
- vCPU: 72 hours × $0.00002400/sec = $6.22
- Memory: 72 hours × $0.00000250/sec = $0.65
- Requests: Minimal for testing
- Total: ~$7 for 3 days continuous

With free tier: Most covered, actual ~$2-5
```

**Compute Engine with GPU**:
```
Configuration:
- n1-standard-4 (4 vCPU, 15GB)
- 1× NVIDIA T4 GPU
- Running 24/7 for 3 days

Cost:
- VM: 72 hours × $0.19/hour = $13.68
- GPU: 72 hours × $0.35/hour = $25.20
- Total: ~$39 for 3 days continuous

With $300 credit: FREE
```

### Setting Up Budget Alerts

```bash
# Create budget (in Console)
1. Go to "Billing" → "Budgets & alerts"
2. Click "Create budget"
3. Name: "miniflow-budget"
4. Amount: $50 (or your limit)
5. Alerts: 50%, 90%, 100%
6. Email: your-email@example.com
```

### Cost Optimization Tips

1. **Use Cloud Run for development**: Scales to zero
2. **Destroy resources when done**: `terraform destroy`
3. **Use preemptible VMs for GPU**: 70% cheaper
4. **Monitor billing daily**: Check the Console
5. **Set up quotas**: Prevent accidental overspending

---

## Security Basics

### Service Accounts

**What**: Special Google accounts for applications/services

```
Service Account: github-actions@project.iam.gserviceaccount.com
├── Used by: GitHub Actions
├── Permissions: Cloud Run Admin
└── Key: None (uses Workload Identity Federation)
```

**Best practice**: Use service accounts, not personal credentials

### Workload Identity Federation

**What**: Let external systems (GitHub) access GCP without service account keys

```
GitHub Actions
       │
       │ OIDC token
       ▼
Workload Identity Pool
       │
       │ Authenticated
       ▼
Service Account
       │
       ▼
GCP Resources
```

**Benefits**:
- No long-lived keys to manage
- Automatic credential rotation
- More secure than JSON keys

### Cloud IAM Best Practices

1. **Principle of least privilege**: Grant minimum required permissions
2. **Use service accounts**: Don't use personal accounts for automation
3. **Enable audit logs**: Track who did what
4. **Regular review**: Check IAM policies monthly

---

## Next Steps

1. **Set up GCP project**: Follow setup guide
2. **Deploy MiniFlow**: Use the Terraform configuration
3. **Monitor costs**: Set up budget alerts
4. **Learn more**: Explore Vertex AI if doing full MLOps

---

## Resources

- [GCP Free Tier](https://cloud.google.com/free)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Compute Engine Documentation](https://cloud.google.com/compute/docs)
- [GCP Pricing Calculator](https://cloud.google.com/products/calculator)
- [GCP Best Practices](https://cloud.google.com/docs/enterprise/best-practices-for-enterprise-organizations)

---

## Comparison: AWS vs GCP for MiniFlow

| Aspect | AWS | GCP |
|--------|-----|-----|
| **Free tier** | 12 months, limited | $300 credit, 90 days |
| **Serverless containers** | Fargate (complex) | Cloud Run (simple) |
| **GPU** | More options | Simpler setup |
| **Deployment time** | 20-25 mins | 2-3 mins |
| **Learning curve** | Steeper | Gentler |
| **ML focus** | Broad platform | Deep AI integration |
| **Cost for 3-4 days** | $10-35 | $0 (with credit) |

**For learning**: GCP is simpler and currently free for you

**For production**: Both are excellent, AWS has more market share

---

Ready to deploy? Continue with `gcp-setup-guide.md` for step-by-step instructions!
