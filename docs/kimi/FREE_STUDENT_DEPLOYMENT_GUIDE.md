# Free/Student Deployment Options for MiniFlow

## The Reality Check

**Current plan (AWS ECS/Fargate):**
- Estimated cost: ~$200/month for staging
- **NOT suitable for free tier** - Fargate has no free tier

**Good news:** Multiple paths to zero-cost deployment exist.

---

## Option 1: AWS Free Tier (Modified Architecture) ⭐ RECOMMENDED

### What AWS Free Tier Includes (12 months)

| Resource | Free Tier | MiniFlow Usage |
|----------|-----------|----------------|
| **EC2 t2.micro/t3.micro** | 750 hrs/month | Run container on EC2 instead of Fargate |
| **Application Load Balancer** | None | ❌ Skip ALB, use EC2 public IP directly |
| **EBS (storage)** | 30GB | Sufficient for container + models |
| **Data transfer** | 100GB out | Should be enough for demo |
| **CloudWatch** | Basic metrics | Included |

**Cost: $0/month** (within free tier limits)

### Modified Architecture

```
Student/Free Tier AWS:

┌─────────────────────────────────────────┐
│  EC2 t3.micro (2 vCPU, 1GB RAM)        │  ◄── Free tier
│  ┌─────────────────────────────────┐   │
│  │  Docker Container               │   │
│  │  ├─ FastAPI app                 │   │
│  │  ├─ Prometheus (metrics)        │   │
│  │  └─ Grafana (dashboards)        │   │
│  └─────────────────────────────────┘   │
│                                         │
│  Public IP: http://3.123.45.67:8000    │
│  (No ALB needed)                        │
└─────────────────────────────────────────┘

Terraform: Creates EC2 + Security Group only
```

### Trade-offs

| Aspect | ECS/Fargate (Original) | EC2 (Free Tier) |
|--------|------------------------|-----------------|
| **Cost** | ~$200/month | **$0** ✅ |
| **Scalability** | Auto-scaling | Manual |
| **Availability** | Multi-AZ | Single instance |
| **Resume value** | Higher | Still good |
| **Interview story** | "Production-grade" | "Cost-optimized" |

**Interview angle:**
> "For the demo, I deployed on EC2 free tier to minimize costs while maintaining the same observability and deployment automation. In production, I'd use ECS/Fargate for auto-scaling."

---

## Option 2: Alternative Cloud Providers (Better Free Tiers)

### 2.1 Google Cloud Platform (GCP) - Free Tier ⭐ STRONG

**What's Free:**
- **Cloud Run**: 2M requests/month, 360,000 GB-seconds/month
- **Compute Engine**: 1 f1-micro instance (always free, not just 12 months)
- **Cloud Storage**: 5GB/month
- **$300 credit** for 90 days (student/educator)

**Architecture:**
```
MiniFlow on GCP Cloud Run (Serverless):

┌─────────────────────────────────────────┐
│  Cloud Run (serverless containers)     │  ◄── Free: 2M requests/month
│  ├─ Auto-scales to 0 when idle         │
│  ├─ Pay only when processing           │
│  └─ HTTPS endpoint included            │
└─────────────────────────────────────────┘
```

**Cost: $0/month** (within free tier)

**Pros:**
- Serverless (no server management)
- HTTPS included
- Auto-scaling (impressive in interviews)
- Better free tier than AWS

**Cons:**
- GPU not supported in Cloud Run (CPU only)
- Cold start latency (3-5 seconds)

---

### 2.2 Railway.app - Generous Free Tier ⭐ EASIEST

**What's Free:**
- 500 hours of runtime/month
- 1GB RAM, shared CPU
- 1GB persistent storage
- Custom domains

**Cost: $0/month** (with limits)

**Pros:**
- Easiest deployment (git push → auto-deploy)
- Built-in metrics and logs
- No AWS complexity
- Perfect for demos

**Cons:**
- Less "enterprise" than AWS
- Smaller resume value
- No GPU support

---

### 2.3 Render - Free Tier

**What's Free:**
- Web services: Always on
- 512MB RAM
- Build from GitHub
- Custom domains

**Cost: $0/month**

**Similar to Railway** - easy but less enterprise credibility.

---

## Option 3: GitHub Student Developer Pack

### What's Included (Free for Students)

| Service | Benefit |
|---------|---------|
| **GitHub Pro** | Free while student |
| **AWS Educate** | $100-150 credits |
| **Azure** | $100 credits |
| **DigitalOcean** | $100 credits |
| **Heroku** | Hobby dyno credits |
| **Namecheap** | Free domain |

**Strategy:** Use AWS Educate credits for original ECS/Fargate plan

---

## Recommended Free/Student Strategy

### Path A: AWS Free Tier (Best Interview Value)

```
1. Sign up for AWS Free Tier (12 months)
2. Use EC2 t3.micro instead of Fargate
3. Skip Application Load Balancer (use public IP)
4. Use GitHub Actions (free for public repos)
5. Use GitHub Container Registry (free for public repos)
6. Self-host Prometheus/Grafana (open source)

Cost: $0/month
Credibility: High (still AWS + IaC)
Duration: 12 months
```

### Path B: Google Cloud (Better Free Tier)

```
1. Sign up for GCP ($300 credit for 90 days)
2. Use Cloud Run (serverless, 2M requests free)
3. Or use Compute Engine f1-micro (always free)
4. Same GitHub Actions for CI/CD

Cost: $0/month (or use credits)
Credibility: High (GCP is enterprise-grade)
Duration: 90 days (credit) or always (free tier)
```

### Path C: Railway/Render (Easiest)

```
1. Sign up for Railway (free tier)
2. Connect GitHub repo
3. Auto-deploy on push
4. Built-in observability

Cost: $0/month
Credibility: Medium (less "enterprise")
Duration: Unlimited (with usage limits)
Effort: Minimal
```

---

## Modified PR Stack for Free/Student

### Changes to Original Plan

| Original (Paid) | Free/Student Alternative | PR Impact |
|-----------------|--------------------------|-----------|
| AWS ECS/Fargate | AWS EC2 free tier OR GCP Cloud Run | PR7 (minor) |
| Application Load Balancer | Skip (use public IP) OR built-in HTTPS | PR7 (remove) |
| Multi-AZ deployment | Single instance (acceptable for demo) | PR7 (simplify) |
| ECR (registry) | GitHub Container Registry (free) | PR6 (change) |

### Updated PR7 (Free Tier Version)

**Title:** `Provision Staging Deployment on EC2 Free Tier (or GCP Cloud Run)`

**Terraform for EC2:**
```hcl
resource "aws_instance" "miniflow" {
  ami           = "ami-0c55b159cbfafe1f0"  # Amazon Linux 2
  instance_type = "t3.micro"               # Free tier eligible

  user_data = templatefile("${path.module}/user_data.sh", {
    docker_image = var.docker_image
  })

  vpc_security_group_ids = [aws_security_group.miniflow.id]

  tags = {
    Name = "miniflow-staging"
  }
}

resource "aws_security_group" "miniflow" {
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Open to internet (for demo only)
  }

  ingress {
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Grafana
  }
}
```

---

## Interview Narrative for Free/Student

### If Using AWS Free Tier:
> "I architected the deployment to run on AWS EC2 free tier to minimize costs while demonstrating production patterns. The infrastructure is managed with Terraform, includes CI/CD, observability with Prometheus and Grafana, and can easily migrate to ECS/Fargate for production scaling."

**This is still impressive** - you're showing:
- Cost consciousness
- AWS knowledge
- IaC skills
- Observability
- CI/CD

### If Using GCP Cloud Run:
> "I chose GCP Cloud Run for the serverless free tier, which provides auto-scaling and HTTPS out of the box. This let me focus on the application and observability while keeping costs at zero. The architecture is portable - the same container runs on AWS ECS or Kubernetes."

**Also impressive** - shows:
- Multi-cloud awareness
- Serverless architecture
- Cost optimization
- Portability

---

## Cost Comparison Summary

| Deployment Option | Monthly Cost | Interview Value | Effort |
|-------------------|--------------|-----------------|--------|
| AWS ECS/Fargate | ~$200 | ⭐⭐⭐⭐⭐ High | Medium |
| **AWS EC2 Free Tier** | **$0** | ⭐⭐⭐⭐☆ Good | Medium |
| **GCP Cloud Run** | **$0** | ⭐⭐⭐⭐☆ Good | Low |
| **Railway.app** | **$0** | ⭐⭐⭐☆☆ Okay | Low |
| AWS (Student Credits) | $0 (credits) | ⭐⭐⭐⭐⭐ High | Medium |

---

## Bottom Line

**Can you do this for free? YES.**

**Best options:**
1. **AWS EC2 Free Tier** - Same architecture, just cheaper instance
2. **GCP Cloud Run** - Serverless, generous free tier
3. **AWS Educate Credits** - Use student credits for original plan

**Interview impact:** Still strong. You're demonstrating:
- AWS/cloud knowledge (regardless of specific service)
- Infrastructure as Code
- CI/CD
- Observability
- Cost optimization (bonus points!)

**The narrative:** "I built a production-grade deployment with zero cost by leveraging free tiers and optimizing for the demo use case."

This shows **practical engineering judgment** - a highly valued trait.
