# MiniFlow Learning Resources

Complete learning path for deploying ML models to production using AWS and Terraform.

---

## Who Is This For?

- ML engineers new to cloud deployment
- Students learning production ML
- Anyone who has never used AWS or Terraform
- People who want to understand *why* decisions are made, not just *how*

---

## Learning Path

### Phase 1: Foundations (Start Here)

**Goal**: Understand core concepts before touching code.

| Resource | Time | What You'll Learn |
|----------|------|-------------------|
| [AWS Foundations](./aws-foundations.md) | 2 hours | Why cloud, AWS core concepts, services for ML |
| [GCP Foundations](./gcp-foundations.md) | 2 hours | Why GCP, Google Cloud concepts, services for ML |
| [Terraform Foundations](./terraform-foundations.md) | 2 hours | IaC concepts, Terraform workflow, state management |
| [Architecture Decisions](./architecture-decisions.md) | 1 hour | Why we chose specific technologies, alternatives, trade-offs |
| [Networking Guide](./networking-guide.md) | 1.5 hours | VPC, subnets, security groups, load balancers, patterns |
| [Staging vs Production](./staging-vs-production.md) | 1 hour | Environment strategy, deployment patterns, cost optimization |
| [Testing Guide](./testing-guide.md) | 2 hours | Unit, integration, E2E tests, ML-specific testing, CI/CD testing |

**After Phase 1**: You understand what AWS and GCP are, why we use them, what Terraform does, how networking works, environment strategy, and testing best practices.

---

### Phase 2: Hands-On Deployment

**Goal**: Deploy MiniFlow to AWS successfully.

| Resource | Time | What You'll Do |
|----------|------|----------------|
| [Deployment Guide](./deployment-guide.md) | 3-4 hours | Set up AWS, configure Terraform, deploy, verify, cleanup |

**After Phase 2**: You have a running ML API on AWS that responds to requests.

---

### Phase 3: Production Hardening (PR7+)

**Goal**: Make deployment production-ready.

Coming soon:
- SSL/TLS setup
- Custom domain with Route 53
- CloudWatch monitoring and alerts
- Auto-scaling configuration
- Security hardening (private subnets, WAF)

---

## Quick Reference

### Why This Stack?

| Technology | Purpose | Why Chosen |
|------------|---------|------------|
| **AWS** | Cloud provider | Most job listings, comprehensive services |
| **ECS Fargate** | Container orchestration | Serverless, simpler than Kubernetes |
| **ALB** | Load balancer | HTTP routing, health checks, SSL |
| **Terraform** | Infrastructure as Code | Multi-cloud, industry standard, state management |
| **GHCR** | Container registry | Free, integrated with GitHub |

### Key Concepts to Master

1. **VPC and Subnets**: Your isolated network in AWS
2. **ECS**: How containers are run and managed
3. **IAM**: Who can do what (security)
4. **Terraform State**: How Terraform remembers what it created
5. **Load Balancing**: Distributing traffic across containers
6. **Security Groups**: Firewall rules for resources
7. **Staging vs Production**: Environment strategy and best practices
8. **Testing Pyramid**: Unit, integration, E2E tests for ML systems

---

## Common Questions

### "Do I need to learn all of AWS?"

**No.** Focus on these services only:
- VPC (networking)
- ECS (containers)
- ALB (load balancing)
- IAM (permissions)
- CloudWatch (logs)

Everything else can wait.

---

### "Why not use [Kubernetes/SageMaker/Lambda]?"

See [Architecture Decisions](./architecture-decisions.md) for detailed comparison.

**Short answer**:
- **Kubernetes**: Powerful but complex (overkill for single service)
- **SageMaker**: Expensive, more suited for training than API deployment
- **Lambda**: 15-minute timeout, not suitable for ML inference
- **ECS**: Right balance of power and simplicity for learning

---

### "How much will this cost?"

**AWS Costs** (staging environment):
- ~$45/month if running 24/7
- ~$0 if destroyed when not using
- Free tier available for 12 months (but not for ECS Fargate)

**Cost control**:
- Budget alerts at $10, $25, $50
- Always run `terraform destroy` when done
- Use smaller instances (1 vCPU, 2 GB)

---

### "What if I don't have a credit card?"

**Options**:
1. **Azure for Students**: $100 credit, no credit card required with student email
2. **Google Cloud**: $300 credit, requires credit card but won't charge
3. **Local deployment**: Use docker-compose locally (no cloud costs)

See [Architecture Decisions](./architecture-decisions.md) for cloud fallback strategy.

---

## Learning Checklist

### Before First Deploy
- [ ] Created AWS account
- [ ] Set up budget alerts
- [ ] Created IAM admin user
- [ ] Configured AWS CLI
- [ ] Read AWS Foundations guide
- [ ] Read Terraform Foundations guide
- [ ] Read Architecture Decisions

### First Deploy
- [ ] Got VPC and subnet IDs
- [ ] Created terraform.tfvars
- [ ] Ran `terraform init`
- [ ] Ran `terraform plan`
- [ ] Ran `terraform apply`
- [ ] Verified endpoints respond
- [ ] Viewed logs
- [ ] Ran `terraform destroy`

### Understanding
- [ ] Can explain what a VPC is
- [ ] Can explain what ECS does
- [ ] Can explain why we use ALB
- [ ] Can explain Terraform state
- [ ] Can troubleshoot a failed deployment

---

## Troubleshooting Resources

| Problem | Resource |
|---------|----------|
| AWS auth issues | [AWS Foundations - Security Basics](./aws-foundations.md#security-basics) |
| Terraform errors | [Terraform Foundations - Troubleshooting](./terraform-foundations.md#troubleshooting) |
| Deployment failures | [Deployment Guide - Troubleshooting](./deployment-guide.md#troubleshooting-common-issues) |
| Understanding choices | [Architecture Decisions](./architecture-decisions.md) |
| Networking issues | [Networking Guide - Troubleshooting](./networking-guide.md#troubleshooting-network-issues) |
| Environment strategy | [Staging vs Production](./staging-vs-production.md)

---

## External Resources

### AWS
- [AWS Free Tier](https://aws.amazon.com/free/)
- [AWS Documentation](https://docs.aws.amazon.com/)
- [AWS Pricing Calculator](https://calculator.aws/)

### Terraform
- [Terraform Tutorials](https://developer.hashicorp.com/terraform/tutorials)
- [AWS Provider Docs](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)

### ML Deployment
- [AWS ML Deployment Guide](https://docs.aws.amazon.com/whitepapers/latest/deploying-machine-learning-models-on-aws/deploying-machine-learning-models-on-aws.html)
- [MLOps Best Practices](https://aws.amazon.com/blogs/machine-learning/best-practices-for-mlops/)

---

## Contributing

These docs evolve based on:
- Questions you have (add to troubleshooting)
- Mistakes made (add warnings)
- Better explanations (clarify concepts)

If something doesn't make sense, ask!

---

## Next Steps

1. **Start with**: [AWS Foundations](./aws-foundations.md)
2. **Then read**: [Terraform Foundations](./terraform-foundations.md)
3. **Understand choices**: [Architecture Decisions](./architecture-decisions.md)
4. **Compare clouds**: [AWS vs GCP Deployment](./aws-vs-gcp-deployment.md) ← NEW: Which cloud to choose
5. **Learn networking**: [Networking Guide](./networking-guide.md)
6. **Understand environments**: [Staging vs Production](./staging-vs-production.md)
7. **Deploy**: [Deployment Guide](./deployment-guide.md)

---

**Remember**: Every expert was once a beginner. Take your time, follow the guides, and don't hesitate to destroy and recreate infrastructure - that's what IaC is for!
