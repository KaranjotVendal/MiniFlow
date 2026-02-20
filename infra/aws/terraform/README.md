# MiniFlow AWS Staging (Terraform)

This module provisions a minimal ECS/Fargate staging stack for MiniFlow.

## Includes
1. ECS cluster/service/task definition
2. ALB + target group + listener
3. CloudWatch log group
4. Budget alert resource (`$10/$25/$50`) when `budget_alert_email` is set

## Required inputs
1. `container_image`
2. `vpc_id`
3. `public_subnet_ids`

## Apply
```bash
terraform init
terraform plan \
  -var "container_image=ghcr.io/<owner>/miniflow:sha-<shortsha>" \
  -var "vpc_id=vpc-xxxx" \
  -var 'public_subnet_ids=["subnet-a","subnet-b"]'
terraform apply \
  -var "container_image=ghcr.io/<owner>/miniflow:sha-<shortsha>" \
  -var "vpc_id=vpc-xxxx" \
  -var 'public_subnet_ids=["subnet-a","subnet-b"]'
```

## Cost control
1. Use `desired_count=1` for staging.
2. Tear down stack when not actively testing.
3. Keep budget alarms enabled.
