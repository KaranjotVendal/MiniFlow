# Deployment Notes

## Image Publishing

MiniFlow publishes versioned container images to GHCR via `.github/workflows/cd.yml`.

Tag strategy:
1. `sha-<shortsha>` immutable release tag
2. `main` mutable convenience tag on default branch

## Pull and Run

```bash
docker pull ghcr.io/<owner>/miniflow:sha-<shortsha>
docker run --rm -p 8000:8000 ghcr.io/<owner>/miniflow:sha-<shortsha>
```

## Rationale

Current release target is GHCR for fast setup and demo velocity.

## TODO (Future Work)

Add AWS ECR publish (or GHCR -> ECR mirroring) in a follow-up PR.
Reasons:
1. AWS-native runtime alignment for ECS/Fargate production path.
2. Better IAM-native access control in AWS environments.
