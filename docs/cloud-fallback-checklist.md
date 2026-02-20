# Cloud Fallback Checklist

If AWS staging is blocked by budget/credits, keep application contracts unchanged and switch infra target.

## Preconditions
1. GHCR image exists for the release tag.
2. App health endpoints are stable (`/health`, `/ready`).
3. Secrets are mapped for target provider.

## Fallback sequence
1. Azure managed container runtime (student credits)
2. DigitalOcean managed container runtime
3. Another low-cost managed container host

## Validation checklist
1. Deploy image tag from GHCR.
2. Confirm `/health` and `/ready` responses.
3. Confirm logs and request IDs are present.
4. Execute rollback to previous known-good image tag.
