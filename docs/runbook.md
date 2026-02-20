# MiniFlow Runbook

## Scope
Operational procedures for staging/demo environment.

## Service checks
1. Liveness: `GET /health`
2. Readiness: `GET /ready`
3. Runtime metrics: `GET /metrics`

## Incident triage
1. Capture failing request ID from application logs.
2. Check `/ready` response and config completeness fields.
3. Check request error counters in `/metrics`.
4. Check ECS task logs and recent deploy metadata.

## Rollback procedure
1. Identify previous known-good image tag (`sha-<shortsha>`).
2. Redeploy staging with previous tag via deploy workflow.
3. Re-run `/health` and `/ready` checks.
4. Execute one `/s2s` smoke request.
5. Record rollback event in incident notes.

## Cost control checklist
1. Keep `desired_count=1` for staging.
2. Disable/scale down staging outside test windows.
3. Verify budget alarms are active.
