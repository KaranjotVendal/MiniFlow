# MiniFlow SLO Draft

## Service-level objectives (staging)
1. Availability SLO: 99.0% successful `/health` checks.
2. Request success SLO: >= 95% `/s2s` requests return 2xx.
3. Latency SLO: `/s2s` p95 latency below agreed benchmark threshold for active model profile.

## Error budget policy
1. If request success drops below SLO for two consecutive windows, freeze non-critical releases.
2. Trigger rollback if latency regresses beyond accepted threshold after deploy.

## Alert suggestions
1. `miniflow_requests_total{status="error"}` spike.
2. `miniflow_request_latency_seconds` p95 breach.
3. `/ready` endpoint not-ready states.
