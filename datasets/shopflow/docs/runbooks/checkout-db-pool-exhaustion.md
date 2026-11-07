# Runbook: Checkout DB Connection Pool Exhaustion

*Owner: Platform Engineering (Checkout) · Last reviewed: 2026-10-14*

## Overview

The `checkout` service (FastAPI) holds a fixed-size pool of Postgres connections to the production `shopflow` database. The default `db_pool_size` is 10. When every connection is in use, new requests wait for a free one and eventually fail with a 5xx. Because `POST /checkout` creates orders, exhaustion directly blocks revenue.

**Severity guidance**
- Error rate >5% or p95 latency >2s for 10 min: **SEV2**.
- Payments or checkout failing for >20% of requests: **SEV1**. Page on-call immediately and update the status page within 15 minutes.

## Symptoms

Open the Grafana dashboard **Checkout Overview** (job="checkout") and look for:

- `db_pool_connections_in_use` pinned at `db_pool_size`
- `db_pool_wait_seconds` rising (check p95/p99 of the histogram)
- `http_requests_total{route="/checkout",method="POST",status=~"5.."}` increasing
- `http_request_duration_seconds{route="/checkout"}` p95 climbing

Confirm in Loki:

```
{service="checkout", level="error"} |= "db pool exhausted"
```

Expected message: `db pool exhausted: timed out waiting for connection`. Note the `deploy_version` field on these lines.

## Diagnosis

1. **Check for a recent deploy.** Look in #deploys or `deploys.json` for a `deployed_at` (UTC) shortly before the symptoms started. If the errors begin right after a `deploy_version` change, suspect leaked connections (sessions not released) and go to *Mitigation C*.
2. **Check for slow queries.** Connect with the read-only role `shopflow_ro` and inspect `pg_stat_activity` for long-running or `idle in transaction` sessions from the checkout service:
   ```sql
   SELECT pid, state, now() - query_start AS runtime, left(query, 120)
   FROM pg_stat_activity
   WHERE datname = 'shopflow' AND state <> 'idle'
   ORDER BY runtime DESC LIMIT 20;
   ```
   Many `idle in transaction` sessions suggest leaks; a few very long active queries suggest slow queries or lock contention.
3. **Check for a traffic spike.** Compare `http_requests_total` rate against the same time last week. A spike with healthy query times points to undersized capacity (e.g., a sale event).
4. **Rule out the payment provider.** If `payment_provider_request_duration_seconds` p95 is high for one provider, connections may be held while waiting on charges. See the payment provider timeout runbook.

## Mitigation

**A. Kill long-running queries** (slow-query cause)
- Terminate offending backends with `pg_terminate_backend(pid)` (requires a write-capable DBA role, not `shopflow_ro`). Only target checkout sessions; never terminate replication or migration sessions.
- Capture the query text for the follow-up ticket.

**B. Raise the pool size temporarily** (traffic spike)
- Increase `db_pool_size` from 10 in small steps (e.g., to 20) and redeploy or reload config per the service config process.
- Verify Postgres `max_connections` has headroom across all checkout replicas first. Revert after the incident.

**C. Roll back** (leaks after a deploy)
```
deployctl rollback checkout --to <previous_version>
```
Use the last version before the suspect deploy, e.g. `v2026.11.07-1`. Roll back first; investigate later.

## Verification

- `db_pool_connections_in_use` drops below `db_pool_size`.
- `db_pool_wait_seconds` returns to baseline.
- 5xx rate on `POST /checkout` returns to normal and `db pool exhausted` log lines stop.
- Hold for 15 minutes before downgrading severity.

## Cautions

- Orders stuck in **pending** with no successful payment auto-cancel after 48h. Do not manually alter order statuses during the incident.
- If customers were charged but an order wasn't created, hand off to Support and Finance. Refunds are idempotent per order+reason and always go to the original payment method.

## Follow-up

- Post a summary in #deploys if a rollback occurred.
- File a ticket with the root cause, the query or code path, and the timeline in UTC.
- Add a pool saturation alert (in-use / size > 0.9 for 5 min) if one did not fire.
