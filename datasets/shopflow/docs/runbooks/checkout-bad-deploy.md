# Runbook: Checkout Bad Deploy

*Owner: Platform Engineering · Last reviewed: 2026-10-14*

## Purpose

Use this runbook when the `checkout` service (FastAPI, Prometheus `job="checkout"`) starts failing right after a deploy. Checkout creates orders and calls Stripe and PayPal, so a bad release can directly block revenue. **Roll back first, investigate later.**

## Severity

| Condition | Severity |
|---|---|
| Checkout down, or payments failing >20% | **SEV1**: page on-call immediately, post to status page within 15 min |
| Error rate >5% or p95 latency >2s for 10 min | **SEV2** |
| Minor regression, limited to a non-critical route | SEV3 |

If unsure, treat it as the higher severity and downgrade later.

## Symptoms

- 5xx rate on `http_requests_total{status=~"5.."}` jumps within minutes of a deploy.
- `unhandled exception` log lines appear with a **new** `deploy_version`.
- Other signals (DB pool, payment provider latency) look normal. If they don't, see the pool exhaustion or provider timeout runbooks.

## 1. Confirm the correlation

1. Open the **Checkout Overview** dashboard in Grafana and look at the 5xx rate and p95 of `http_request_duration_seconds` by `route`.
2. Check the last deploy in `#deploys` or `deploys.json` (`version`, `sha`, `deployed_at` in UTC, `author`, `summary`).
3. Confirm the error spike starts at or just after `deployed_at`.
4. In Loki, confirm errors are tied to the new version:

```
{service="checkout", level="error"} | json | msg="unhandled exception"
```

Group by `deploy_version`. If errors appear only on the new version, this is a bad deploy. If both old and new versions error, look elsewhere (database, provider).

Useful PromQL:

```
sum(rate(http_requests_total{job="checkout",status=~"5.."}[5m]))
  / sum(rate(http_requests_total{job="checkout"}[5m]))
```

## 2. Mitigate: roll back

Do not wait for a root cause. Identify the previous known-good version in `#deploys` (e.g. `v2026.11.07-1`) and run:

```
deployctl rollback checkout --to <version>
```

Then announce in `#deploys` and the incident channel: the version rolled back from and to, and who executed it.

If the deploy included a database migration, check with the author before rolling back that the old code is compatible with the new schema. Don't delay the rollback for a long discussion; escalate to the on-call lead if unclear.

## 3. Verify recovery

Within 10 minutes of the rollback:

- 5xx ratio returns to baseline on Checkout Overview.
- `http_request_duration_seconds` p95 is back under 2s.
- New `unhandled exception` lines stop, and `deploy_version` in logs shows the rolled-back version.
- `db_pool_connections_in_use` is below `db_pool_size` (default 10).
- `payment_provider_errors_total` is flat for both `stripe` and `paypal`.

## 4. Check for customer impact

- Orders left `pending` by failed checkouts are auto-cancelled after 48h. No manual action is needed.
- Failed payment attempts are not charged, though a temporary bank authorization hold may appear for 1–3 business days. Tell Support so agents can answer customer questions (support@shopflow.example).
- Do not retry charges blindly. Reuse the original idempotency key when replaying anything.
- For SEV1, update the status page and resolve it once metrics are stable.

## 5. Investigate afterward

1. Reproduce the failure using the offending `sha` in staging.
2. Use the `error` and `route` log fields to find the failing code path.
3. Fix forward with a new version, or revert the commit, then redeploy with a canary check on Checkout Overview.
4. Open a post-incident review for every SEV1 and SEV2, covering the timeline, detection delay, and why tests didn't catch it.

## Escalation

- On-call engineer → on-call lead if no recovery within 15 minutes of rollback.
- Notify the deploy `author` as soon as the rollback completes.
- Refund or charge-correction questions go through the standard approval limits (agent ≤$100, team lead ≤$500, above $500 via Finance in the finance-approvals channel).
