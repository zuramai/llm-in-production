# Runbook: Payment Provider Timeout (Checkout)

*Owner: Payments Platform · Last reviewed: 2026-10-14*

## Summary

Use this runbook when one payment provider (Stripe or PayPal) is slow or timing out and `checkout` latency rises, while the other provider stays healthy. Customers see slow or failed payments. Failed attempts are never charged, and orders stay `pending` until a payment succeeds.

**Severity guide:**
- Payments failing >20%: **SEV1**. Page on-call immediately and update the status page within 15 minutes.
- p95 latency >2s or error rate >5% for 10 minutes: **SEV2**.
- A brief blip affecting a small share of requests: **SEV3**.

## Symptoms

- `payment_provider_request_duration_seconds{provider="<stripe|paypal>"}` p95 above 5s for one provider.
- `payment_provider_errors_total{reason="timeout"}` rising for that provider.
- `http_request_duration_seconds{route="/checkout"}` rising.
- Loki logs show `"payment provider timeout"` with a `provider` field.
- The other provider's p95 and error rate look normal.

## Triage (first 5 minutes)

1. Open the Grafana dashboard **Checkout Overview** and filter to `job="checkout"`.
2. Compare p95 by provider:
   ```
   histogram_quantile(0.95, sum by (le, provider) (rate(payment_provider_request_duration_seconds_bucket{job="checkout"}[5m])))
   ```
3. Check timeout rate by provider:
   ```
   sum by (provider) (rate(payment_provider_errors_total{job="checkout",reason="timeout"}[5m]))
   ```
4. Confirm the logs in Loki:
   ```
   {service="checkout", level="error"} | json | msg="payment provider timeout"
   ```
   Group by `provider`.
5. Rule out other causes:
   - **Recent deploy:** check `deploys.json` or #deploys. If errors began right after a `deploy_version` change, use the bad-deploy runbook instead.
   - **DB pool:** if `db_pool_connections_in_use` is pinned at `db_pool_size` and `db_pool_wait_seconds` is rising, use the DB pool exhaustion runbook.
6. Check the provider's public status page and note any acknowledged incident.

## Mitigation

1. **Fail over to the healthy provider** by changing the feature flag `payments.primary_provider` to the other provider (`stripe` or `paypal`).
2. Watch Checkout Overview for 5-10 minutes. Expect:
   - the unhealthy provider's request volume to fall,
   - `http_request_duration_seconds` for checkout to recover,
   - 5xx on `POST /checkout` to drop.
3. **Do NOT retry charges blindly.** A timed-out request may still have succeeded at the provider. Always reuse the original idempotency key for the same order and attempt. Never create a new charge for an order until the earlier attempt's outcome is confirmed.
4. Do not manually refund to a different card or method. Any refund goes to the original payment method, and refunds are idempotent per order and reason.

## Communication

- SEV1/SEV2: post in the incident channel and update the status page (within 15 minutes for SEV1).
- Tell Support leads so agents can answer "my payment failed" contacts. Standard guidance: failed attempts are not charged, a temporary bank authorization hold clears in 1-3 business days, and pending orders auto-cancel after 48 hours.

## Reconciliation after recovery

1. Find `pending` orders from the incident window that have a provider charge or authorization recorded.
2. Verify each against the provider dashboard using its idempotency key, then update the order status through the normal payment flow (`paid`).
3. For duplicate charges, refund through the standard process. Approval limits apply: agent up to $100, team lead up to $500, above $500 needs Finance (#finance-approvals).
4. Analytics queries use the read-only role `shopflow_ro`, which cannot read `customers.email`. Money is stored as integer cents and timestamps are UTC. Convert to ET for business-day reporting.

## Rollback of the mitigation

Switch `payments.primary_provider` back only when:
- the provider status page shows resolved,
- p95 stays below 5s (ideally below 2s) for 30 minutes,
- timeout errors are near zero.

Switch back during low traffic, and keep watching Checkout Overview.

## Follow-up

- Create a ticket in the post-incident tracker for any SEV1/SEV2.
- Record the timeline, flag change times, and the number of affected orders.
- Review whether alert thresholds (p95 >5s per provider) fired promptly.
