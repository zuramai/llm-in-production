# Incident Severity Definitions

*Owner: Platform Reliability · Last reviewed: 2026-10-14*

This page defines how ShopFlow classifies production incidents, what each severity requires, and how the checkout service scenarios map to those levels. When in doubt between two levels, declare the higher one and downgrade later.

## Severity Levels

| Severity | Definition | Response |
|---|---|---|
| **SEV1** | Checkout is down, or payments are failing for more than 20% of attempts | Page on-call immediately. Update the status page within 15 minutes. |
| **SEV2** | Degraded service: p95 latency above 2s, or error rate above 5%, sustained for 10 minutes | Notify on-call and open an incident channel. Investigate right away. |
| **SEV3** | Minor issue with limited customer impact and a workaround available | Track as a ticket and fix during normal working hours. |

## Examples

**SEV1**
- `checkout` returns 5xx on nearly all `POST /checkout` requests.
- Stripe and PayPal are both failing, or one provider fails and failover does not recover the failure rate below 20%.
- DB connection pool exhaustion is blocking order creation for most users.

**SEV2**
- p95 of `http_request_duration_seconds` for checkout routes stays above 2s for 10 minutes.
- Error rate above 5% right after a deploy (`deploy_version` change) with "unhandled exception" logs.
- One payment provider is timing out (p95 of `payment_provider_request_duration_seconds` above 5s) while the other provider is healthy and carries traffic.

**SEV3**
- A single non-critical route shows intermittent errors with no effect on checkout.
- A dashboard panel in 'Checkout Overview' is missing data.
- A cosmetic issue in order confirmation emails.

## Mapping Checkout Scenarios

- **DB pool exhaustion:** Start at SEV2 when `db_pool_connections_in_use` is pinned at `db_pool_size` and `db_pool_wait_seconds` is rising. Escalate to SEV1 if checkout is effectively down.
- **Bad deploy:** Treat as SEV2 at minimum. Escalate to SEV1 if checkout is down or payments fail above 20%. Roll back first with `deployctl rollback checkout --to <version>`; investigate afterward.
- **Payment provider timeout:** Usually SEV2. Fail over using the `payments.primary_provider` feature flag. Never retry charges blindly; rely on idempotency keys. Escalate to SEV1 if payment failures exceed 20% after failover.

## Response Expectations

1. **Declare.** The first responder sets the severity in the incident channel and states the evidence (dashboard link, query, or log line).
2. **Assign roles.** Name an incident commander and a communications owner for SEV1 and SEV2.
3. **Mitigate before diagnosing.** Prefer rollback, failover, or temporary capacity changes over live debugging.
4. **Communicate.** For SEV1, the status page is updated within 15 minutes and refreshed as the situation changes. Support leads are told so agents can answer customers consistently.
5. **Record.** Link relevant entries from `deploys.json` or `#deploys` in the incident timeline.
6. **Resolve and review.** Downgrade or close only when metrics have been stable for at least 30 minutes. A written review is expected for every SEV1 and SEV2.

## Customer Impact Notes

- A customer whose payment failed is never charged for the failed attempt. The order stays *pending* and is auto-cancelled after 48 hours.
- Any refund created as part of incident remediation follows normal rules: it goes to the original payment method only, is idempotent per order and reason, and respects approval limits (agent up to $100, team lead up to $500, above $500 via Finance in the finance-approvals channel).
- Support first-response SLAs still apply during incidents: urgent 1h, high 4h.

## Adjusting Severity

Anyone on the incident may propose a change. The incident commander decides, and the change and reason are noted in the channel.
