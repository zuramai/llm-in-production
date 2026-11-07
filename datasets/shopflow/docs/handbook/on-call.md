# Engineering On-Call Handbook

*Owner: Platform Engineering · Last reviewed: 2026-10-14*

This handbook covers how on-call works for ShopFlow engineering: rotation, paging, severity levels, communication, and postmortems. Service-specific steps are in the runbooks (for example, the `checkout` runbooks).

## Rotation

- One primary on-call engineer at a time, with one secondary as backup. Rotations are weekly and hand off on Monday at 10:00 AM ET.
- Handoff is a short written note in the on-call channel covering open incidents, flaky alerts, and risky deploys in flight.
- Swaps are allowed. Arrange them with the other engineer and update the schedule before the shift starts.
- Support (about 40 agents) escalates to engineering through the on-call channel. Support does not page engineering directly for SEV3 issues.

## Paging

- Any alert that matches SEV1 criteria pages the primary immediately. If the primary does not acknowledge within 5 minutes, the page goes to the secondary.
- Acknowledge the page, then post "ack" in the incident channel so others know someone owns it.
- Start with the Grafana dashboard **Checkout Overview** (`job="checkout"`), then check Loki logs with `{service="checkout", level="error"}`.
- Check `#deploys` (or `deploys.json`) for any change shortly before the symptoms began. If a deploy matches, rolling back comes first, and investigation happens afterward: `deployctl rollback checkout --to <version>`.

## Severities

| Severity | Definition | Response |
|---|---|---|
| **SEV1** | Checkout down, or payments failing >20% | Page on-call immediately. Update the status page within **15 minutes**. |
| **SEV2** | Degraded: p95 latency >2s or error rate >5% for 10 minutes | Notify on-call. Open an incident channel. Update the status page if customers are visibly affected. |
| **SEV3** | Minor, with no meaningful customer impact | Handle during business hours through a ticket. |

If you are unsure between two levels, choose the higher one. You can downgrade later.

## Running an incident

1. **Open a channel** named `#inc-YYYYMMDD-short-name` (date in ET). Pin the dashboard link and the alert.
2. **Assign roles.** The primary on-call is the incident commander by default. For SEV1, add a communications owner who handles the status page and updates to Support.
3. **Mitigate first.** Use the runbook for the matching scenario:
   - DB connection pool exhaustion: `db_pool_connections_in_use` pinned at `db_pool_size`
   - Bad deploy: errors jump after a `deploy_version` change
   - Payment provider timeout: one provider's p95 above 5s. Use the `payments.primary_provider` flag to fail over, and never retry charges blindly.
4. **Post updates** in the channel at least every 30 minutes during SEV1 and every hour during SEV2, even if the update is "no change."
5. **Record a timeline** as you go. Use UTC timestamps in notes, since timestamps are stored as UTC `timestamptz`. Convert to ET only when writing customer-facing text.

## Status page

- Post an initial update within 15 minutes of a SEV1 declaration. Say what customers experience, not the suspected cause.
- Use plain language, for example: "Some customers may be unable to complete checkout."
- Resolve the status page entry only after metrics have been healthy for at least 30 minutes.

## Data access during incidents

- Use the read-only role `shopflow_ro` for investigation queries. It cannot read `customers.email`, so do not paste customer emails into incident channels.
- Money is stored as integer cents. Do not hand-edit orders or refunds in production. Refunds go through the standard tooling, are idempotent per order and reason, and always go to the original payment method.

## Postmortems

- Required for every SEV1 and SEV2, and optional for SEV3.
- Publish a blameless postmortem within **5 business days** of resolution (business days in ET).
- Include: summary, customer impact, timeline (UTC), root cause, what went well and poorly, and action items with owners and due dates.
- Review it in the weekly engineering sync. Action items are tracked until closed.

## After your shift

- Hand off any open items in the written note.
- File tickets for noisy or unactionable alerts so they get fixed or removed.
- Update the runbooks if a step was missing or wrong.
