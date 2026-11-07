# Support Escalation Handbook

*Owner: Customer Support Operations · Last reviewed: 2026-10-14*

This handbook explains when and how support agents escalate tickets. It covers ticket priorities, first-response SLAs, refund approval limits, and the cases that always go to a specialist. Internal use only.

## Escalation Tiers

| Tier | Who | Handles |
|------|-----|---------|
| 1 | Support agent | Standard tickets, refunds up to **$100.00** |
| 2 | Team lead | Refunds up to **$500.00**, disputes, repeat contacts, policy exceptions within limits |
| 3 | Finance | Refunds **above $500.00** |
| Urgent | On-call / safety handling | Safety issues, suspected account compromise, checkout or payment outages |

Escalate when you cannot resolve a ticket within your authority. Do not hold a ticket to avoid escalating, since the SLA clock keeps running.

## Ticket Categories and Priorities

Categories: `shipping`, `refund`, `product_defect`, `account`, `payment`, `other`.
Priorities: `low`, `medium`, `high`, `urgent`.

## First-Response SLA

| Priority | First response |
|----------|----------------|
| urgent | 1 hour |
| high | 4 hours |
| medium | 1 business day |
| low | 2 business days |

Business days and hours are measured in ET. Support hours are Mon-Fri 9AM-6PM ET and Sat 10AM-4PM ET. Customers are told email replies arrive within one business day, so never promise faster unless the ticket is high or urgent.

## When to Escalate

### To a team lead
- The refund is more than $100.00 and up to $500.00.
- The customer disputes a policy decision (for example, an opened beauty item outside the damaged/defective exception).
- The customer has contacted us three or more times on the same issue.
- A delivered-but-missing carrier investigation has stalled.

### To Finance
- The refund is above $500.00. Team lead reviews first, then posts the request in the **finance-approvals** channel with the order ID, amount, reason, and payment method.
- Anything that would require refunding to a different card or account. **This is never allowed.** Refunds always go to the original payment method. Tell the customer this plainly rather than escalating for an exception.

### To the safety team (urgent)
Treat any report of overheating, smoke, or swelling batteries as **urgent** immediately:
1. Tell the customer to stop using the product.
2. Set category `product_defect`, priority `urgent`.
3. Ask for photos if the customer can take them safely.
4. Escalate to the safety team and your team lead. Do not wait for the return to be received.

Respond within **1 hour**.

### To on-call engineering
Escalate to the incident process if you see multiple customers reporting failed checkout or payments. Severity is set by engineering:
- **SEV1**: checkout down or payments failing >20%. On-call is paged immediately; status page updated within 15 minutes.
- **SEV2**: degraded service (p95 latency >2s or error rate >5% for 10 minutes).
- **SEV3**: minor issues.

## Refund Rules to Remember

- Refunds are **idempotent per order + reason**. Check the order history before issuing; do not submit a second refund for the same order and reason.
- Return window: 30 days from delivery. Defective or wrong items: return shipping is free; otherwise $5.99 is deducted.
- Timeline to quote customers: 5-7 business days after receipt and inspection (or after cancellation).
- Damaged/defective: photo required, reported within 30 days of delivery. Offer a free replacement or full refund including shipping.
- Missing packages: ask the customer to wait 24 hours after the delivered scan; contact must be within 14 days of delivery.

## Account Security

If a customer suspects unauthorized access, lock the account immediately, set the ticket to `account` with priority `high` at minimum, and notify your team lead. Account deletion requests must come from the email on the account and are completed within 30 days, except tax records.

## Handoff Notes

When escalating, include in the ticket: order ID, customer segment (consumer, business, or vip), what was already tried, the amount at stake, and the specific decision needed. Tell the customer who now owns the ticket and when to expect the next update.
