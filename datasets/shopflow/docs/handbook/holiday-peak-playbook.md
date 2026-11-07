# Holiday Peak Playbook (Black Friday – Cyber Monday)

*Owner: Support Operations & Platform Engineering · Last reviewed: 2026-10-14*

## Purpose

This playbook covers how Support and Engineering prepare for and run the Black Friday through Cyber Monday window and the weeks around it. It supplements the standard incident and ticket policies; it does not replace them.

## Key Dates (all times ET)

| Milestone | Date |
|---|---|
| Change freeze begins | Mon, Nov 16 |
| VIP early sale access opens | Wed, Nov 25, 9:00 AM |
| Black Friday | Fri, Nov 27 |
| Cyber Monday | Mon, Nov 30 |
| Freeze lifts | Thu, Dec 3 |

Only fixes for SEV1/SEV2 issues may ship during the freeze, and each needs sign-off from the engineering on-call lead.

## Engineering

**Change freeze**
- No schema migrations, dependency upgrades, or checkout and payment flow changes.
- Hotfixes need a rollback plan documented in the PR.

**Capacity**
- Run load tests against checkout by Nov 6, targeting 3x last year's peak order rate.
- Pre-scale web and worker pools before the VIP early access window opens.
- Confirm Stripe and PayPal webhook handling is healthy before each peak day.

**Monitoring and severity**
- SEV1: checkout down or payments failing >20%. Page on-call immediately; post to the status page within 15 minutes.
- SEV2: p95 latency >2s or error rate >5% for 10 minutes.
- SEV3: minor issues, handled in normal hours.
- On-call is double-staffed (primary + secondary) from Nov 25 to Dec 1.

**Data and reporting**
- Analytics and LLM tools use the read-only `shopflow_ro` role on the `shopflow` database. It cannot read `customers.email`. Don't run heavy ad-hoc queries on production during peak hours; use the read-only role off-peak.
- Money is stored as integer cents, timestamps are UTC `timestamptz`, and "business day" is defined in ET.
- Use finance-approved definitions: gross revenue counts only paid/shipped/delivered orders, and net revenue subtracts refunds created in the period.

## Fulfillment

- Orders placed before 2 PM ET on a business day ship the same day. Black Friday and Cyber Monday are business days.
- Confirm inventory by SKU across NJ-01 (Edison), TX-01 (Dallas), and CA-01 (Ontario) by Nov 20.
- Carrier cutoffs (UPS, FedEx, USPS, DHL) can shift during peak; Fulfillment posts updates in the ops channel.

## Support

**Staffing**
- Hours stay Mon-Fri 9 AM-6 PM and Sat 10 AM-4 PM ET. Support is not extended unless the Support Director announces otherwise.
- Block PTO for Nov 25 – Dec 2. Of our ~40 agents, schedule at least 75% on shift each weekday in that window.
- Assign one team lead per shift as escalation point and one agent as safety-ticket watcher.

**Ticket priorities and SLAs (first response)**
- urgent: 1 hour
- high: 4 hours
- medium: 1 business day
- low: 2 business days

Safety reports (overheating, smoke, swelling batteries) are always **urgent**. Tell the customer to stop using the product immediately.

**Expected top ticket drivers**
1. Shipping: "where is my order", delivered-but-missing. Advise 24 hours of waiting, then contact within 14 days of delivery, then open a carrier investigation.
2. Payment: failed payments. Failed attempts are never charged; temporary holds clear in 1–3 business days. Pending orders auto-cancel after 48 hours.
3. Cancellations: free until the order ships.
4. Returns: the window is 30 days from delivery, so holiday purchases are still returnable after the freeze lifts. Do not offer exceptions without a team lead.

**Refund approval limits**
- Agent: up to $100
- Team lead: up to $500
- Above $500: Finance, via the finance-approvals channel

Refunds always go to the original payment method and are idempotent per order + reason. Never refund to a different card. Don't re-submit a refund that appears stuck; check the order first.

## Customer Messaging

- Do not promise delivery dates for the holiday window. Quote standard ranges only (US standard 3–5 business days, express 1–2).
- ShopFlow does not sell gift cards, and does not accept crypto or cash on delivery. Decline these requests politely.
- Free standard shipping applies at $75.00 after discounts, and always for VIP.

## After the Peak

- Hold a retro within one week of the freeze lifting.
- Review SLA misses by category and any SEV1/SEV2 incidents.
- Update this playbook before Oct 31 of next year.
