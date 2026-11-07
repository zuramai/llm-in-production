# Data Access Policy

*Owner: Security & Data Platform · Last reviewed: 2026-10-14*

## Purpose

This policy explains who can access ShopFlow data, how we handle personally identifiable information (PII), and which rules apply to analytics tools and AI/LLM assistants. It applies to all employees, contractors, and automated tools that touch the production Postgres database `shopflow`.

## Principles

- **Least privilege:** access is granted only for what your role needs.
- **PII stays where it belongs:** customer contact details are used for support and fulfillment, not for analysis or casual sharing.
- **No workarounds:** if a tool cannot see a field, don't route around it. Ask for a reviewed exception.

## Access by Role

| Role | Typical access | Notes |
|---|---|---|
| Support agents (~40) | Support tooling: order, ticket, and customer contact details for the customer they are helping | No direct database access |
| Support team leads | Same as agents, plus refund approval up to $500 | |
| Finance | Refund approvals above $500, finance reporting | Requests arrive via the finance-approvals channel |
| Analysts / Data | `shopflow_ro` (read-only) | Cannot read `customers.email` |
| Engineers | Production access via on-call or approved request only | Time-limited, logged |
| AI/LLM tools | `shopflow_ro` only | Same restrictions as analysts, no exceptions |

## The Read-Only Role: `shopflow_ro`

All analytics dashboards, notebooks, and AI/LLM tools must connect with `shopflow_ro`.

- It has **SELECT-only** permissions. It cannot insert, update, or delete.
- It **cannot read `customers.email`**. Queries that select this column will fail with a permission error. Do not request a broader role to get around this.
- Use customer IDs for joins and counts. You rarely need an email for analysis.

### Data conventions for queries

- Money is stored as **integer cents**. Divide by 100 for USD.
- Timestamps are **UTC `timestamptz`**. Convert to America/New_York (ET) when reporting by business day.
- Use the finance-approved metric definitions (transaction, gross revenue, net revenue, AOV, active customer, refund rate). Don't create variants for external reporting.

## PII Rules

PII includes names, emails, phone numbers, shipping and billing addresses, and anything that identifies a person.

1. **Never post customer emails in Slack**, including in channels, threads, DMs, and screenshots. Refer to the order ID or ticket ID instead.
2. Do not paste PII into AI tools, public documents, or personal notes.
3. Do not export PII to spreadsheets or local files unless a approved process requires it. Delete any such file when finished.
4. Card data is handled by Stripe and PayPal. ShopFlow never stores or requests full card numbers. Never ask a customer to send one.
5. Share the minimum needed. An order ID is almost always enough.

**Example**
- ❌ "Customer jane@example.com says her package is missing."
- ✅ "Order #48213 (ticket 90117): delivered-but-missing, within the 14-day window."

## Refund and Account Actions

- Refunds are issued only to the **original payment method**, never to a different card or account.
- Refunds are idempotent per order and reason. Do not retry by creating duplicate entries.
- Approval limits: agent up to $100, team lead up to $500, above $500 requires Finance.
- Account deletion requests must come from the email address on the account. Personal data is removed within 30 days, except records required for tax purposes.

## Requesting Access or an Exception

1. Open a request with your manager's approval and a business justification.
2. Security & Data Platform reviews it. Approved access is time-limited.
3. Access is logged and reviewed quarterly.

Exceptions to the `shopflow_ro` email restriction are not granted for AI tools.

## Reporting a Suspected Exposure

If you posted PII in the wrong place or suspect a data leak:

1. Delete the message if you can, and note where it was posted.
2. Notify Security & Data Platform immediately.
3. If customer-facing systems are affected, the incident is classified using our severity levels (SEV1-SEV3) and handled by on-call.

Reporting quickly is always better than staying quiet. We treat prompt reports as a good-faith action.
