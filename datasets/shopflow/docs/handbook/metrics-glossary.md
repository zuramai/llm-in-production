# ShopFlow Metrics Glossary

*Owner: Finance Analytics · Last reviewed: 2026-10-14*

This glossary defines the finance-approved metrics used in dashboards, reports, and LLM/analytics tooling. If a number you produce disagrees with these definitions, the number is wrong, not the definition. Questions go to Finance Analytics.

## Ground Rules

- **Money is stored as integer cents** (e.g. `total_cents`). Divide by 100 only when presenting USD values. Never sum floats.
- **Timestamps are UTC `timestamptz`.** Reporting periods and "business days" are defined in **America/New_York (ET)**. Convert before bucketing.
- **Data access:** analytics and LLM tools use the read-only role `shopflow_ro` on the production Postgres database `shopflow`. This role cannot read `customers.email`, so do not build metrics that depend on it. Use customer IDs.
- **Segments** in the data are `consumer`, `business`, and `vip`.

## Definitions

| Metric | Definition |
|---|---|
| **Transaction** | Any order placed in the period, regardless of status (including pending and cancelled). |
| **Gross revenue** | `sum(total_cents) / 100` for orders with status `paid`, `shipped`, or `delivered`. |
| **Net revenue** | Gross revenue minus refunds created in the period. |
| **AOV** (average order value) | Gross revenue divided by the number of orders counted in gross revenue. |
| **Active customer** | A customer who placed an order in the last 90 days. |
| **Refund rate** | Refunded amount divided by gross revenue. |

## Worked Example

For one ET day, suppose there are 120 orders placed:

- 100 are `paid`/`shipped`/`delivered` totaling 650,000 cents
- 12 are `pending`
- 6 are `cancelled`
- 2 are `refunded`

Results:

- **Transactions:** 120 (all statuses count)
- **Gross revenue:** $6,500.00
- **AOV:** $6,500.00 / 100 = **$65.00** (not divided by 120)
- If $260.00 in refunds were created that day, **net revenue** = $6,240.00 and **refund rate** = 260 / 6,500 = **4.0%**

## Common Pitfalls

**Transactions vs. payments.** "Transactions" is an order-count metric and includes `pending` orders that never received a successful payment (these auto-cancel after 48 hours). Do not use it as a count of successful payments. For revenue-bearing orders, filter on `paid`, `shipped`, `delivered`.

**Cancelled and refunded statuses.** `cancelled` means cancelled after payment and fully refunded; `refunded` means returned after delivery. Neither status is in gross revenue. Their money movement shows up through refunds, so do not subtract these orders from gross revenue *and* subtract the refund again.

**Refund timing.** Net revenue uses refunds *created in the period*, not refunds tied to orders placed in the period. A refund created today for an order placed last month reduces today's net revenue. Because a refunded order is excluded from gross revenue by its status, the refund-rate numerator and denominator can cover different order populations. Report it as defined and do not "correct" it.

**Cents vs. dollars.** Mixing `total_cents` with dollar values is the most common error. Check that a value of `6500` is not shown as "$6,500". Round only at presentation time.

**Timezone bucketing.** An order placed at 02:00 UTC on the 15th belongs to the **14th** in ET. Use `created_at AT TIME ZONE 'America/New_York'` before truncating to a date. Daylight saving shifts the UTC offset, so never hard-code `-5` or `-4`.

**Active customer window.** The 90-day window is rolling and counts any order placed, including one later cancelled. State this explicitly if you report it alongside revenue figures, which exclude those statuses.

**Shipping and discounts.** Free standard shipping applies at $75.00 or more *after discounts*, and always for VIP. Do not infer shipping thresholds from `total_cents` without checking how discounts are stored.

**Refund handling is not a metric input to change.** Refunds are idempotent per order and reason and always go to the original payment method. Duplicate refund rows for the same order and reason indicate a data or integration bug; flag them rather than summing them.

## Reporting Checklist

1. Convert period boundaries from ET to UTC.
2. Filter statuses correctly for the metric.
3. Compute in cents; convert to USD at the end.
4. Label the period and timezone on every chart or report.
5. Cite this glossary when sharing figures outside Finance.
