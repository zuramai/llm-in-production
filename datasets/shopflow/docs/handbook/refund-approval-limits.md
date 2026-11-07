# Refund Approval Limits and Process

*Owner: Support Operations · Last reviewed: 2026-10-14*

This page explains who can approve refunds, how refunds are issued, and what to do when something looks wrong. It applies to all support agents (about 40) and team leads.

## Approval limits

| Refund amount (USD) | Approver |
|---|---|
| Up to $100.00 | Support agent |
| $100.01 to $500.00 | Team lead |
| Above $500.00 | Finance, via the `finance-approvals` channel |

- The limit applies to the **total refund amount per order**, not per request. Do not split a refund into smaller pieces to stay under a limit.
- Post the order ID, amount, reason, and ticket link in `finance-approvals` for anything above $500.00. Finance will confirm in the thread before you issue the refund.

## Standard process

1. Confirm the order status. Refunds apply to orders that are `paid`, `shipped`, or `delivered`. A `pending` order has no successful payment, so there is nothing to refund; it cancels automatically after 48 hours.
2. Verify the reason against policy (see below) and attach the ticket (category `refund` or `product_defect`).
3. Get the required approval for the amount.
4. Issue the refund from the admin tool. Set the reason code.
5. Tell the customer the refund will reach the **original payment method** within 5-7 business days.

Cancelled orders (cancelled after payment) are refunded in full, and the customer should expect the money within 5-7 business days of the cancellation. Returns are refunded within 5-7 business days after we receive and inspect the item. After a return completes, the order moves to `refunded`.

## Original payment method only

Never refund to a different card, account, or payment method, even if the customer asks or the original card has been replaced. If the original card is closed, the bank or PayPal handles routing the funds back. Tell the customer to contact their card issuer if the money doesn't appear. Do not offer workarounds.

## Idempotency

Refunds are idempotent per **order + reason**. Submitting the same order and reason again returns the existing refund rather than creating a second one.

- If a request times out, retry with the same order and reason. Do not change the reason to "force" a new refund.
- Check the order's refund history before assuming a refund failed.
- A genuinely separate refund (for example, a later damaged-item claim on the same order) needs its own distinct reason and its own approval check against the combined total.

## Partial refunds

- **Return shipping deduction:** for ordinary returns, $5.99 is deducted from the refund. For defective or wrong items, return shipping is free and nothing is deducted.
- **Damaged or defective items:** the customer is entitled to a free replacement or a full refund including shipping, if reported within 30 days of delivery with a photo.
- **Multi-item orders:** refund only the returned items. Shipping is refunded only when the whole order is returned for a defect or error on our side.
- Amounts are stored in integer cents. Enter exact values and never round.

## Not eligible

- Returns after 30 days from delivery, or used items and items not in original packaging.
- Opened beauty and grocery items, unless damaged or defective.

Escalate exceptions to a team lead rather than approving them yourself.

## Fraud flags

Pause the refund and escalate to your team lead if you see any of these:

- The customer asks for the refund to go to a different card or account.
- Repeated "delivered but missing" claims across several orders. Remember the standard process: wait 24 hours, claims within 14 days of delivery, carrier investigation.
- Multiple refunds on one account in a short period with different reasons.
- A refund request that would push the order's total refunds above what was paid.

Mark the ticket `high` priority. Do not tell the customer they are under review for fraud; say the request needs additional verification.

## Safety issues

Reports of overheating, smoke, or swelling batteries are `urgent` (1-hour first response). Tell the customer to stop using the product immediately, then process the refund or replacement as a defect.
