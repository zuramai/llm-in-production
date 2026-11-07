# Fraud Review: Signals and Agent Actions

*Owner: Trust & Support Operations · Last reviewed: 2026-10-14*

## Purpose

This page covers how support agents spot and handle suspicious orders and accounts. A signal is a reason to look closer, not proof of fraud. Never accuse a customer, and never share which signals we check.

## Signals to watch

**Many failed payments**
- Several failed attempts on one order or account in a short window (guideline: 5 or more within an hour).
- Multiple different cards tried on the same account, or one card tried across many accounts.
- A run of failures followed by a success on a high-value Electronics order.

Failed attempts are never charged, and orders stay `pending` until a payment succeeds, then auto-cancel after 48 hours. A customer who retries a few times after a bank decline is normal. Check the Stripe or PayPal decline reason before escalating.

**Mismatched addresses**
- The shipping address is in a different country or region from the billing address, with no prior history.
- The address changes shortly after payment, especially after a `paid` order is already queued.
- Many accounts shipping to one address.
- Gift orders are common and usually legitimate. Look for other signals alongside the mismatch.

**Refund abuse**
- Repeated "wrong item" or "defective" claims, which get free return shipping, across several orders.
- Returns of used items, or items missing original packaging, marked as unused.
- Repeated "delivered but missing" claims. The carrier investigation window is 14 days from the delivery date.
- Requests to refund to a different card, account, or method.

## What agents do

1. **Check the facts.** Review the order, payment, and ticket history. You can't see customer emails through the read-only analytics role (`shopflow_ro`), so use the support console for contact details.
2. **Set the ticket.** Use category `payment`, `refund`, or `account`, with priority `high`. Use `urgent` only if an order is about to ship or there is account takeover risk.
3. **Act before shipment if time-sensitive.** Orders placed before 2 PM ET on business days ship the same day. Tell your team lead right away so the order can be held or cancelled before it ships. Cancelled paid orders are fully refunded to the original payment method in 5-7 business days.
4. **Lock the account** immediately if the customer reports unauthorized access or you see strong signs of takeover. Tell the customer it is locked and why, in neutral terms.
5. **Document.** Add an internal note listing the signals observed, order IDs, and your actions.

## Refund rules during a review

- Refunds always go to the **original payment method**. Never refund to a different card, even if the customer or a "cardholder" asks.
- Refunds are idempotent per order and reason. Do not retry a refund that appears to have failed. Check its status first.
- Approval limits: agent up to $100, team lead up to $500, and anything above $500 needs Finance via the `finance-approvals` channel.
- Don't split a large refund into smaller ones to stay under your limit.
- Safety reports (overheating, smoke, swelling batteries) are never treated as fraud first. Handle them as urgent, tell the customer to stop using the product, and review the history afterward.

## Escalation

| Situation | Escalate to |
|---|---|
| Single suspicious order or account | Team lead |
| Pattern across several accounts or addresses | Team lead, then Finance |
| Refund over $500 | Finance (`finance-approvals`) |
| Payment failures affecting many customers (over 20%) | Not fraud. Page on-call as a SEV1 |

A sudden spike in failures across many customers is more likely an outage than an attack. Check with on-call before treating it as fraud.

## Talking to customers

- Be calm, brief, and factual. Example: "We need to verify some details before we can process this order."
- Don't reveal thresholds, internal tools, or the reason a signal was triggered.
- Remind customers that failed payments are not charged and that a bank hold clears in 1-3 business days.
- For general account deletion or privacy requests, follow the standard process. Deletion happens within 30 days, except for tax records.

## Related

- Refund approval limits
- Ticket categories and SLAs
- Incident severities
