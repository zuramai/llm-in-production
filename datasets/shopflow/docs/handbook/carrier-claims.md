# Carrier Claims Handbook

*Owner: Support Operations · Last reviewed: 2026-10-14*

This handbook explains how support agents handle carrier claims for packages that are lost in transit, marked delivered but not received, or damaged on arrival. It applies to all four carriers ShopFlow uses: UPS, FedEx, USPS and DHL.

## When to Use This Process

| Situation | Customer window | Action |
|---|---|---|
| Marked delivered, not received | Contact within 14 days of delivery date, after a 24h wait | Open carrier investigation |
| Damaged on arrival | Report within 30 days of delivery, with photo | Replacement or refund, then claim |
| Lost in transit (no scan updates) | No fixed window; check tracking first | Open carrier claim |

Do not make the customer wait for the carrier. Resolve the customer's issue first (replacement or refund), then file the claim to recover cost.

## Before You File

1. Confirm the order status is `shipped` or `delivered` and note the warehouse (NJ-01, TX-01 or CA-01) and carrier.
2. For "delivered but missing", confirm at least 24 hours have passed since the delivery scan. Ask the customer to check with neighbors and the building's mail room.
3. Confirm the delivery address matches the order. If it doesn't, escalate to a team lead.
4. For damage, check that the customer attached a photo of the item and the packaging. Without a photo, request one before proceeding.
5. **Safety issues** (overheating, smoke, swelling batteries): tell the customer to stop using the product immediately, set the ticket to `urgent` (1h first-response SLA), and then continue with the claim.

## Filing by Carrier

Use the tracking number from the order record and attach the order ID to every claim.

- **UPS:** File through the UPS claims portal using the shipper account. Attach the customer photo, the invoice value and the tracking number.
- **FedEx:** File through the FedEx claims portal. Keep the original packaging details from the customer, as FedEx may request them.
- **USPS:** File a missing-mail search request for lost packages, or a damage claim for insured parcels. Searches can take several days to produce a result.
- **DHL:** Used mainly for international destinations. File through the DHL claims contact for the shipping account and include the customs reference if present.

For international orders (Canada, UK, DE, AU, ID), allow longer transit times (up to 14 business days for UK/DE/AU/ID) before treating a package as lost.

## Resolving the Customer's Ticket

- **Ticket category:** `shipping` for lost or missing packages, `product_defect` for damaged items.
- **Priority:** `medium` by default, `high` if the customer has waited past the expected delivery window, `urgent` for safety issues.
- **Resolution:** Offer a free replacement or a full refund including shipping. Customers don't pay return shipping for defective or damaged items.
- **Refunds:** Always go to the original payment method. Never refund to a different card. Refunds are idempotent per order and reason, so don't resubmit a refund that has already been created.

### Refund Approval Limits

| Refund amount | Approver |
|---|---|
| Up to $100 | Agent |
| Up to $500 | Team lead |
| Over $500 | Finance, via the finance-approvals channel |

Customers see the refund within 5-7 business days.

## Recording the Claim

Add an internal note to the ticket with:

- Carrier and tracking number
- Claim filing date and claim reference
- Resolution given to the customer (replacement or refund amount)
- Expected recovery amount, in dollars

Money is stored as integer cents in the database, so enter amounts in cents wherever the internal tool asks for them.

## Escalation

Escalate to a team lead if a carrier rejects a claim, the same address has repeated missing-package reports, or the order value exceeds $500. If multiple customers report the same carrier or lane failing, notify Support Operations so the pattern can be raised with the carrier.
