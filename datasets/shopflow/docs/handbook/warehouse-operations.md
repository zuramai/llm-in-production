# Warehouse Operations Overview

*Owner: Fulfillment Operations · Last reviewed: 2026-10-14*

This handbook summarizes how ShopFlow's three US warehouses operate, how orders are routed, same-day cutoffs, and how inventory is counted. It is intended for fulfillment staff, support leads, and analysts who need to understand where an order ships from.

## Warehouses

| Code | Location | Region served (default) |
|------|----------|-------------------------|
| NJ-01 | Edison, NJ | Northeast, Southeast, Canada, and international (UK, DE) |
| TX-01 | Dallas, TX | South Central, Midwest |
| CA-01 | Ontario, CA | West Coast, Mountain West, and international (AU, ID) |

All three sites stock all twelve categories: Electronics, Home & Kitchen, Apparel, Footwear, Beauty, Sports & Outdoors, Toys & Games, Books, Office, Pet Supplies, Garden, and Grocery. Stock depth varies by site. SKUs follow the pattern `SF-<CAT>-<number>`, for example `SF-ELE-0073`.

## Carriers

We ship with UPS, FedEx, USPS, and DHL. Carrier choice depends on destination and parcel profile. DHL is used mainly for international lanes. Tracking numbers must be attached to the order when it moves to **shipped**, which triggers the customer tracking email.

## Cutoffs and Same-Day Shipping

- Orders **paid before 2:00 PM ET on a business day** ship the same day.
- Orders paid after the cutoff, or on weekends and holidays, ship the next business day.
- "Business day" is always measured in **ET**, even for TX-01 and CA-01. Warehouse leads at those sites should plan pick waves around the 2 PM ET cutoff (1 PM CT, 11 AM PT).
- Orders in **pending** status (no successful payment) are never picked. They are auto-cancelled after 48 hours.

Customer-facing transit estimates (counted after shipment) are US standard 3-5 business days, US express 1-2, Canada 5-8, and UK/DE/AU/ID 7-14.

## Routing Rules

The order router chooses a fulfillment warehouse in this order of priority:

1. **Single-site fulfillment.** Prefer one warehouse that has every line item in stock.
2. **Proximity.** Among eligible sites, choose the one closest to the destination.
3. **Stock tie-break.** If sites are equally close, choose the site with the most available units of the order's lowest-stock item.
4. **Split shipment.** Only if no single site can fulfill the order. Customers receive separate tracking emails and are charged shipping once.

Express orders are routed purely by speed to destination. Fulfillment staff must not manually reroute a paid order without noting the reason on the order. Never reroute after the label has been created.

## Inventory Counts

- **Cycle counts:** Each site counts a rotating set of bins daily, so every SKU is counted at least once per quarter. High-value Electronics SKUs are counted monthly.
- **Full physical count:** Once per year per site, scheduled off-peak.
- **Discrepancies:** Variances over 2 units or $200.00 on a SKU require a recount by a second person before adjustment.
- **Adjustments:** Logged with a reason code (damaged, miscount, lost, found). Adjustments do not change order records.
- **Damaged returns:** Items returned as defective are quarantined and never restocked. Unused returns in original packaging are restocked after inspection. Opened Beauty and Grocery returns are not restocked.

## Returns Receiving

Returns are inspected on receipt. Inspection completion is what starts the 5-7 business day refund window, so inspect within one business day of arrival. Refunds always go to the original payment method. Safety-related returns (overheating, smoke, swelling batteries) must be flagged immediately to Support and quarantined in a fire-safe area.

## Escalation

- Stock or routing problems: warehouse shift lead, then Fulfillment Operations.
- Customer-impacting delays: open a ticket in the **shipping** category; set priority per SLA (urgent 1h first response, high 4h).
- Systems issues (labels failing, order sync down): follow the incident severity process. A SEV2 applies for sustained degradation, and SEV1 for checkout or payment outages.
