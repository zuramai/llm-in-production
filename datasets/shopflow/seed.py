"""Seed the ShopFlow database.

    uv run python datasets/shopflow/seed.py --reset

Invariant the book relies on: exactly ``--today-orders`` orders are placed between local
midnight (``--timezone``) and the moment the seeder runs. With the default of 100:

* 96 paid/shipped orders, each with 1 succeeded payment
* 2 cancelled orders, each with 1 succeeded payment and a full refund
* 2 pending orders, each with 2 failed payment attempts

so "how many transactions today?" is genuinely ambiguous: 100 orders, 102 payment attempts,
98 successful payments. Same ``--seed`` and same anchor date produce identical historical data.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg
from dotenv import find_dotenv, load_dotenv
from faker import Faker
from rich.console import Console
from rich.table import Table

HERE = Path(__file__).resolve().parent
DEFAULT_DB = "postgresql://shopflow:shopflow@localhost:5433/shopflow"

CATEGORIES = {
    "Electronics": (["Wireless Earbuds", "Bluetooth Speaker", "USB-C Hub", "Power Bank",
                     "Smart Plug", "Webcam"], (1999, 19999)),
    "Home & Kitchen": (["Chef Knife", "Pour-Over Kettle", "Cast Iron Skillet", "Spice Rack",
                        "Cutting Board", "French Press"], (999, 12999)),
    "Apparel": (["Merino T-Shirt", "Rain Jacket", "Hoodie", "Chino Pants", "Wool Socks",
                 "Denim Jacket"], (1499, 14999)),
    "Footwear": (["Trail Runner", "Leather Boot", "Canvas Sneaker", "Slip-On Loafer",
                  "Sandal"], (2999, 17999)),
    "Beauty": (["Face Serum", "Lip Balm", "Sunscreen SPF 50", "Hand Cream", "Shampoo Bar"],
               (599, 5999)),
    "Sports & Outdoors": (["Yoga Mat", "Water Bottle", "Camping Lantern", "Dry Bag",
                           "Resistance Bands"], (999, 9999)),
    "Toys & Games": (["Puzzle 1000pc", "Building Blocks Set", "Card Game", "Plush Fox",
                      "Kite"], (799, 6999)),
    "Books": (["Cookbook", "Field Guide", "Notebook A5", "Sketchbook", "Planner 2027"],
              (999, 4999)),
    "Office": (["Desk Lamp", "Monitor Stand", "Gel Pen Pack", "Laptop Sleeve", "Desk Mat"],
               (599, 8999)),
    "Pet Supplies": (["Dog Leash", "Cat Tree", "Pet Bed", "Chew Toy", "Food Bowl"], (599, 11999)),
    "Garden": (["Pruning Shears", "Seed Starter Kit", "Watering Can", "Garden Gloves",
                "Planter Pot"], (799, 6999)),
    "Grocery": (["Single-Origin Coffee", "Matcha Powder", "Dark Chocolate Bar", "Olive Oil",
                 "Hot Sauce"], (499, 3999)),
}
ADJECTIVES = ["Classic", "Pro", "Lite", "Eco", "Premium", "Mini", "Max", "Everyday"]
COUNTRIES = [("US", 0.70), ("CA", 0.08), ("GB", 0.07), ("DE", 0.05), ("AU", 0.05), ("ID", 0.05)]
FAKER_LOCALE = {"US": "en_US", "CA": "en_CA", "GB": "en_GB", "DE": "de_DE", "AU": "en_AU",
                "ID": "id_ID"}
WAREHOUSES = ["NJ-01", "TX-01", "CA-01"]
CARRIERS = ["UPS", "FedEx", "USPS", "DHL"]
# Relative order volume by weekday (Mon=0). Weekends are busier for a consumer shop.
WEEKDAY_WEIGHT = [0.90, 0.85, 0.90, 0.95, 1.05, 1.25, 1.20]
# Relative order volume by local hour of day.
HOUR_WEIGHT = [0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.7, 0.9, 1.0, 1.1, 1.2,
               1.4, 1.3, 1.1, 1.0, 1.0, 1.1, 1.3, 1.5, 1.6, 1.4, 1.0, 0.5]
FREE_SHIPPING_THRESHOLD_CENTS = 7500
SHIPPING_CENTS = 599


@dataclass
class Data:
    categories: list[tuple] = field(default_factory=list)
    products: list[tuple] = field(default_factory=list)
    customers: list[tuple] = field(default_factory=list)
    orders: list[tuple] = field(default_factory=list)
    order_items: list[tuple] = field(default_factory=list)
    payments: list[tuple] = field(default_factory=list)
    refunds: list[tuple] = field(default_factory=list)
    shipments: list[tuple] = field(default_factory=list)
    inventory: list[tuple] = field(default_factory=list)
    support_tickets: list[tuple] = field(default_factory=list)


COLUMNS = {
    "categories": ("id", "name"),
    "products": ("id", "sku", "name", "category_id", "price_cents", "cost_cents", "active"),
    "customers": ("id", "email", "full_name", "country", "city", "segment", "created_at"),
    "orders": ("id", "customer_id", "status", "channel", "currency", "subtotal_cents",
               "discount_cents", "shipping_cents", "total_cents", "placed_at"),
    "order_items": ("id", "order_id", "product_id", "quantity", "unit_price_cents"),
    "payments": ("id", "order_id", "provider", "status", "amount_cents", "created_at"),
    "refunds": ("id", "order_id", "amount_cents", "reason", "created_at"),
    "shipments": ("id", "order_id", "carrier", "status", "shipped_at", "delivered_at"),
    "inventory": ("product_id", "warehouse", "on_hand"),
    "support_tickets": ("id", "customer_id", "order_id", "channel", "subject", "body",
                        "created_at", "label_category", "label_priority"),
}


class Builder:
    """Builds all rows in memory. Separate RNG streams keep stages independent."""

    def __init__(self, seed: int, days: int, today_orders: int, tz: ZoneInfo, now: datetime):
        self.seed, self.days, self.today_orders, self.tz, self.now = (
            seed, days, today_orders, tz, now)
        self.today: date = now.astimezone(tz).date()
        self.midnight = datetime.combine(self.today, time(0), tzinfo=tz)
        # Last instant of yesterday (UTC): historical payments/refunds are clamped before it.
        self.cutoff = self.midnight.astimezone(UTC) - timedelta(seconds=1)
        self.d = Data()
        self.next_payment_id = 1
        self.next_refund_id = 1
        self.next_shipment_id = 1
        self.next_item_id = 1
        self.customer_orders: dict[int, list[int]] = {}

    def rng(self, stage: str) -> random.Random:
        return random.Random(f"{self.seed}-{stage}")

    # -- catalog ---------------------------------------------------------------------------
    def catalog(self) -> None:
        rng = self.rng("catalog")
        names = list(CATEGORIES)
        self.d.categories = [(i + 1, n) for i, n in enumerate(names)]
        pid = 1
        used: set[str] = set()
        while pid <= 200:
            cat_idx = (pid - 1) % len(names)
            nouns, (lo, hi) = CATEGORIES[names[cat_idx]]
            name = f"{rng.choice(ADJECTIVES)} {rng.choice(nouns)}"
            if name in used:
                name = f"{name} v{pid}"
            used.add(name)
            price = rng.randrange(lo // 100, hi // 100) * 100 + 99
            cost = int(price * rng.uniform(0.35, 0.65))
            sku = f"SF-{names[cat_idx][:3].upper()}-{pid:04d}"
            self.d.products.append((pid, sku, name, cat_idx + 1, price, cost, rng.random() > 0.05))
            for wh in WAREHOUSES:
                on_hand = 0 if rng.random() < 0.08 else rng.randint(1, 500)
                self.d.inventory.append((pid, wh, on_hand))
            pid += 1
        # Popularity weights: a few best-sellers, a long tail.
        self.product_weights = [1 / (i + 1) ** 0.8 for i in range(len(self.d.products))]
        rng.shuffle(self.product_weights)

    # -- customers -------------------------------------------------------------------------
    def customers(self) -> None:
        rng = self.rng("customers")
        fakers = {c: Faker(loc) for c, loc in FAKER_LOCALE.items()}
        for i, f in enumerate(fakers.values()):
            f.seed_instance(self.seed * 100 + i)
        countries, weights = zip(*COUNTRIES)
        earliest = self.midnight - timedelta(days=self.days + 540)
        latest = self.midnight - timedelta(days=self.days)
        span = (latest - earliest).total_seconds()
        for cid in range(1, 5001):
            country = rng.choices(countries, weights)[0]
            f = fakers[country]
            full_name = f"{f.first_name()} {f.last_name()}"
            local = "".join(ch for ch in full_name.lower().replace(" ", ".") if ch.isalnum()
                            or ch == ".")
            email = f"{local}.{cid}@example.com"
            segment = rng.choices(["consumer", "business", "vip"], [0.80, 0.15, 0.05])[0]
            created = earliest + timedelta(seconds=rng.uniform(0, span))
            self.d.customers.append((cid, email, full_name, country, f.city(), segment,
                                     created.astimezone(UTC)))
        # Repeat-purchase skew: some customers order much more often.
        self.customer_weights = [rng.paretovariate(1.5) for _ in range(5000)]

    # -- orders ----------------------------------------------------------------------------
    def _order_lines(self, rng: random.Random) -> list[tuple[int, int, int]]:
        n = rng.choices([1, 2, 3, 4], [0.55, 0.28, 0.12, 0.05])[0]
        picks = rng.choices(range(len(self.d.products)), self.product_weights, k=n)
        lines = []
        for idx in dict.fromkeys(picks):
            p = self.d.products[idx]
            lines.append((p[0], rng.choices([1, 2, 3], [0.8, 0.15, 0.05])[0], p[4]))
        return lines

    def _add_order(self, rng: random.Random, placed: datetime, status: str) -> tuple:
        oid = len(self.d.orders) + 1
        cust = rng.choices(range(5000), self.customer_weights)[0]
        cid = cust + 1
        segment = self.d.customers[cust][5]
        lines = self._order_lines(rng)
        subtotal = sum(q * price for _, q, price in lines)
        discount = round(subtotal * 0.10) if rng.random() < 0.15 else 0
        # Policy (docs/policies/shipping-policy.md): the threshold applies *after* discounts.
        free = segment == "vip" or subtotal - discount >= FREE_SHIPPING_THRESHOLD_CENTS
        shipping = 0 if free else SHIPPING_CENTS
        total = subtotal - discount + shipping
        channel = rng.choices(["web", "ios", "android"], [0.55, 0.28, 0.17])[0]
        placed_utc = placed.astimezone(UTC)
        self.d.orders.append((oid, cid, status, channel, "USD", subtotal, discount, shipping,
                              total, placed_utc))
        for pid, q, price in lines:
            self.d.order_items.append((self.next_item_id, oid, pid, q, price))
            self.next_item_id += 1
        self.customer_orders.setdefault(cid, []).append(oid)
        return oid, total, placed_utc

    def _payment(self, oid: int, status: str, amount: int, at: datetime, provider: str) -> None:
        self.d.payments.append((self.next_payment_id, oid, provider, status, amount, at))
        self.next_payment_id += 1

    def _refund(self, oid: int, amount: int, reason: str, at: datetime) -> None:
        self.d.refunds.append((self.next_refund_id, oid, amount, reason, at))
        self.next_refund_id += 1

    def _shipment(self, oid, carrier, status, shipped_at, delivered_at) -> None:
        self.d.shipments.append((self.next_shipment_id, oid, carrier, status, shipped_at,
                                 delivered_at))
        self.next_shipment_id += 1

    def _local_time(self, rng: random.Random, day: date) -> datetime:
        hour = rng.choices(range(24), HOUR_WEIGHT)[0]
        naive = datetime.combine(day, time(hour)) + timedelta(seconds=rng.uniform(0, 3599))
        return naive.replace(tzinfo=self.tz)

    def historical_orders(self) -> None:
        rng = self.rng("orders")
        base = 30000 / 180
        now = self.now
        for age in range(self.days, 0, -1):
            day = self.today - timedelta(days=age)
            n = round(base * WEEKDAY_WEIGHT[day.weekday()] * rng.uniform(0.85, 1.15))
            for _ in range(n):
                placed = self._local_time(rng, day)
                r = rng.random()
                provider = rng.choices(["stripe", "paypal"], [0.8, 0.2])[0]
                shipped_at = placed + timedelta(hours=rng.uniform(6, 48))
                delivered_at = shipped_at + timedelta(days=rng.uniform(1, 6))
                carrier = rng.choice(CARRIERS)
                if r < 0.03:
                    status = "cancelled"
                elif r < 0.07 and delivered_at < now - timedelta(days=1):
                    status = "refunded"
                elif r < 0.08 and age <= 2:
                    status = "pending"
                elif shipped_at > now:
                    status = "paid"
                elif delivered_at > now:
                    status = "shipped"
                else:
                    status = "delivered"
                oid, total, placed_utc = self._add_order(rng, placed, status)
                # Money events of historical orders never spill into today (or the future).
                pay_at = min(placed_utc + timedelta(seconds=rng.uniform(2, 90)),
                             self.cutoff - timedelta(minutes=2))
                if status == "pending":
                    for k in range(rng.randint(1, 2)):
                        self._payment(oid, "failed", total, pay_at + timedelta(minutes=k),
                                      provider)
                    continue
                if rng.random() < 0.04:
                    self._payment(oid, "failed", total, pay_at, provider)
                    pay_at += timedelta(minutes=1)
                self._payment(oid, "succeeded", total, pay_at, provider)
                if status == "cancelled":
                    self._refund(oid, total, "order cancelled by customer",
                                 min(pay_at + timedelta(hours=rng.uniform(0.2, 5)), self.cutoff))
                    continue
                if status in ("shipped", "delivered", "refunded"):
                    delivered = delivered_at.astimezone(UTC) if status != "shipped" \
                        else None
                    self._shipment(oid, carrier, "delivered" if delivered else "in_transit",
                                   shipped_at.astimezone(UTC), delivered)
                if status == "refunded":
                    reason = rng.choice(["damaged on arrival", "wrong item", "not as described",
                                         "changed mind"])
                    self._refund(oid, total, reason, min(
                        delivered_at.astimezone(UTC)
                        + timedelta(days=rng.uniform(1, 10)), self.cutoff))
        self.historical_order_count = len(self.d.orders)

    def today_split(self) -> tuple[int, int, int]:
        n = self.today_orders
        cancelled = max(1, round(n * 0.02))
        pending = max(1, round(n * 0.02))
        return n - cancelled - pending, cancelled, pending

    def todays_orders(self) -> None:
        rng = self.rng("today")
        paid, cancelled, pending = self.today_split()
        window = max((self.now - self.midnight).total_seconds() - 1, 1.0)
        offsets = sorted(rng.uniform(0, window) for _ in range(self.today_orders))
        kinds = ["paid"] * paid + ["cancelled"] * cancelled + ["pending"] * pending
        rng.shuffle(kinds)
        for kind, off in zip(kinds, offsets):
            placed = self.midnight + timedelta(seconds=off)
            provider = rng.choices(["stripe", "paypal"], [0.8, 0.2])[0]
            if kind == "paid":
                shipped_at = placed + timedelta(hours=rng.uniform(2, 8))
                status = "shipped" if shipped_at <= self.now else "paid"
            else:
                status = kind
            oid, total, placed_utc = self._add_order(rng, placed, status)
            pay_at = min(placed_utc + timedelta(seconds=rng.uniform(2, 30)), self.now)
            if kind == "pending":
                self._payment(oid, "failed", total, pay_at, provider)
                self._payment(oid, "failed", total, pay_at, provider)
                continue
            self._payment(oid, "succeeded", total, pay_at, provider)
            if kind == "cancelled":
                self._refund(oid, total, "order cancelled by customer", pay_at)
            elif status == "shipped":
                self._shipment(oid, rng.choice(CARRIERS), "in_transit",
                               shipped_at.astimezone(UTC), None)

    # -- support tickets -------------------------------------------------------------------
    def tickets(self) -> None:
        rng = self.rng("tickets")
        templates = json.loads((HERE / "ticket_templates.json").read_text())
        hard = json.loads((HERE / "tickets_hard.json").read_text())
        order_by_id = {o[0]: o for o in self.d.orders[: self.historical_order_count]}
        product_names = [p[2] for p in self.d.products]
        customers_with_orders = sorted(self.customer_orders)
        start = self.midnight - timedelta(days=self.days)
        span = (self.midnight - start).total_seconds()
        tid = 1

        def pick_order(cid: int) -> int | None:
            ids = [o for o in self.customer_orders.get(cid, []) if o in order_by_id]
            return rng.choice(ids) if ids else None

        for _ in range(1500):
            t = rng.choice(templates)
            cid = rng.choice(customers_with_orders)
            oid = pick_order(cid) if t["needs_order"] else None
            name = self.d.customers[cid - 1][2].split()[0]
            fill = {"order_id": oid or "", "product": rng.choice(product_names), "name": name,
                    "days": rng.randint(2, 21)}
            if oid:
                created = order_by_id[oid][9] + timedelta(days=rng.uniform(1, 20))
                created = min(created, self.midnight.astimezone(UTC))
            else:
                created = (start + timedelta(seconds=rng.uniform(0, span))).astimezone(
                    UTC)
            priority = t["priority"] if isinstance(t["priority"], str) else rng.choice(
                t["priority"])
            self.d.support_tickets.append((
                tid, cid, oid, rng.choice(["email", "chat", "web_form"]),
                t["subject"].format(**fill), t["body"].format(**fill), created,
                t["category"], priority))
            tid += 1
        for h in hard:
            cid = rng.choice(customers_with_orders)
            oid = pick_order(cid) if h.get("has_order") else None
            body = h["body"].replace("{order_id}", str(oid or ""))
            created = (start + timedelta(seconds=rng.uniform(0, span))).astimezone(UTC)
            self.d.support_tickets.append((
                tid, cid, oid, h["channel"], h["subject"], body, created,
                h["label_category"], h["label_priority"]))
            tid += 1

    def build(self) -> Data:
        self.catalog()
        self.customers()
        self.historical_orders()
        self.tickets()
        self.todays_orders()
        return self.d


def load(conn: psycopg.Connection, data: Data, tz_name: str) -> None:
    with conn.cursor() as cur:
        for table, cols in COLUMNS.items():
            cur.execute(f"TRUNCATE {table} CASCADE")
        for table, cols in COLUMNS.items():
            rows = getattr(data, table)
            with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN") as copy:
                for row in rows:
                    copy.write_row(row)
            if "id" in cols:
                cur.execute(f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                            f"(SELECT coalesce(max(id), 1) FROM {table}))")
        cur.execute("INSERT INTO shop_settings (key, value) VALUES ('timezone', %s), "
                    "('currency', 'USD') ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    (tz_name,))
        cur.execute("ANALYZE")


def main() -> None:
    load_dotenv(find_dotenv(usecwd=True))
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reset", action="store_true", help="apply schema.sql before seeding")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--today-orders", type=int, default=100)
    p.add_argument("--timezone", default=os.environ.get("BUSINESS_TIMEZONE", "America/New_York"))
    args = p.parse_args()
    if args.today_orders < 3:
        p.error("--today-orders must be >= 3 (one order per paid/cancelled/pending group)")

    tz = ZoneInfo(args.timezone)
    now = datetime.now(UTC).replace(microsecond=0)
    builder = Builder(args.seed, args.days, args.today_orders, tz, now)
    data = builder.build()

    db_url = os.environ.get("DATABASE_URL", DEFAULT_DB)
    with psycopg.connect(db_url, autocommit=True) as conn:
        if args.reset:
            conn.execute((HERE / "schema.sql").read_text())
        with conn.transaction():
            load(conn, data, args.timezone)
        conn.execute("SELECT set_config('TimeZone', %s, false)", (args.timezone,))
        today_count = conn.execute(
            "SELECT count(*) FROM orders WHERE placed_at >= date_trunc('day', now())"
        ).fetchone()[0]

    console = Console()
    console.print(f"Anchor (now): {now.isoformat()}  |  local: "
                  f"{now.astimezone(tz).isoformat()} ({args.timezone})")
    paid, cancelled, pending = builder.today_split()
    console.print(f"Orders placed today (local day): [bold]{today_count}[/bold]  "
                  f"(paid/shipped {paid}, cancelled {cancelled}, pending {pending})")
    table = Table(title="ShopFlow seed summary")
    table.add_column("table")
    table.add_column("rows", justify="right")
    for name in COLUMNS:
        table.add_row(name, f"{len(getattr(data, name)):,}")
    console.print(table)
    if today_count != args.today_orders:
        raise SystemExit(f"Invariant violated: {today_count} != {args.today_orders}")


if __name__ == "__main__":
    main()
