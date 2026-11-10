import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import sqlglot
from sqlglot import exp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch07_semantic import (
    Filter,
    MetricQuery,
    OrderBy,
    QueryError,
    SemanticModelError,
    TimeRange,
    compile_query,
    default_model,
    load_model,
)
from ch07_semantic.agent import plan_schema

NOW = datetime(2026, 11, 7, 13, 30, tzinfo=UTC)          # Saturday 8:30 AM in New York


def sql(**kwargs) -> str:
    return compile_query(MetricQuery(**kwargs), now=NOW)


def tree(**kwargs) -> exp.Expression:
    return sqlglot.parse_one(sql(**kwargs), dialect="postgres")


def test_output_is_one_select_over_known_tables():
    statements = sqlglot.parse(sql(metrics=["net_revenue", "aov", "refund_rate"],
                                   dimensions=["customer.segment"],
                                   time_range=TimeRange(preset="last_month")), dialect="postgres")
    assert len(statements) == 1 and isinstance(statements[0], exp.Select)
    tables = {t.name for t in statements[0].find_all(exp.Table)}
    assert tables <= {"orders", "customers", "refunds", "m_order", "m_refund"}


def test_time_range_is_a_literal_dst_aware_window():
    # Last week ran Mon Oct 26 (EDT, -04:00) to Mon Nov 2 (EST, -05:00): 169 hours.
    text = sql(metrics=["transactions"], time_range=TimeRange(preset="last_week"))
    assert "'2026-10-26T00:00:00-04:00'" in text
    assert "'2026-11-02T00:00:00-05:00'" in text


def test_grain_buckets_in_business_time_zone_not_session():
    text = sql(metrics=["gross_revenue"], grain="day", time_range=TimeRange(preset="this_week"))
    assert "AT TIME ZONE 'America/New_York'" in text
    assert "ORDER BY day" in text


def test_metrics_on_two_entities_full_join_on_dimensions():
    t = tree(metrics=["net_revenue"], dimensions=["customer.segment"],
             time_range=TimeRange(preset="this_month"))
    join = t.find(exp.Join)
    assert join.side == "FULL" and [i.name for i in join.args["using"]] == ["customer_segment"]
    # Refunds are dated by refund date, orders by order date.
    text = t.sql("postgres")
    assert "refunds.created_at >=" in text and "orders.placed_at >=" in text


def test_metrics_on_two_entities_without_dimensions_cross_join():
    join = tree(metrics=["net_revenue"]).find(exp.Join)
    assert join.args.get("kind") == "CROSS"


def test_ratio_casts_and_guards_division_by_zero():
    text = sql(metrics=["refund_rate"])
    assert "AS DECIMAL" in text and "NULLIF(" in text


def test_dimension_that_would_fan_out_is_rejected():
    with pytest.raises(QueryError, match="product_revenue"):
        sql(metrics=["gross_revenue"], dimensions=["category.name"])


def test_filter_value_case_is_canonicalized():
    text = sql(metrics=["transactions"],
               filters=[Filter(dimension="customer.segment", op="=", values=["VIP"])])
    assert "customers.segment = 'vip'" in text


def test_unknown_filter_value_is_an_error_not_zero_rows():
    with pytest.raises(QueryError, match="allowed: consumer, business, vip"):
        sql(metrics=["transactions"],
            filters=[Filter(dimension="customer.segment", op="=", values=["gold"])])


def test_filter_values_are_literals_not_sql():
    text = sql(metrics=["transactions"],
               filters=[Filter(dimension="customer.name", op="=",
                               values=["x'; DROP TABLE orders; --"])])
    assert len(sqlglot.parse(text, dialect="postgres")) == 1
    assert "'x''; DROP TABLE orders; --'" in text


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"metrics": ["revenue"]}, "unknown metric"),
    ({"metrics": ["transactions"], "dimensions": ["order.colour"]}, "unknown dimension"),
    ({"metrics": ["units_in_stock"], "grain": "day"}, "snapshot"),
    ({"metrics": ["transactions"], "order_by": [OrderBy(field="gross_revenue")]}, "order_by"),
    ({"metrics": ["transactions", "transactions"]}, "only once"),
    ({"metrics": ["active_customers", "transactions"]}, "default period"),
    ({"metrics": ["transactions"], "filters": [Filter(dimension="order.status", op=">",
                                                      values=["paid", "shipped"])]},
     "exactly one"),
    ({"metrics": ["product_count"], "filters": [Filter(dimension="product.active", op="=",
                                                       values=["yes"])]}, "boolean"),
])
def test_invalid_queries_raise_actionable_errors(kwargs, message):
    with pytest.raises(QueryError, match=message):
        sql(**kwargs)


def test_default_time_range_applies_only_without_explicit_range():
    assert "'2026-08-10T00:00:00-04:00'" in sql(metrics=["active_customers"])   # 90 days
    assert "CAST(" not in sql(metrics=["active_customers"],
                              time_range=TimeRange(preset="all_time"))


def test_names_that_are_not_unique_group_by_key():
    t = tree(metrics=["gross_revenue"], dimensions=["customer.name"])
    assert set(t.named_selects) >= {"customer_id", "customer_name"}


def test_limit_is_capped():
    assert tree(metrics=["transactions"], dimensions=["customer.name"],
                limit=50_000).args["limit"].expression.this == "1000"


def test_subset_hides_other_metrics():
    core = default_model().subset(["aov"])
    assert set(core.metrics) == {"aov", "gross_revenue", "revenue_order_count"}
    with pytest.raises(QueryError, match="unknown metric"):
        compile_query(MetricQuery(metrics=["ticket_count"]), model=core, now=NOW)


def test_plan_schema_is_strict_everywhere():
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    schema = plan_schema(default_model())
    walk(schema)
    query = schema["properties"]["query"]["anyOf"][0]
    assert query["properties"]["metrics"]["items"]["enum"] == list(default_model().metrics)


def _write_model(tmp_path, text: str) -> Path:
    path = tmp_path / "metrics.yaml"
    path.write_text(text)
    return path


BASE = """
timezone: America/New_York
grains: [day]
entities: {order: {table: orders, primary_key: id, time: placed_at}}
dimensions: {}
measures: {orders: {entity: order, agg: count}}
"""


def test_model_rejects_unknown_measure(tmp_path):
    path = _write_model(tmp_path, BASE + "metrics: {x: {type: simple, measure: nope, "
                                         "description: d}}")
    with pytest.raises(SemanticModelError, match="unknown measure"):
        load_model(path)


def test_model_rejects_metric_cycles(tmp_path):
    path = _write_model(tmp_path, BASE + """metrics:
  a: {type: derived, expr: b + 1, description: d}
  b: {type: derived, expr: a - 1, description: d}
""")
    with pytest.raises(SemanticModelError, match="cycle"):
        load_model(path)


def test_shipped_model_loads_with_the_six_core_metrics():
    model = default_model()
    assert {"transactions", "gross_revenue", "net_revenue", "aov", "refund_rate",
            "active_customers"} <= set(model.metrics)
    assert model.timezone == "America/New_York"


def test_phrase_rows_are_formatted_in_code():
    from ch07_semantic.agent import format_rows

    rows = format_rows([{"customer_segment": "vip", "net_revenue": 25208.1, "refund_rate":
                         0.030075187969, "transactions": 1234}], default_model())
    assert rows == [{"customer_segment": "vip", "net_revenue": "$25,208.10",
                     "refund_rate": "3.01%", "transactions": "1,234"}]
