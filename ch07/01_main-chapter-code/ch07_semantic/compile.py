"""Compile a MetricQuery into Postgres SQL with the sqlglot builder.

    MetricQuery --validate--> leaf measures grouped by entity --> one CTE per entity
    (joins, time range, filters, GROUP BY) --> FULL JOIN the CTEs on the dimensions -->
    metric expressions (scale, ratios, derived arithmetic) --> ORDER BY / LIMIT

Each entity is aggregated on its own before anything is joined, and dimensions are only
reachable through many-to-one joins, so no join can multiply rows (no "fan-out").
"""

from __future__ import annotations

from datetime import UTC, datetime

import sqlglot
from sqlglot import exp

from .model import Dimension, SemanticModel, default_model
from .query import Filter, MetricQuery, QueryError, TimeRange, resolve

MAX_LIMIT = 1000
_OPS = {"=": exp.EQ, "!=": exp.NEQ, ">": exp.GT, ">=": exp.GTE, "<": exp.LT, "<=": exp.LTE}


def _col(table: str, column: str) -> exp.Column:
    return exp.column(column, table=table)


def _qualify(node: exp.Expression, table: str) -> exp.Expression:
    """Copy of `node` with unqualified columns qualified by `table`."""
    node = node.copy()
    for col in node.find_all(exp.Column):
        if not col.table:
            col.set("table", exp.to_identifier(table))
    return node


def _literal(dim: Dimension, value: str) -> exp.Expression:
    if dim.type == "boolean":
        lowered = value.strip().lower()
        if lowered not in ("true", "false"):
            raise QueryError(f"{dim.name} is boolean: use 'true' or 'false', not {value!r}")
        return exp.Boolean(this=lowered == "true")
    if dim.type == "number":
        try:
            float(value)
        except ValueError:
            raise QueryError(f"{dim.name} is numeric, got {value!r}") from None
        return exp.Literal.number(value)
    if dim.values:
        # Canonicalize case ('VIP' -> 'vip'); reject values that don't exist instead of
        # silently returning zero rows.
        canonical = {v.lower(): v for v in dim.values}
        if value.lower() not in canonical:
            raise QueryError(f"{dim.name} has no value {value!r}; allowed: "
                             + ", ".join(dim.values))
        value = canonical[value.lower()]
    return exp.Literal.string(value)


class _Compiler:
    def __init__(self, q: MetricQuery, model: SemanticModel, now: datetime) -> None:
        self.q, self.model, self.now = q, model, now

    # ---- validation ----------------------------------------------------------------------
    def validate(self) -> None:
        m, q = self.model, self.q
        for name in q.metrics:
            if name not in m.metrics:
                raise QueryError(f"unknown metric {name!r}; available: {', '.join(m.metrics)}")
        if len(set(q.metrics)) != len(q.metrics):
            raise QueryError("each metric may appear only once")
        names = [*q.dimensions, *(f.dimension for f in q.filters)]
        for name in names:
            if name not in m.dimensions:
                raise QueryError(f"unknown dimension {name!r}; available: "
                                 + ", ".join(m.dimensions))
        if len(set(q.dimensions)) != len(q.dimensions):
            raise QueryError("each dimension may appear only once")
        for metric in q.metrics:
            allowed = set(m.dimensions_for(metric))
            for name in names:
                if name not in allowed:
                    supported = [x for x in m.metrics if name in m.dimensions_for(x)]
                    raise QueryError(
                        f"metric {metric} cannot be split or filtered by {name} without "
                        f"double counting; metrics that can: {', '.join(supported) or 'none'}")
        if q.grain and q.grain not in m.grains:
            raise QueryError(f"grain must be one of {', '.join(m.grains)}")
        if q.limit is not None and q.limit < 1:
            raise QueryError("limit must be a positive integer")
        outputs = set(self.output_names())
        for ob in q.order_by:
            if ob.field not in outputs and ob.field not in q.dimensions:
                raise QueryError(f"order_by {ob.field!r} must be one of the selected metrics "
                                 f"or dimensions: {', '.join(sorted(outputs))}")

    def output_names(self) -> list[str]:
        dims = [self.model.dimensions[d].alias for d in self.q.dimensions]
        return ([self.q.grain] if self.q.grain else []) + dims + list(self.q.metrics)

    def time_range_for(self, entity: str) -> TimeRange | None:
        if self.q.time_range is not None:
            return self.q.time_range
        for name in self.q.metrics:
            default = self.model.metrics[name].default_time_range
            if default is None:
                continue
            if len(self.q.metrics) > 1:
                # One CTE per entity: the default would silently apply to the other metrics.
                raise QueryError(f"metric {name} has a default period; give a time_range "
                                 "when combining it with other metrics")
            if entity in self.model.metric_entities(name):
                return default
        return None

    def time_column(self, entity: str) -> exp.Column:
        source = self.model.time_entity(entity)
        if source is None:
            raise QueryError(f"{entity} data is a snapshot with no date: drop time_range/grain")
        ent = self.model.entities[source]
        return _col(ent.table, ent.time)

    # ---- one CTE per entity --------------------------------------------------------------
    def entity_cte(self, entity: str, measures: list[str]) -> exp.Select:
        m, q, tz = self.model, self.q, self.model.timezone
        root = m.entities[entity]
        tree = m.join_tree(entity)
        needed: set[str] = set()
        group: list[tuple[exp.Expression, str]] = []

        if q.grain:
            bucket = exp.func("date_trunc", exp.Literal.string(q.grain),
                              exp.AtTimeZone(this=self.time_column(entity),
                                             zone=exp.Literal.string(tz)))
            group.append((exp.cast(bucket, "date"), q.grain))
            needed.add(m.time_entity(entity))
        for name in q.dimensions:
            dim = m.dimensions[name]
            table = m.entities[dim.entity].table
            if dim.with_key:
                key = m.entities[dim.entity].primary_key
                group.append((_col(table, key), f"{dim.entity}_id"))
            group.append((_col(table, dim.column), dim.alias))
            needed.add(dim.entity)

        conditions: list[exp.Expression] = []
        tr = self.time_range_for(entity)
        bounds = resolve(tr, self.now, tz) if tr else None
        if bounds:
            column = self.time_column(entity)
            needed.add(m.time_entity(entity))
            lo, hi = (exp.cast(exp.Literal.string(b.isoformat()), "timestamptz") for b in bounds)
            conditions += [exp.GTE(this=column.copy(), expression=lo),
                           exp.LT(this=column.copy(), expression=hi)]
        for f in q.filters:
            conditions.append(self.filter_condition(f))
            needed.add(m.dimensions[f.dimension].entity)

        selects = [e.as_(alias) for e, alias in group]
        tables_by_name = {e.table: name for name, e in m.entities.items()}
        for name in measures:
            meas = m.measures[name]
            agg = self.aggregate(meas.agg, meas.expr, root.table)
            if meas.filter is not None:
                cond = _qualify(meas.filter, root.table)
                needed |= {tables_by_name[c.table] for c in cond.find_all(exp.Column)}
                agg = exp.Filter(this=agg, expression=exp.Where(this=cond))
            selects.append(agg.as_(name))

        query = sqlglot.select(*selects).from_(exp.to_table(root.table))
        for target in self.join_order(entity, needed, tree):
            parent, fk = tree[target]
            ent, par = m.entities[target], m.entities[parent]
            on = exp.EQ(this=_col(ent.table, ent.primary_key), expression=_col(par.table, fk))
            query = query.join(exp.to_table(ent.table), on=on)
        if conditions:
            query = query.where(*conditions)
        if group:
            query = query.group_by(*(e.copy() for e, _ in group))
        return query

    @staticmethod
    def join_order(root: str, needed: set[str], tree: dict) -> list[str]:
        """Entities to join, parents before children, so every ON clause can resolve."""
        ordered: list[str] = []
        for target in sorted(needed - {root}):
            path = []
            while target != root and target not in ordered and target not in path:
                path.append(target)
                target = tree[target][0]
            ordered += reversed(path)
        return list(dict.fromkeys(ordered))

    @staticmethod
    def aggregate(agg: str, expr: exp.Expression | None, table: str) -> exp.Expression:
        if agg == "count":
            return exp.Count(this=exp.Star())
        inner = _qualify(expr, table)
        if agg == "count_distinct":
            return exp.Count(this=exp.Distinct(expressions=[inner]))
        return exp.Sum(this=inner)

    def filter_condition(self, f: Filter) -> exp.Expression:
        dim = self.model.dimensions[f.dimension]
        column = _col(self.model.entities[dim.entity].table, dim.column)
        if not f.values:
            raise QueryError(f"filter on {f.dimension} needs at least one value")
        values = [_literal(dim, v) for v in f.values]
        if f.op in ("in", "not in"):
            cond = exp.In(this=column, expressions=values)
            return exp.Not(this=cond) if f.op == "not in" else cond
        if len(values) != 1:
            raise QueryError(f"operator {f.op} takes exactly one value; use 'in' for lists")
        return _OPS[f.op](this=column, expression=values[0])

    # ---- metric expressions over the joined CTEs -----------------------------------------
    def metric_expr(self, name: str) -> exp.Expression:
        metric = self.model.metrics[name]
        if metric.type == "simple":
            value = exp.func("coalesce", exp.column(metric.measure), exp.Literal.number(0))
            if metric.scale is not None:
                value = exp.Mul(this=value, expression=exp.Literal.number(metric.scale))
            return value
        if metric.type == "ratio":
            num = exp.cast(self.metric_expr(metric.numerator), "numeric")
            den = exp.func("nullif", self.metric_expr(metric.denominator),
                           exp.Literal.number(0))
            # typed=True: we already cast to numeric. A plain exp.Div means "true division",
            # and sqlglot would add CAST(... AS DOUBLE PRECISION) on Postgres.
            return exp.Div(this=exp.paren(num), expression=den, typed=True)
        tree = metric.expr.copy()
        for col in list(tree.find_all(exp.Column)):
            col.replace(exp.paren(self.metric_expr(col.name)))
        return tree

    def build(self) -> exp.Select:
        self.validate()
        by_entity: dict[str, list[str]] = {}
        for metric in self.q.metrics:
            for meas in self.model.leaf_measures(metric):
                entity = self.model.measures[meas].entity
                if meas not in by_entity.setdefault(entity, []):
                    by_entity[entity].append(meas)
        snapshot_ok = self.q.time_range is None or self.q.time_range.preset == "all_time"
        if self.q.grain or not snapshot_ok:
            for entity in by_entity:
                self.time_column(entity)          # raises for snapshot entities

        keys = self.group_aliases()
        ctes = {f"m_{entity}": self.entity_cte(entity, ms) for entity, ms in by_entity.items()}
        names = list(ctes)
        selects: list[exp.Expression] = [exp.column(k) for k in keys]
        selects += [self.metric_expr(n).as_(n) for n in self.q.metrics]
        query = sqlglot.select(*selects).from_(names[0])
        for name in names[1:]:
            query = (query.join(name, using=keys, join_type="full outer") if keys
                     else query.join(name, join_type="cross"))
        for name, cte in ctes.items():
            query = query.with_(name, as_=cte)
        order = self.q.order_by
        if order:
            fields = {d: self.model.dimensions[d].alias for d in self.q.dimensions}
            query = query.order_by(*(
                exp.Ordered(this=exp.column(fields.get(o.field, o.field)), desc=o.descending)
                for o in order))
        elif self.q.grain:
            query = query.order_by(exp.column(self.q.grain))
        return query.limit(min(self.q.limit or MAX_LIMIT, MAX_LIMIT))

    def group_aliases(self) -> list[str]:
        aliases = [self.q.grain] if self.q.grain else []
        for name in self.q.dimensions:
            dim = self.model.dimensions[name]
            aliases += ([f"{dim.entity}_id"] if dim.with_key else []) + [dim.alias]
        return aliases


def compile_query(q: MetricQuery, *, model: SemanticModel | None = None,
                  now: datetime | None = None, pretty: bool = False) -> str:
    """Postgres SQL for `q`. Raises QueryError with a message meant for the model to fix."""
    compiler = _Compiler(q, model or default_model(), now or datetime.now(UTC))
    return compiler.build().sql(dialect="postgres", pretty=pretty)
