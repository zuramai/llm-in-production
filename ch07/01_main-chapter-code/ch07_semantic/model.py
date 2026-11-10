"""Load and validate semantic/metrics.yaml into typed objects.

Everything here is authored by humans and reviewed like code. Validation happens once, at load
time, so a typo in the YAML fails the test suite instead of a CFO's question.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import sqlglot
import yaml
from pydantic import ValidationError
from sqlglot import exp

from .query import TimeRange

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "semantic" / "metrics.yaml"
AGGREGATIONS = ("count", "count_distinct", "sum")
METRIC_TYPES = ("simple", "ratio", "derived")


class SemanticModelError(ValueError):
    """metrics.yaml is inconsistent. Raised at load time, never at question time."""


@dataclass(frozen=True)
class Entity:
    name: str
    table: str
    primary_key: str | None
    time: str | None
    time_via: str | None
    joins: dict[str, str]          # target entity -> foreign-key column on this table


@dataclass(frozen=True)
class Dimension:
    name: str                      # "customer.segment"
    entity: str
    column: str
    type: str = "string"           # string | boolean | number
    values: tuple[str, ...] = ()
    with_key: bool = False
    description: str = ""

    @property
    def alias(self) -> str:
        return self.name.replace(".", "_")


@dataclass(frozen=True)
class Measure:
    name: str
    entity: str
    agg: str
    expr: exp.Expression | None    # None means COUNT(*)
    filter: exp.Expression | None


@dataclass(frozen=True)
class Metric:
    name: str
    type: str
    description: str
    unit: str | None = None
    measure: str | None = None
    scale: float | None = None
    numerator: str | None = None
    denominator: str | None = None
    expr: exp.Expression | None = None
    default_time_range: TimeRange | None = None
    synonyms: tuple[str, ...] = ()


@dataclass
class SemanticModel:
    timezone: str
    grains: tuple[str, ...]
    entities: dict[str, Entity]
    dimensions: dict[str, Dimension]
    measures: dict[str, Measure]
    metrics: dict[str, Metric]

    # ---- graph helpers -------------------------------------------------------------------
    def join_tree(self, root: str) -> dict[str, tuple[str, str]]:
        """BFS over many-to-one joins: entity -> (parent entity, fk column on parent)."""
        parents: dict[str, tuple[str, str]] = {}
        seen, queue = {root}, deque([root])
        while queue:
            current = queue.popleft()
            for target, fk in self.entities[current].joins.items():
                if target not in seen:
                    seen.add(target)
                    parents[target] = (current, fk)
                    queue.append(target)
        return parents

    def reachable(self, root: str) -> set[str]:
        return {root, *self.join_tree(root)}

    def time_entity(self, entity: str) -> str | None:
        """The entity whose `time` column dates rows of `entity` (itself, or via a join)."""
        ent = self.entities[entity]
        if ent.time:
            return entity
        return ent.time_via

    def leaf_measures(self, metric: str) -> list[str]:
        """Measures a metric is built from, in a stable order."""
        m = self.metrics[metric]
        if m.type == "simple":
            return [m.measure]
        if m.type == "ratio":
            names = [*self.leaf_measures(m.numerator), *self.leaf_measures(m.denominator)]
        else:
            names = [n for ref in metric_refs(m.expr) for n in self.leaf_measures(ref)]
        return list(dict.fromkeys(names))

    def metric_entities(self, metric: str) -> set[str]:
        return {self.measures[n].entity for n in self.leaf_measures(metric)}

    def dimensions_for(self, metric: str) -> list[str]:
        """Dimensions every measure of `metric` can reach without fan-out."""
        common = set.intersection(*(self.reachable(e) for e in self.metric_entities(metric)))
        return [d for d, dim in self.dimensions.items() if dim.entity in common]

    def subset(self, metrics: list[str]) -> SemanticModel:
        """Same model, but only `metrics` (plus the metrics they're built from) are available."""
        keep: set[str] = set()

        def visit(name: str) -> None:
            keep.add(name)
            m = self.metrics[name]
            refs = [m.numerator, m.denominator] if m.type == "ratio" else (
                metric_refs(m.expr) if m.type == "derived" else [])
            for ref in refs:
                visit(ref)

        for name in metrics:
            visit(name)
        return SemanticModel(self.timezone, self.grains, self.entities, self.dimensions,
                             self.measures, {k: v for k, v in self.metrics.items() if k in keep})


def metric_refs(expr: exp.Expression | None) -> list[str]:
    if expr is None:
        return []
    return list(dict.fromkeys(c.name for c in expr.find_all(exp.Column)))


def _parse(sql: str, where: str) -> exp.Expression:
    try:
        return sqlglot.parse_one(sql, dialect="postgres")
    except sqlglot.errors.ParseError as e:
        raise SemanticModelError(f"{where}: cannot parse {sql!r}: {e}") from None


def _time_range(spec: dict | None, metric: str) -> TimeRange | None:
    if spec is None:
        return None
    try:
        return TimeRange(**spec)
    except ValidationError as e:
        raise SemanticModelError(f"metric {metric}: bad default_time_range: {e}") from None


def load_model(path: Path | str = DEFAULT_PATH) -> SemanticModel:
    raw = yaml.safe_load(Path(path).read_text())
    entities = {
        name: Entity(name=name, table=spec["table"], primary_key=spec.get("primary_key"),
                     time=spec.get("time"), time_via=spec.get("time_via"),
                     joins=dict(spec.get("joins") or {}))
        for name, spec in raw["entities"].items()
    }
    dimensions = {}
    for name, spec in raw["dimensions"].items():
        entity, _, _ = name.partition(".")
        dimensions[name] = Dimension(
            name=name, entity=entity, column=spec["column"], type=spec.get("type", "string"),
            values=tuple(str(v) for v in spec.get("values", ())),
            with_key=spec.get("with_key", False), description=spec.get("description", ""))
    measures = {
        name: Measure(
            name=name, entity=spec["entity"], agg=spec["agg"],
            expr=_parse(spec["expr"], f"measure {name}") if spec.get("expr") else None,
            filter=_parse(spec["filter"], f"measure {name}") if spec.get("filter") else None)
        for name, spec in raw["measures"].items()
    }
    metrics = {
        name: Metric(
            name=name, type=spec["type"], description=" ".join(spec["description"].split()),
            unit=spec.get("unit"), measure=spec.get("measure"), scale=spec.get("scale"),
            numerator=spec.get("numerator"), denominator=spec.get("denominator"),
            expr=_parse(spec["expr"], f"metric {name}") if spec.get("expr") else None,
            default_time_range=_time_range(spec.get("default_time_range"), name),
            synonyms=tuple(spec.get("synonyms", ())))
        for name, spec in raw["metrics"].items()
    }
    model = SemanticModel(timezone=raw["timezone"], grains=tuple(raw["grains"]),
                          entities=entities, dimensions=dimensions, measures=measures,
                          metrics=metrics)
    _validate(model)
    return model


def _validate(model: SemanticModel) -> None:
    tables = {e.table: name for name, e in model.entities.items()}
    for ent in model.entities.values():
        for target in ent.joins:
            if target not in model.entities:
                raise SemanticModelError(f"entity {ent.name}: joins unknown entity {target}")
        if ent.time_via and ent.time_via not in model.reachable(ent.name):
            raise SemanticModelError(f"entity {ent.name}: time_via {ent.time_via} not joinable")
    for dim in model.dimensions.values():
        if dim.entity not in model.entities:
            raise SemanticModelError(f"dimension {dim.name}: unknown entity {dim.entity}")
        if dim.with_key and not model.entities[dim.entity].primary_key:
            raise SemanticModelError(f"dimension {dim.name}: with_key needs a primary key")
    for m in model.measures.values():
        if m.entity not in model.entities:
            raise SemanticModelError(f"measure {m.name}: unknown entity {m.entity}")
        if m.agg not in AGGREGATIONS:
            raise SemanticModelError(f"measure {m.name}: agg must be one of {AGGREGATIONS}")
        if m.agg != "count" and m.expr is None:
            raise SemanticModelError(f"measure {m.name}: {m.agg} needs expr")
        for node in (m.expr, m.filter):
            for col in node.find_all(exp.Column) if node else ():
                if col.table and (col.table not in tables
                                  or tables[col.table] not in model.reachable(m.entity)):
                    raise SemanticModelError(
                        f"measure {m.name}: table {col.table} is not joinable from {m.entity}")
    for metric in model.metrics.values():
        if metric.type not in METRIC_TYPES:
            raise SemanticModelError(f"metric {metric.name}: type must be one of {METRIC_TYPES}")
        if metric.type == "simple" and metric.measure not in model.measures:
            raise SemanticModelError(f"metric {metric.name}: unknown measure {metric.measure}")
        refs = {"ratio": [metric.numerator, metric.denominator],
                "derived": metric_refs(metric.expr)}.get(metric.type, [])
        for ref in refs:
            if ref not in model.metrics:
                raise SemanticModelError(f"metric {metric.name}: unknown metric {ref}")
    for name in model.metrics:            # cycles would recurse forever at question time
        _check_acyclic(model, name, ())


def _check_acyclic(model: SemanticModel, name: str, stack: tuple[str, ...]) -> None:
    if name in stack:
        raise SemanticModelError(f"metric cycle: {' -> '.join((*stack, name))}")
    m = model.metrics[name]
    refs = [m.numerator, m.denominator] if m.type == "ratio" else metric_refs(m.expr)
    for ref in refs:
        _check_acyclic(model, ref, (*stack, name))


@cache
def default_model() -> SemanticModel:
    return load_model()
