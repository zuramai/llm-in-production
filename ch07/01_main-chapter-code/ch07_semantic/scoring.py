"""Execution-accuracy comparison of result sets (Chapter 6's, plus two normalizations).

Dates compare by calendar day ("2026-11-02" == "2026-11-02T00:00:00-05:00"), and a fraction
may be reported as a percentage (0.0533 == 5.33). Both apply to every variant we score.
"""

from __future__ import annotations

import itertools
from datetime import datetime


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def as_day(v):
    """'2026-11-02T00:00:00-05:00' -> '2026-11-02', '2026-08' -> '2026-08-01'; else unchanged."""
    if isinstance(v, str) and len(v) == 7 and v[4] == "-" and v[:4].isdigit() \
            and v[5:].isdigit():
        return f"{v}-01"
    if isinstance(v, str) and len(v) >= 10 and v[4] == "-" and v[7] == "-":
        try:
            datetime.fromisoformat(v)
        except ValueError:
            return v
        return v[:10]
    return v


def values_equal(gold, pred) -> bool:
    x, y = _num(gold), _num(pred)
    if x is not None and y is not None:
        if abs(x - y) <= 1e-6 * max(1.0, abs(x), abs(y)) or round(x, 2) == round(y, 2):
            return True
        # A fraction reported as a percentage (0.0533 -> 5.33).
        return 0 < abs(x) < 1 and abs(100 * x - y) <= 0.01
    if (x is None) != (y is None):
        return False
    return str(as_day(gold)) == str(as_day(pred))


def _multiset_equal(gold: list[tuple], pred: list[tuple]) -> bool:
    remaining = list(pred)
    for g in gold:
        for i, p in enumerate(remaining):
            if all(values_equal(a, b) for a, b in zip(g, p)):
                del remaining[i]
                break
        else:
            return False
    return True


def rows_match(gold: list[tuple], pred: list[dict] | None) -> bool:
    """True if some choice of predicted columns reproduces the gold rows (as a multiset).

    Order and column names are ignored and extra predicted columns are allowed, as in
    Chapter 6.
    """
    if pred is None or len(gold) != len(pred):
        return False
    if not gold:
        return True
    pred_rows = [tuple(r.values()) for r in pred]
    n_gold, n_pred = len(gold[0]), len(pred_rows[0])
    if n_pred < n_gold or n_pred > 10:
        return False
    return any(_multiset_equal(gold, [tuple(r[c] for c in cols) for r in pred_rows])
               for cols in itertools.permutations(range(n_pred), n_gold))
