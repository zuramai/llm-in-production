"""Print every wrong field of a saved run, and whether the validation rules flagged it.

    uv run python eval/show_mistakes.py scan-vision-gemma3-4b
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ch08_eval import judge, load_run
from make_invoices import load_truth

FIELDS = ("supplier", "invoice_no", "date", "currency", "subtotal_cents", "tax_cents",
          "total_cents")


def mistakes(name: str) -> list[str]:
    truth = {k: inv for k, (_, inv) in load_truth(HERE.parent / "data").items()}
    _, run = load_run(HERE / "results" / f"{name}.json")
    out = []
    for f, ex in run.items():
        r = judge(f, ex, truth[f])
        if r.correct:
            continue
        flag = "caught: " + ",".join(v.rule for v in r.violations) if r.flagged else "SILENT"
        out.append(f"{f} [{flag}]")
        if ex.invoice is None:
            out.append(f"    {ex.error}")
            continue
        t, p = truth[f], ex.invoice
        out += [f"    {k}: {getattr(p, k)!r} (truth {getattr(t, k)!r})"
                for k in FIELDS if r.score[k] < 1]
        got = {(li.sku, li.qty, li.unit_price_cents) for li in p.line_items}
        want = {(li.sku, li.qty, li.unit_price_cents) for li in t.line_items}
        out += [f"    line got {g}" for g in sorted(got - want)]
        out += [f"    line missed {w}" for w in sorted(want - got)]
    return out


if __name__ == "__main__":
    for n in sys.argv[1:]:
        print(f"== {n}")
        print("\n".join(mistakes(n)))
