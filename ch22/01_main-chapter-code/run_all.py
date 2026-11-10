"""Run every Chapter 22 repro and print a pass/fail table.

    uv run python ch22/01_main-chapter-code/run_all.py                 # everything
    uv run python ch22/01_main-chapter-code/run_all.py --only A1 G1    # a few entries
    uv run python ch22/01_main-chapter-code/run_all.py --skip-hosted   # no OpenRouter calls

PASS: the fix held (and the failure reproduced). PASS (not reproduced): the fix held but the
failure didn't show up this time. FAIL: the fix didn't hold. PENDING: needs hardware we don't
have right now (see the chapter README).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ch22_common import LEDGER, Result

RESULTS = HERE / "results" / "run_all.json"


def discover() -> list:
    names = sorted(p.stem for p in (HERE / "repros").glob("[a-z][0-9]*_*.py"))
    return [importlib.import_module(f"repros.{n}") for n in names]


def run(only: set[str] | None = None, skip_hosted: bool = False) -> list[Result]:
    results = []
    for mod in discover():
        if only and mod.ID not in only:
            continue
        if skip_hosted and mod.KIND == "hosted":
            continue
        try:
            res = mod.run()
        except Exception as e:  # noqa: BLE001  a crashed repro is a FAIL row, not a crashed run
            traceback.print_exc()
            res = Result(mod.ID, mod.TITLE, mod.LAYER, mod.KIND, None, False,
                         f"repro crashed: {type(e).__name__}: {e}"[:300])
        print(f"  {res.id} {res.status} ({res.seconds:.1f} s)", file=sys.stderr, flush=True)
        results.append(res)
    return results


def table(results: list[Result]) -> str:
    """Compact pass/fail table; `evidence()` prints what each repro measured."""
    head = f"{'id':<3} {'layer':<17} {'entry':<48} {'kind':<6} {'status':<21} {'s':>5} {'$':>8}"
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(f"{r.id:<3} {r.layer:<17} {r.title[:48]:<48} {r.kind:<6} {r.status:<21} "
                     f"{r.seconds:>5.1f} {r.cost_usd:>8.5f}")
    passed = sum(r.status.startswith("PASS") for r in results)
    reproduced = sum(bool(r.reproduced) for r in results)
    lines += ["-" * len(head),
              (f"{passed}/{len(results)} fixes hold · {reproduced}/{len(results)} failures "
               f"reproduced · {LEDGER.calls} hosted calls · OpenRouter ${LEDGER.cost:.4f}")]
    return "\n".join(lines)


def evidence(results: list[Result]) -> str:
    return "\n".join(f"{r.id}: {r.evidence}" for r in results)


def save(results: list[Result], path: Path = RESULTS) -> None:
    path.parent.mkdir(exist_ok=True)
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update({r.id: r.to_dict() for r in results})
    path.write_text(json.dumps(dict(sorted(old.items())), indent=1, default=str))


def main(argv: list[str] | None = None) -> list[Result]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip-hosted", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)
    results = run(set(args.only) if args.only else None, args.skip_hosted)
    print(table(results))
    print()
    print(evidence(results))
    if not args.no_save:
        save(results)
    return results


if __name__ == "__main__":
    main()
