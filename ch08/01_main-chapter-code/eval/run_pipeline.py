"""Run the whole pipeline (extract -> validate -> repair -> escalate) and save every attempt.

    cd ch08/01_main-chapter-code
    uv run python eval/run_pipeline.py local-first --files scans      # slow on CPU
    uv run python eval/run_pipeline.py hosted --files all

Output: eval/results/pipeline-<config>-<files>.json. The notebook scores it against the truth.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))

from ch08_eval import extraction_from_dict, extraction_to_dict
from ch08_extract import ollama_client
from ch08_invoice import Invoice, Violation
from ch08_pipeline import Attempt, ModelSpec, Outcome, process_file

from llmprod import chat_client, get_settings

DATA = CODE / "data"
NO_THINK = {"reasoning_effort": "none"}


def configs() -> dict[str, dict[str, ModelSpec]]:
    s = get_settings()
    hosted = chat_client(timeout=180)
    ollama = ollama_client(timeout=1800)
    return {
        "hosted": {"text": ModelSpec(hosted, s.llm_model, max_side=1200),
                   "strong": ModelSpec(hosted, s.llm_model_strong)},
        "local-first": {"text": ModelSpec(ollama, "qwen3:8b", NO_THINK),
                        "vision": ModelSpec(ollama, "gemma3:4b"),
                        "strong": ModelSpec(hosted, s.llm_model)},
        "local-first-printed": {"text": ModelSpec(ollama, "qwen3:8b", NO_THINK, printed=True),
                                "vision": ModelSpec(ollama, "gemma3:4b", printed=True),
                                "strong": ModelSpec(hosted, s.llm_model)},
    }


def outcome_to_dict(o: Outcome) -> dict:
    return {"file": o.file, "method": o.method, "status": o.status,
            "invoice": o.invoice.model_dump() if o.invoice else None,
            "reasons": [[v.rule, v.message] for v in o.reasons],
            "attempts": [{"stage": a.stage, "extraction": extraction_to_dict(a.extraction),
                          "violations": [[v.rule, v.message] for v in a.violations]}
                         for a in o.attempts]}


def outcome_from_dict(d: dict) -> Outcome:
    return Outcome(
        d["file"], d["method"], d["status"],
        Invoice.model_validate(d["invoice"]) if d["invoice"] else None,
        attempts=[Attempt(a["stage"], extraction_from_dict(a["extraction"]),
                          [Violation(*v) for v in a["violations"]]) for a in d["attempts"]],
        reasons=[Violation(*v) for v in d["reasons"]])


def save_outcomes(path: Path, meta: dict, outcomes: list[Outcome]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "outcomes": [outcome_to_dict(o) for o in outcomes]},
                               indent=1, ensure_ascii=False) + "\n")


def load_outcomes(path: Path) -> tuple[dict, list[Outcome]]:
    body = json.loads(path.read_text())
    return body["meta"], [outcome_from_dict(d) for d in body["outcomes"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("config", choices=["hosted", "local-first", "local-first-printed"])
    ap.add_argument("--files", choices=["all", "pdf", "scans"], default="all")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    files = []
    if a.files in ("all", "pdf"):
        files += sorted((DATA / "pdf").glob("*.pdf"))
    if a.files in ("all", "scans"):
        files += sorted((DATA / "scans").glob("*.png"))
    cfg = configs()[a.config]

    def one(p: Path) -> Outcome:
        o = process_file(p, **cfg)
        print(f"  {o.file}: {o.status} after {'+'.join(x.stage for x in o.attempts)}, "
              f"{o.latency_s:.0f} s", flush=True)
        return o

    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.workers) as pool:
        outcomes = list(pool.map(one, files))
    meta = {"config": a.config, "files": a.files, "note": a.note,
            "models": {k: v.model for k, v in cfg.items()},
            "wall_s": round(time.perf_counter() - t0, 1),
            "run_at": datetime.now().isoformat(timespec="seconds")}
    out = HERE / "results" / f"pipeline-{a.config}-{a.files}.json"
    save_outcomes(out, meta, outcomes)
    print("saved", out.relative_to(CODE))


if __name__ == "__main__":
    main()
