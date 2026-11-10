"""Run one extraction configuration over the invoice dataset and save raw outputs + metadata.

    cd ch08/01_main-chapter-code
    uv run python make_invoices.py --out data            # once
    uv run python eval/run_eval.py text-v1-luna          # one config
    uv run python eval/run_eval.py --list                # all configs

Results land in eval/results/<config>.json (every model answer, token count, cost and latency),
so the notebook re-scores them without paying again. `--limit N` runs the first N files only.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))

from ch08_eval import judge, load_run, save_run, summarize
from ch08_extract import (
    JSON_OBJECT_FORMAT,
    NAIVE_PROMPT,
    Extraction,
    extract_from_text,
    ocr_page,
    ollama_client,
    pdf_text,
    repair_messages,
)
from ch08_invoice import validate_invoice, violations_to_text
from ch08_pipeline import ModelSpec
from make_invoices import load_truth

from llmprod import chat_client

DATA = CODE / "data"
RESULTS = HERE / "results"
LUNA, HAIKU, SONNET = "openai/gpt-6-luna", "anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"
NO_THINK = {"reasoning_effort": "none"}  # qwen3 on Ollama thinks for minutes otherwise


@dataclass(frozen=True)
class Config:
    kind: str  # text | vision | ocr
    provider: str  # openrouter | ollama
    model: str
    files: str  # pdf | scan
    naive: bool = False
    max_side: int | None = None  # downscale images before sending (local models)
    llm_kwargs: dict = field(default_factory=dict)
    printed: bool = False  # copy-then-convert (ch08_normalize)
    after: str | None = None  # saved run whose answers this model must repair (escalation)


CONFIGS: dict[str, Config] = {
    # Escalation experiment: Luna fixes gemma3:4b's answers, seeing them (vs fresh = scan-vision-luna)
    "scan-escalate-context-luna": Config("vision", "openrouter", LUNA, "scan",
                                         after="scan-vision-gemma3-4b"),
    # Clean PDFs, text layer -> LLM
    "text-naive-luna": Config("text", "openrouter", LUNA, "pdf", naive=True),
    "text-naive-qwen3-8b": Config("text", "ollama", "qwen3:8b", "pdf", naive=True,
                                  llm_kwargs=NO_THINK),
    "text-v1-luna": Config("text", "openrouter", LUNA, "pdf"),
    "text-v1-haiku": Config("text", "openrouter", HAIKU, "pdf"),
    "text-v1-qwen3-8b": Config("text", "ollama", "qwen3:8b", "pdf", llm_kwargs=NO_THINK),
    # Scans -> vision LLM
    "scan-vision-luna": Config("vision", "openrouter", LUNA, "scan"),
    "scan-vision-sonnet": Config("vision", "openrouter", SONNET, "scan"),
    "scan-vision-gemma3-4b": Config("vision", "ollama", "gemma3:4b", "scan"),
    "scan-vision-gemma3-12b": Config("vision", "ollama", "gemma3:12b", "scan"),
    "scan-vision-qwen2.5vl-7b": Config("vision", "ollama", "qwen2.5vl:7b", "scan"),
    # Copy-then-convert: the model copies printed strings, Python converts them
    "scan-printed-gemma3-4b": Config("vision", "ollama", "gemma3:4b", "scan", printed=True),
    "scan-printed-qwen2.5vl-7b": Config("vision", "ollama", "qwen2.5vl:7b", "scan",
                                        printed=True),
    "scan-printed-luna": Config("vision", "openrouter", LUNA, "scan", printed=True),
    "text-printed-qwen3-8b": Config("text", "ollama", "qwen3:8b", "pdf", llm_kwargs=NO_THINK,
                                    printed=True),
    "scan-ocr-glm-printed-qwen3-8b": Config("ocr", "ollama", "qwen3:8b", "scan",
                                            llm_kwargs=NO_THINK, printed=True),
    # Resolution sweep: how small can the page get before Luna misreads it? (fewer image tokens)
    **{f"scan-vision-luna-{px}px": Config("vision", "openrouter", LUNA, "scan", max_side=px)
       for px in (1200, 900, 700, 500)},
    # Scans -> glm-ocr (local) -> text LLM. The OCR transcript is cached and shared.
    "scan-ocr-glm-luna": Config("ocr", "openrouter", LUNA, "scan"),
    "scan-ocr-glm-qwen3-8b": Config("ocr", "ollama", "qwen3:8b", "scan", llm_kwargs=NO_THINK),
}
OCR_CACHE = RESULTS / "ocr-glm-ocr.json"


def files_for(cfg: Config, truth: dict, limit: int | None,
              per_layout: int | None = None) -> list[Path]:
    """Files of the config's kind, sorted. `per_layout=N` keeps the first N of each layout
    (a small stratified subset for slow local runs)."""
    names = sorted(n for n, (meta, _) in truth.items() if meta["kind"] == cfg.files)
    if per_layout:
        seen: dict[str, int] = {}
        keep = []
        for n in names:
            lay = truth[n][0]["layout"]
            if seen.get(lay, 0) < per_layout:
                seen[lay] = seen.get(lay, 0) + 1
                keep.append(n)
        names = keep
    sub = "pdf" if cfg.files == "pdf" else "scans"
    return [DATA / sub / n for n in names][:limit]


def ocr_transcripts(paths: list[Path]) -> dict[str, Extraction]:
    """glm-ocr every page once and cache the transcripts (they are the slow part)."""
    cache = load_run(OCR_CACHE)[1] if OCR_CACHE.exists() else {}
    missing = [p for p in paths if p.name not in cache]
    if missing:
        ollama = ollama_client()
        for p in missing:
            cache[p.name] = ocr_page(ollama, p)
            print(f"  ocr {p.name}: {len(cache[p.name].raw)} chars, "
                  f"{cache[p.name].latency_s:.1f} s", flush=True)
        save_run(OCR_CACHE, {"model": "glm-ocr", "max_side": 1000,
                             "run_at": datetime.now().isoformat(timespec="seconds")}, cache)
    return cache


def run(name: str, limit: int | None = None, workers: int | None = None,
        per_layout: int | None = None, note: str = "") -> Path:
    cfg = CONFIGS[name]
    meta_truth = load_truth(DATA)
    truth = {k: inv for k, (_, inv) in meta_truth.items()}
    paths = files_for(cfg, meta_truth, limit, per_layout)
    client = ollama_client() if cfg.provider == "ollama" else chat_client(timeout=180)
    kw = dict(cfg.llm_kwargs)
    if cfg.naive:
        kw |= {"system": NAIVE_PROMPT, "response_format": JSON_OBJECT_FORMAT}
    ocr = ocr_transcripts(paths) if cfg.kind == "ocr" else {}
    after = load_run(RESULTS / f"{cfg.after}.json")[1] if cfg.after else {}

    def extract(p: Path) -> Extraction:
        if cfg.naive:
            return extract_from_text(client, cfg.model, pdf_text(p), **kw)
        spec = ModelSpec(client, cfg.model, kw, cfg.max_side, printed=cfg.printed)
        if cfg.kind == "ocr":
            first = ocr[p.name]
            second = spec.extract(spec.messages("text", p, first.raw), "ocr+llm")
            out = first + second
            out.method, out.notes["ocr_chars"] = "ocr+llm", len(first.raw)
            return out
        text = pdf_text(p) if cfg.kind == "text" else ""
        messages = spec.messages("text" if cfg.kind == "text" else "vision", p, text)
        if cfg.after:
            # Escalation WITH context: the strong model also sees the weak model's answer and
            # the rules it broke (the alternative to reading the page fresh).
            prev = after[p.name]
            broken = validate_invoice(prev.invoice) if prev.invoice else []
            messages = repair_messages(messages, prev.raw,
                                       violations_to_text(broken) or (prev.error or ""))
        return spec.extract(messages, cfg.kind)

    def one(p: Path) -> tuple[str, Extraction]:
        ex = extract(p)
        print(f"  {p.name}: {ex.latency_s:.1f} s, {ex.completion_tokens} out tok, "
              f"{ex.finish_reason}, {ex.error or 'ok'}", flush=True)
        return p.name, ex

    workers = workers or (2 if cfg.provider == "ollama" else 8)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = dict(pool.map(one, paths))
    wall = time.perf_counter() - t0
    meta = {"config": name, "kind": cfg.kind, "provider": cfg.provider, "model": cfg.model,
            "naive": cfg.naive, "printed": cfg.printed, "max_side": cfg.max_side,
            "workers": workers, "per_layout": per_layout, "note": note,
            "n": len(paths), "wall_s": round(wall, 1),
            "run_at": datetime.now().isoformat(timespec="seconds")}
    path = RESULTS / f"{name}.json"
    save_run(path, meta, out)
    s = summarize([judge(n, ex, truth[n]) for n, ex in out.items()])
    print(f"{name}: all_correct={s['all_correct']:.1f}% parse_fail={s['parse_fail']:.1f}% "
          f"caught={s['caught']:.0f} silent_wrong={s['silent_wrong']:.0f} "
          f"p50={s['p50_s']:.1f}s cost=${sum(e.cost_usd for e in out.values()):.4f} "
          f"wall={wall:.0f}s -> {path.relative_to(CODE)}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--per-layout", type=int, help="first N files of each layout only")
    ap.add_argument("--note", default="", help="free text stored in the run metadata")
    a = ap.parse_args()
    if a.list or not a.configs:
        print("\n".join(f"{k:28} {v.kind:7} {v.provider:11} {v.model}" for k, v in CONFIGS.items()))
    for c in a.configs:
        run(c, a.limit, a.workers, a.per_layout, a.note)
