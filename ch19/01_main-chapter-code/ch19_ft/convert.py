"""Chapter 18's triage split -> the fine-tuning formats of OpenAI, Bedrock and Vertex, validated.

    cd ch19/01_main-chapter-code
    uv run python -m ch19_ft.convert            # writes formats/<provider>/{train,val}.jsonl

Every format carries the same three strings per ticket: Chapter 18's one-line system prompt, the
delimited ticket, and the gold answer as compact JSON. Only the envelope differs:

* OpenAI chat JSONL:   {"messages": [{"role": "system"|"user"|"assistant", "content": str}]}
* Bedrock (Nova):      {"schemaVersion": "bedrock-conversation-2024", "system": [{"text"}],
                        "messages": [{"role": "user"|"assistant", "content": [{"text"}]}]}
* Vertex (Gemini):     {"systemInstruction": {"parts": [{"text"}]},
                        "contents": [{"role": "user"|"model", "parts": [{"text"}]}]}

Each validator parses a row back into one canonical shape, ``(system, [(role, text), ...])``, so
the structural rules live in one place per provider and the shared rules (turn order, a final
assistant turn, a valid answer, a token budget) live in one place for all of them.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from ch19_ft import DATA, FORMATS_DIR, previous

FORMATS = ("openai", "bedrock", "vertex")
BEDROCK_SCHEMA = "bedrock-conversation-2024"

Canonical = tuple[str | None, list[tuple[str, str]]]  # (system prompt, [(user|assistant, text)])


@dataclass(frozen=True)
class Limits:
    """Dataset rules one provider enforces before it starts a job. ``None``: not documented."""

    min_examples: int
    max_examples: int | None
    max_tokens_per_example: int
    max_file_mb: float | None
    max_val_examples: int | None = None


# From the providers' docs, accessed 2026-11-08 (links in the README's References).
# OpenAI: >= 10 examples, 65,536 tokens per example (gpt-4.1 / 4.1-mini), files up to 512 MB.
# Bedrock (Nova Micro/Lite/Pro, Nova 2 Lite): 8..20,000 records (the quota counts train + val
# together), 32k context. Vertex (Gemini): no documented minimum (100+ recommended),
# 131,072 tokens per example, 1 GB file, validation up to 5,000 examples.
LIMITS: dict[str, Limits] = {
    "openai": Limits(min_examples=10, max_examples=None, max_tokens_per_example=65_536,
                     max_file_mb=512),
    "bedrock": Limits(min_examples=8, max_examples=20_000, max_tokens_per_example=32_000,
                      max_file_mb=None),
    "vertex": Limits(min_examples=1, max_examples=None, max_tokens_per_example=131_072,
                     max_file_mb=1024, max_val_examples=5_000),
}

# Nova fails the whole job if any text contains one of these (Nova user guide, data prep).
NOVA_RESERVED = ("User:", "Bot:", "Assistant:", "System:", "<image>", "<video>", "[EOS]")


@dataclass
class Problem:
    line: int  # 1-based line in the JSONL file; 0 for dataset-level problems
    code: str
    message: str


@dataclass
class Report:
    provider: str
    split: str
    n: int = 0
    errors: list[Problem] = field(default_factory=list)
    warnings: list[Problem] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)  # per valid example, o200k_base estimate
    file_bytes: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> dict[str, Any]:
        t = sorted(self.tokens)
        return {
            "provider": self.provider, "split": self.split, "examples": self.n,
            "errors": len(self.errors), "warnings": len(self.warnings),
            "tokens_total": sum(t), "tokens_mean": round(sum(t) / len(t), 1) if t else 0,
            "tokens_max": t[-1] if t else 0, "file_kb": round(self.file_bytes / 1024, 1),
        }

    def to_json(self) -> dict[str, Any]:
        return {**self.summary(), "error_list": [asdict(p) for p in self.errors],
                "warning_list": [asdict(p) for p in self.warnings]}


class FormatError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------- converters


def _parts(record: dict[str, Any]) -> tuple[str, str, str]:
    """(system, user, assistant) for one ticket: Chapter 18's training example, verbatim."""
    system, user, assistant = (m["content"] for m in previous().to_example(record)["messages"])
    return system, user, assistant


def to_openai(record: dict[str, Any]) -> dict[str, Any]:
    system, user, assistant = _parts(record)
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]}


def to_bedrock(record: dict[str, Any]) -> dict[str, Any]:
    system, user, assistant = _parts(record)
    return {"schemaVersion": BEDROCK_SCHEMA,
            "system": [{"text": system}],
            "messages": [{"role": "user", "content": [{"text": user}]},
                         {"role": "assistant", "content": [{"text": assistant}]}]}


def to_vertex(record: dict[str, Any]) -> dict[str, Any]:
    system, user, assistant = _parts(record)
    return {"systemInstruction": {"role": "system", "parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]},
                         {"role": "model", "parts": [{"text": assistant}]}]}


CONVERTERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "openai": to_openai, "bedrock": to_bedrock, "vertex": to_vertex,
}


# --------------------------------------------------------------------------- parsers


def _require_keys(row: Any, allowed: set[str], required: set[str], where: str) -> None:
    if not isinstance(row, dict):
        raise FormatError("not_object", f"{where} must be a JSON object, got {type(row).__name__}")
    unknown = set(row) - allowed
    if unknown:
        raise FormatError("unknown_key", f"{where} has unknown keys {sorted(unknown)}")
    missing = required - set(row)
    if missing:
        raise FormatError("missing_key", f"{where} is missing {sorted(missing)}")


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise FormatError("bad_text", f"{where} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise FormatError("empty_text", f"{where} is empty")
    return value


def _text_blocks(blocks: Any, where: str) -> str:
    """Bedrock ``content`` / Vertex ``parts``: a non-empty list of ``{"text": str}`` blocks."""
    if not isinstance(blocks, list) or not blocks:
        raise FormatError("bad_content", f"{where} must be a non-empty list of text blocks")
    texts = []
    for i, block in enumerate(blocks):
        _require_keys(block, {"text"}, {"text"}, f"{where}[{i}]")
        texts.append(_text(block["text"], f"{where}[{i}].text"))
    return "".join(texts)


def parse_openai(row: Any) -> Canonical:
    _require_keys(row, {"messages", "tools", "parallel_tool_calls"}, {"messages"}, "row")
    messages = row["messages"]
    if not isinstance(messages, list) or not messages:
        raise FormatError("bad_messages", "messages must be a non-empty list")
    system, turns = None, []
    for i, m in enumerate(messages):
        _require_keys(m, {"role", "content", "name", "weight"}, {"role", "content"},
                      f"messages[{i}]")
        role = m["role"]
        if role == "system":
            if i != 0:
                raise FormatError("role_order", f"messages[{i}]: system must come first")
            system = _text(m["content"], f"messages[{i}].content")
            continue
        if role not in ("user", "assistant"):
            raise FormatError("bad_role", f"messages[{i}]: unsupported role {role!r}")
        if "weight" in m and (role != "assistant" or m["weight"] not in (0, 1)):
            raise FormatError("bad_weight", f"messages[{i}]: weight must be 0 or 1, on assistant")
        turns.append((role, _text(m["content"], f"messages[{i}].content")))
    return system, turns


def parse_bedrock(row: Any) -> Canonical:
    _require_keys(row, {"schemaVersion", "system", "messages"}, {"schemaVersion", "messages"},
                  "row")
    if not isinstance(row["schemaVersion"], str) or not row["schemaVersion"]:
        raise FormatError("bad_schema", f"schemaVersion must be a string like {BEDROCK_SCHEMA!r}")
    system = None
    if "system" in row:
        system = _text_blocks(row["system"], "system")
    messages = row["messages"]
    if not isinstance(messages, list) or not messages:
        raise FormatError("bad_messages", "messages must be a non-empty list")
    turns = []
    for i, m in enumerate(messages):
        _require_keys(m, {"role", "content"}, {"role", "content"}, f"messages[{i}]")
        if m["role"] not in ("user", "assistant"):
            raise FormatError("bad_role", f"messages[{i}]: unsupported role {m['role']!r}")
        turns.append((m["role"], _text_blocks(m["content"], f"messages[{i}].content")))
    for where, text in [("system", system or ""), *turns]:
        hit = next((k for k in NOVA_RESERVED if k in text), None)
        if hit:
            raise FormatError("reserved_text", f"{where} text contains {hit!r}, reserved by Nova")
    return system, turns


def parse_vertex(row: Any) -> Canonical:
    _require_keys(row, {"systemInstruction", "contents"}, {"contents"}, "row")
    system = None
    if "systemInstruction" in row:
        si = row["systemInstruction"]
        _require_keys(si, {"role", "parts"}, {"parts"}, "systemInstruction")
        system = _text_blocks(si["parts"], "systemInstruction.parts")
    contents = row["contents"]
    if not isinstance(contents, list) or not contents:
        raise FormatError("bad_messages", "contents must be a non-empty list")
    turns = []
    for i, c in enumerate(contents):
        _require_keys(c, {"role", "parts"}, {"role", "parts"}, f"contents[{i}]")
        role = {"user": "user", "model": "assistant"}.get(c["role"])
        if role is None:
            hint = " (Vertex calls it 'model')" if c["role"] == "assistant" else ""
            raise FormatError("bad_role", f"contents[{i}]: unsupported role {c['role']!r}{hint}")
        turns.append((role, _text_blocks(c["parts"], f"contents[{i}].parts")))
    return system, turns


PARSERS: dict[str, Callable[[Any], Canonical]] = {
    "openai": parse_openai, "bedrock": parse_bedrock, "vertex": parse_vertex,
}


# --------------------------------------------------------------------------- shared checks


@cache
def _encoding() -> Any:
    import tiktoken

    return tiktoken.get_encoding("o200k_base")


def count_tokens(canonical: Canonical) -> int:
    """Tokens in one example with OpenAI's chat accounting (o200k_base, +3 per message).

    Bedrock's Nova and Vertex's Gemini use their own tokenizers, so for them this is an
    estimate; their consoles report the billed count after the job.
    """
    system, turns = canonical
    enc = _encoding()
    texts = ([system] if system else []) + [t for _, t in turns]
    return sum(3 + len(enc.encode(t)) for t in texts) + 3


def check_turns(turns: list[tuple[str, str]]) -> None:
    """User and assistant alternate, starting with the user and ending with the assistant."""
    if not turns or turns[0][0] != "user":
        raise FormatError("role_order", "the first turn after the system prompt must be user")
    for (a, _), (b, _) in itertools.pairwise(turns):
        if a == b:
            raise FormatError("role_order", f"two {a} turns in a row")
    if turns[-1][0] != "assistant":
        raise FormatError("no_assistant", "the last turn must be the assistant's answer")


def check_answer(text: str) -> None:
    """The answer we train on must itself be a valid triage (strict: no fences, no prose)."""
    p = previous()
    try:
        p.parse_triage(text, lenient=False)
    except p.TriageError as e:
        raise FormatError("bad_answer", f"assistant answer: {e}") from e


def _digest(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False).encode()).hexdigest()


def validate_rows(provider: str, lines: Iterable[str], split: str = "train") -> Report:
    """Validate JSONL lines for one provider: structure, turns, answer, tokens, limits.

    Two dataset-quality warnings don't stop a job but cost money or accuracy: ``duplicate``
    (the same example again: you pay for it every epoch) and ``conflict`` (the same prompt with
    a different answer: the model is trained on both).
    """
    limits, parse = LIMITS[provider], PARSERS[provider]
    rep = Report(provider, split)
    seen: dict[str, int] = {}
    answers: dict[str, tuple[str, int]] = {}
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        rep.n += 1
        rep.file_bytes += len(line.encode()) + (0 if line.endswith("\n") else 1)
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            rep.errors.append(Problem(lineno, "bad_json", f"not JSON: {e.msg}"))
            continue
        try:
            canonical = parse(row)
            check_turns(canonical[1])
            check_answer(canonical[1][-1][1])
            n_tok = count_tokens(canonical)
            if n_tok > limits.max_tokens_per_example:
                raise FormatError("too_long", f"{n_tok} tokens > {limits.max_tokens_per_example}")
        except FormatError as e:
            rep.errors.append(Problem(lineno, e.code, str(e)))
            continue
        rep.tokens.append(n_tok)
        key = _digest(canonical)
        prompt_key = _digest((canonical[0], canonical[1][:-1]))
        answer = canonical[1][-1][1]
        if key in seen:
            rep.warnings.append(Problem(lineno, "duplicate", f"same example as line {seen[key]}"))
        elif prompt_key in answers and answers[prompt_key][0] != answer:
            rep.warnings.append(Problem(lineno, "conflict", "same prompt as line "
                                        f"{answers[prompt_key][1]} with a different answer"))
        seen.setdefault(key, lineno)
        answers.setdefault(prompt_key, (answer, lineno))
    cap = limits.max_val_examples if split == "val" else limits.max_examples
    if rep.n < limits.min_examples:
        rep.errors.append(Problem(0, "too_few", f"{rep.n} examples < {limits.min_examples}"))
    if cap is not None and rep.n > cap:
        rep.errors.append(Problem(0, "too_many", f"{rep.n} {split} examples > {cap}"))
    if limits.max_file_mb is not None and rep.file_bytes > limits.max_file_mb * 1024 * 1024:
        rep.errors.append(Problem(0, "file_too_big", f"{rep.file_bytes} bytes"))
    return rep


def validate_file(provider: str, path: Path) -> Report:
    split = "val" if path.stem.startswith("val") else "train"
    with path.open() as f:
        return validate_rows(provider, f, split)


def leakage(train: Iterable[dict[str, Any]], test: Iterable[dict[str, Any]]) -> list[int]:
    """Ids of test tickets whose (subject, body) appears verbatim in the training data."""
    seen = {(r["subject"].strip().lower(), r["body"].strip().lower()) for r in train}
    return [r["id"] for r in test
            if (r["subject"].strip().lower(), r["body"].strip().lower()) in seen]


def dedupe(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop records whose training example (prompt and answer) equals an earlier one."""
    keep, seen = [], set()
    for r in records:
        key = _digest(to_openai(r))
        if key not in seen:
            seen.add(key)
            keep.append(r)
    return keep, len(records) - len(keep)


# --------------------------------------------------------------------------- main


def convert_all(out_dir: Path = FORMATS_DIR, data_dir: Path = DATA,
                drop_duplicates: bool = True) -> dict[str, list[Report]]:
    """Write every provider's train/val files and validate what was written."""
    p = previous()
    splits = {s: p.read_jsonl(data_dir / f"{s}.jsonl") for s in ("train", "val")}
    if drop_duplicates:
        splits = {s: dedupe(rows)[0] for s, rows in splits.items()}
    reports: dict[str, list[Report]] = {}
    for provider in FORMATS:
        reports[provider] = []
        for split, records in splits.items():
            path = out_dir / provider / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:
                for r in records:
                    f.write(json.dumps(CONVERTERS[provider](r), ensure_ascii=False) + "\n")
            reports[provider].append(validate_file(provider, path))
    return reports


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=FORMATS_DIR)
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--check", type=Path, nargs="*", default=[],
                    help="only validate these files (provider taken from the parent folder)")
    args = ap.parse_args()
    if args.check:
        reports = [validate_file(path.parent.name, path) for path in args.check]
    else:
        converted = convert_all(args.out, drop_duplicates=not args.keep_duplicates)
        reports = [r for rs in converted.values() for r in rs]
    for rep in reports:
        s = rep.summary()
        print(f"{s['provider']:8} {s['split']:5} {s['examples']:5} examples  "
              f"{s['tokens_total']:7} tokens (max {s['tokens_max']})  {s['file_kb']:7} KB  "
              f"errors={s['errors']} warnings={s['warnings']}")
        for prob in rep.errors[:5]:
            print(f"    line {prob.line}: {prob.code}: {prob.message}")
    raise SystemExit(0 if all(r.ok for r in reports) else 1)


if __name__ == "__main__":
    main()
