"""Generate the 40 bot answers used to calibrate the judge. Writes calibration/answers.json.

Run once (about 40 cheap calls); the committed JSON is what the humans labelled.
    uv run python ch03/01_main-chapter-code/make_calibration_answers.py
"""

import asyncio
import json
import sys
from pathlib import Path

import yaml
from ch03_evalkit import Case, Dataset, run
from ch03_faqbot import VARIANTS, make_bot

HERE = Path(__file__).resolve().parent


async def main() -> None:
    rows = yaml.safe_load((HERE / "calibration" / "questions.yaml").read_text())
    bots = {name: make_bot(prompt) for name, prompt in VARIANTS.items()}
    by_question = {r["id"]: r for r in rows}

    def answer(case_input: str) -> str:
        qid, question = case_input.split("|", 1)
        return bots[by_question[qid]["variant"]](question)

    only = set(sys.argv[1:])  # optional: regenerate just these ids and merge into answers.json
    todo = [r for r in rows if not only or r["id"] in only]
    ds = Dataset("calibration", [Case(r["id"], f"{r['id']}|{r['question']}") for r in todo])
    results = await run(ds, answer, concurrency=8, scorer=lambda c, o: 1.0)
    out = [{**by_question[r.case.id], "answer": r.output, "error": r.error} for r in results]
    path = HERE / "calibration" / "answers.json"
    if only:
        merged = {o["id"]: o for o in json.loads(path.read_text())}
        merged.update({o["id"]: o for o in out})
        out = [merged[r["id"]] for r in rows]
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {len(out)} answers, {sum(o['error'] is not None for o in out)} errors")


if __name__ == "__main__":
    asyncio.run(main())
