"""C2: Postgres full-text search ANDs every word of a question, so real questions match nothing."""

from __future__ import annotations

from pathlib import Path

from ch22_common import Result, timed

from llmprod import connect

ID, TITLE, LAYER, KIND = "C2", "websearch_to_tsquery ANDs every word: zero hits", "Retrieval", "db"

DOCS = Path(__file__).resolve().parents[3] / "datasets/shopflow/docs"
QUESTIONS = {  # question -> the doc that answers it
    "Can I send back a cast iron skillet I already cooked with?": "care/cast-iron-skillet.md",
    "How long do refunds take to show up on my credit card?": "policies/refund-policy.md",
    "Do you ship to Canada and who pays the customs duties?": "policies/international-shipping.md",
    "My earbuds stopped charging after two months, is that covered?": "policies/warranty-policy.md",
    "How do I turn on two factor authentication for my account?": "policies/account-security.md",
    "What should on-call do when checkout errors spike after a deploy?":
        "runbooks/checkout-bad-deploy.md",
    "Can I cancel an order that already shipped?": "policies/order-cancellation.md",
    "Who can approve a refund above 500 dollars?": "handbook/refund-approval-limits.md",
}

OR_QUERY = """
SELECT doc, ts_rank_cd(tsv, q) AS rank
FROM ch22.doc_chunks,
     to_tsquery('english', array_to_string(tsvector_to_array(to_tsvector('english', %s)), ' | ')) q
WHERE tsv @@ q ORDER BY rank DESC LIMIT 3
"""
AND_QUERY = """
SELECT doc FROM ch22.doc_chunks, websearch_to_tsquery('english', %s) q
WHERE tsv @@ q ORDER BY ts_rank_cd(tsv, q) DESC LIMIT 3
"""


def chunks(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]


def setup(conn) -> int:
    conn.execute("CREATE SCHEMA IF NOT EXISTS ch22")
    conn.execute("CREATE TABLE IF NOT EXISTS ch22.doc_chunks (doc text, body text, tsv tsvector "
                 "GENERATED ALWAYS AS (to_tsvector('english', body)) STORED)")
    conn.execute("TRUNCATE ch22.doc_chunks")
    rows = [(str(p.relative_to(DOCS)), c) for p in sorted(DOCS.rglob("*.md"))
            for c in chunks(p.read_text())]
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO ch22.doc_chunks (doc, body) VALUES (%s, %s)", rows)
    return len(rows)


@timed
def run() -> Result:
    with connect() as conn:
        n = setup(conn)
        naive, fixed = {}, {}
        for q in QUESTIONS:
            naive[q] = [r[0] for r in conn.execute(AND_QUERY, (q,)).fetchall()]
            fixed[q] = [r[0] for r in conn.execute(OR_QUERY, (q,)).fetchall()]
    zero_naive = sum(not v for v in naive.values())
    zero_fixed = sum(not v for v in fixed.values())
    hit_naive = sum(QUESTIONS[q] in v for q, v in naive.items())
    hit_fixed = sum(QUESTIONS[q] in v for q, v in fixed.items())
    return Result(
        ID, TITLE, LAYER, KIND,
        reproduced=zero_naive >= len(QUESTIONS) // 2,
        fixed=zero_fixed == 0 and hit_fixed > hit_naive,
        evidence=f"{n} chunks from 36 docs, {len(QUESTIONS)} questions: websearch_to_tsquery "
                 f"zero hits {zero_naive}/{len(QUESTIONS)}, right doc in top 3 {hit_naive}; "
                 f"OR of lexemes zero hits {zero_fixed}, right doc in top 3 {hit_fixed}",
        data={"naive": naive, "fixed": fixed},
    )
