"""Build the whole book (22 chapters + appendices) into one PDF.

    uv run python tools/build_pdf.py            # writes dist/llm-in-production.pdf

Renders each README with markdown-it, turns GitHub alerts and Mermaid blocks into styled HTML,
rewrites links (chapter links become in-PDF jumps, code links point to GitHub) and prints the
result with headless Chromium (Playwright). Mermaid is loaded from jsDelivr, so the build needs
network access. Install the browser once with `uv run playwright install chromium`.
"""

from __future__ import annotations

import argparse
import html
import re
from datetime import date
from pathlib import Path
from urllib.parse import unquote

from markdown_it import MarkdownIt
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/zuramai/llm-in-production"
MERMAID_JS = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"

PARTS = [
    ("Part I · Foundations", ["ch01", "ch02", "ch03"]),
    ("Part II · Data & Retrieval", ["ch04", "ch05", "ch06", "ch07", "ch08"]),
    ("Part III · Agents & Tools", ["ch09", "ch10", "ch11", "ch12"]),
    ("Part IV · Production Hardening", ["ch13", "ch14", "ch15"]),
    ("Part V · Models & Compute", ["ch16", "ch17", "ch18", "ch19"]),
    ("Part VI · Production at Scale", ["ch20", "ch21", "ch22"]),
    ("Appendices", ["appendix-A", "appendix-B", "appendix-C"]),
]
DOCS = [d for _, ds in PARTS for d in ds]
ALERTS = {"NOTE": "Note", "TIP": "Tip", "IMPORTANT": "Important", "WARNING": "Warning",
          "CAUTION": "Caution"}


def slug(text: str) -> str:
    """GitHub's heading anchor: lowercase, drop punctuation, spaces to hyphens."""
    text = re.sub(r"[^\w\- ]", "", text.strip().lower())
    return text.replace(" ", "-")


def doc_of(path: Path) -> str | None:
    """Book doc id for a README path inside the book, else None."""
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else None
    if rel and len(rel.parts) == 2 and rel.parts[1] == "README.md" and rel.parts[0] in DOCS:
        return rel.parts[0]
    return None


def render(doc: str) -> tuple[str, str]:
    """HTML of one chapter/appendix with ids and links rewritten for a single-file book."""
    src = ROOT / doc / "README.md"
    md = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
    seen: dict[str, int] = {}

    def heading_open(self, tokens, idx, options, env):
        text = "".join(t.content for t in (tokens[idx + 1].children or []) if t.type != "image")
        base = slug(re.sub(r"`", "", text))
        n = seen.get(base, 0)
        seen[base] = n + 1
        tokens[idx].attrSet("id", f"{doc}--{base if n == 0 else f'{base}-{n}'}")
        return md.renderer.renderToken(tokens, idx, options, env)

    def fence(self, tokens, idx, options, env):
        tok = tokens[idx]
        if tok.info.strip() == "mermaid":
            return f'<pre class="mermaid">{html.escape(tok.content)}</pre>\n'
        lang = html.escape(tok.info.strip().split(" ")[0])
        return f'<pre class="code" data-lang="{lang}"><code>{html.escape(tok.content)}</code></pre>\n'

    md.add_render_rule("heading_open", heading_open)
    md.add_render_rule("fence", fence)
    body = md.render(src.read_text(encoding="utf-8"))
    title = next((ln[2:].strip() for ln in src.read_text().splitlines() if ln.startswith("# ")), doc)

    def link(m: re.Match) -> str:
        attr, url = m.group(1), html.unescape(m.group(2))
        if re.match(r"^(https?:|mailto:|data:)", url):
            return m.group(0)
        path, _, frag = url.partition("#")
        if not path:
            new = f"#{doc}--{frag}" if attr == "href" else url
        else:
            target = (src.parent / unquote(path)).resolve()
            if attr == "src":
                new = target.as_uri()
            elif (d := doc_of(target)) is not None:
                new = f"#{d}--{frag}" if frag else f"#{d}"
            else:
                rel = target.relative_to(ROOT).as_posix() if target.is_relative_to(ROOT) else path
                new = f"{REPO_URL}/blob/main/{rel}" + (f"#{frag}" if frag else "")
        return f'{attr}="{html.escape(new, quote=True)}"'

    body = re.sub(r'(href|src)="([^"]*)"', link, body)

    def alert(m: re.Match) -> str:
        kind = m.group(1)
        return (f'<blockquote class="alert alert-{kind.lower()}"><p class="alert-title">'
                f"{ALERTS[kind]}</p><p>")

    body = re.sub(r"<blockquote>\s*<p>\[!(" + "|".join(ALERTS) + r")\]\s*", alert, body)
    return title, f'<section class="doc" id="{doc}">{body}</section>'


CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
body { font: 10.5pt/1.55 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1f2328; }
h1, h2, h3, h4 { line-height: 1.25; break-after: avoid; }
h1 { font-size: 22pt; border-bottom: 1px solid #d0d7de; padding-bottom: 6px; }
h2 { font-size: 15pt; border-bottom: 1px solid #d0d7de; padding-bottom: 4px; margin-top: 1.6em; }
h3 { font-size: 12.5pt; } h4 { font-size: 11pt; }
.doc { break-before: page; }
a { color: #0969da; text-decoration: none; }
code { font: 9pt/1.45 ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
       background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }
pre.code { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; padding: 8px 10px;
           white-space: pre-wrap; word-break: break-word; }
pre.code code { background: none; padding: 0; font-size: 8.3pt; }
pre.mermaid { text-align: center; background: none; break-inside: avoid; }
table { border-collapse: collapse; font-size: 8.8pt; margin: 8px 0; break-inside: auto; }
th, td { border: 1px solid #d0d7de; padding: 4px 7px; vertical-align: top; }
th { background: #f6f8fa; } tr { break-inside: avoid; }
img { max-width: 100%; break-inside: avoid; }
blockquote { margin: 10px 0; padding: 4px 12px; color: #59636e; border-left: 4px solid #d0d7de; }
.alert { color: #1f2328; break-inside: avoid; }
.alert-title { font-weight: 600; margin: 4px 0; }
.alert-note { border-color: #0969da; } .alert-note .alert-title { color: #0969da; }
.alert-tip { border-color: #1a7f37; } .alert-tip .alert-title { color: #1a7f37; }
.alert-important { border-color: #8250df; } .alert-important .alert-title { color: #8250df; }
.alert-warning { border-color: #9a6700; } .alert-warning .alert-title { color: #9a6700; }
.alert-caution { border-color: #cf222e; } .alert-caution .alert-title { color: #cf222e; }
.cover { height: 250mm; display: flex; flex-direction: column; justify-content: center; }
.cover h1 { font-size: 40pt; border: none; margin: 0; }
.cover .sub { font-size: 15pt; color: #59636e; margin-top: 10px; }
.cover .meta { margin-top: 60px; color: #59636e; }
.toc { break-before: page; } .toc h2 { border: none; }
.toc ol { list-style: none; padding-left: 0; } .toc li { margin: 3px 0; }
.toc .part { font-weight: 600; margin-top: 12px; }
"""


def build(out: Path) -> None:
    sections, toc = [], []
    for part, docs in PARTS:
        toc.append(f'<li class="part">{html.escape(part)}</li>')
        for doc in docs:
            title, sec = render(doc)
            sections.append(sec)
            toc.append(f'<li><a href="#{doc}">{html.escape(title)}</a></li>')
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>LLM in Production</title>
<style>{CSS}</style><script src="{MERMAID_JS}"></script></head><body>
<div class="cover"><h1>LLM in Production</h1>
<div class="sub">A hands-on book about shipping large language models in real systems</div>
<div class="meta">Edition of {date.today():%Y-%m-%d} · code, data and notebooks:
<a href="{REPO_URL}">{REPO_URL.removeprefix("https://")}</a><br>
Code: Apache-2.0 · Prose and images: CC BY-SA 4.0</div></div>
<nav class="toc"><h2>Contents</h2><ol>{"".join(toc)}</ol></nav>
{"".join(sections)}</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    html_path = out.with_suffix(".html")
    html_path.write_text(page, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        tab = browser.new_page()
        tab.goto(html_path.as_uri(), wait_until="networkidle", timeout=120_000)
        failed = tab.evaluate("""async () => {
            mermaid.initialize({startOnLoad: false, theme: 'neutral', securityLevel: 'loose'});
            const nodes = [...document.querySelectorAll('pre.mermaid')];
            let failed = 0;
            for (const n of nodes) {
                try { await mermaid.run({nodes: [n]}); }
                catch (e) { failed++; n.className = 'code'; }
            }
            return [nodes.length, failed];
        }""")
        tab.pdf(path=str(out), format="A4", print_background=True, display_header_footer=True,
                header_template="<span></span>",
                footer_template='<div style="font-size:8pt;width:100%;text-align:center;'
                                'color:#59636e"><span class="pageNumber"></span></div>',
                margin={"top": "18mm", "bottom": "20mm", "left": "16mm", "right": "16mm"})
        browser.close()
    print(f"mermaid diagrams: {failed[0]} rendered, {failed[1]} failed (shown as code)")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "llm-in-production.pdf")
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
