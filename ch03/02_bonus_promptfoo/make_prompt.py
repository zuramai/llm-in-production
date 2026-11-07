"""Write prompt.json (system prompt = the Chapter 1 prompt + FAQ) for promptfoo to load."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_main-chapter-code"))
from ch03_faqbot import SYSTEM_PROMPT  # noqa: E402

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "{{question}}"},
]
Path(__file__).with_name("prompt.json").write_text(json.dumps(messages, indent=2))
print("wrote prompt.json")
