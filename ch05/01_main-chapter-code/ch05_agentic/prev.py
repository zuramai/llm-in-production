"""Load `previous_chapters.py` (Chapter 4 retrieval, Chapter 6 SQL guard) under a unique name.

Every chapter ships a file called `previous_chapters.py`, and the book's tests run in one Python
process, so a plain `import previous_chapters` would pick up whichever chapter got there first.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_NAME = "ch05_previous_chapters"
_PATH = Path(__file__).resolve().parents[1] / "previous_chapters.py"


def _load():
    if _NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(_NAME, _PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_NAME] = module
        spec.loader.exec_module(module)
    return sys.modules[_NAME]


prev = _load()
