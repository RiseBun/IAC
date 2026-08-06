"""Make sibling script directories importable by bare module name.

Historically all scripts lived in ``IAC/tools/`` and imported each other by
bare module name (e.g. ``from train_visual_mismatch_gate_scorer import ...``).
After the split into ``pipeline/``, ``training/``, ``audit/``, ``repair/``,
``ordered_motion/`` we keep those imports unchanged and add each directory to
``sys.path`` here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("", "pipeline", "training", "audit", "repair", "ordered_motion"):
    _path = _PROJECT_ROOT if not _sub else _PROJECT_ROOT / _sub
    _s = str(_path)
    if _s not in sys.path:
        sys.path.insert(0, _s)
