"""Compatibility namespace for the XRT code migrated from the monolith.

The repository intentionally keeps the historical ``projects.*`` import paths so
existing launch scripts and recorded-study tooling continue to work.  When run
directly from a checkout, make the pinned ``g1_teleop`` submodule importable without
requiring an editable install first.
"""

from __future__ import annotations

import sys
from pathlib import Path

_G1_SUBMODULE = Path(__file__).resolve().parent.parent / "external" / "g1_teleop"
if _G1_SUBMODULE.is_dir() and str(_G1_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(_G1_SUBMODULE))

