from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
G1_SUBMODULE = ROOT / "external" / "g1_teleop"

# Mirror the source-checkout bootstrap in projects/__init__.py for tests that
# intentionally import the base package before an XRT compatibility module.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(G1_SUBMODULE))
