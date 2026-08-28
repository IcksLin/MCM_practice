"""兼容入口；权威B题算法位于B题/algorithms/solve_b.py。"""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "B题" / "algorithms" / "solve_b.py"), run_name="__main__")
