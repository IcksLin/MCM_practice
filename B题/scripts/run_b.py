from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    completed = subprocess.run([sys.executable, str(ROOT / "algorithms" / script)], cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    run("solve_b.py")
    run("robustness_b.py")
