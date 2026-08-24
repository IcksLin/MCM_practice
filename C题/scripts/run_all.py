"""C题统一复现入口：默认做轻量审计，--full 执行完整训练。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_GENERAL = Path(r"E:\anaconda\python.exe")
PY_XGB = Path(r"E:\anaconda\envs\yolov_env\python.exe")


def run(python: Path, relative: str, *args: str) -> None:
    subprocess.run([str(python), str(ROOT / relative), *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="重跑耗时的分组搜索和嵌套验证")
    args = parser.parse_args()
    run(PY_GENERAL, "algorithms/q1/solve_q1.py")
    run(PY_GENERAL, "algorithms/q2/validate_q2_policy.py")
    if args.full:
        run(PY_XGB, "algorithms/q3/extended_group_search_v3.py")
        run(PY_XGB, "algorithms/q4/run_q4_full.py")
        run(PY_XGB, "algorithms/q4/patient_triage_v6.py")
        run(PY_GENERAL, "algorithms/q2/policy_sensitivity_v11.py")
        run(PY_GENERAL, "algorithms/q1/diagnostics_v2.py", "--repeats", "200")
    run(PY_GENERAL, "scripts/visualization/plot_q123.py")
    run(PY_XGB, "scripts/visualization/plot_q4.py")
    run(Path(sys.executable), "scripts/build_thoughts_pdf.py")


if __name__ == "__main__":
    main()
