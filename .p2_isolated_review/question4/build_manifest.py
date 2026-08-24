"""生成问题4可复现清单；在代码、结果、图或报告变化后重新运行。"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
import numpy
import pandas
import sklearn
import xgboost

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    inputs = [ROOT / "C题" / "附件.xlsx"]
    code = sorted((ROOT / "question4").glob("*.py"))
    results = sorted(p for p in (ROOT / "results").glob("q4_*") if p.name != "复现清单.json")
    figures = sorted((ROOT / "figures").glob("*q4*"))
    reports = sorted((ROOT / "doc").glob("*问题4*.docx")) if (ROOT / "doc").exists() else []
    files = inputs + code + results + figures + reports
    manifest = {
        "task": "C题问题4女胎非整倍体判定",
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "dependencies": {"numpy": numpy.__version__, "pandas": pandas.__version__, "scikit_learn": sklearn.__version__,
                         "xgboost": xgboost.__version__, "matplotlib": matplotlib.__version__},
        "fixed_parameters": {"seed": 20250824, "outer_folds": 4, "inner_folds": 3,
                             "split_search_repeats": 2000, "bootstrap_repeats": 2000},
        "commands": [
            r"E:\anaconda\envs\yolov_env\python.exe question4\run_q4_full.py",
            r"E:\anaconda\envs\yolov_env\python.exe question4\plot_q4.py",
            r"E:\anaconda\envs\yolov_env\python.exe question4\build_manifest.py",
        ],
        "working_directory": str(ROOT),
        "files": {str(p.relative_to(ROOT)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files if p.is_file()},
    }
    out = ROOT / "results" / "复现清单.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
