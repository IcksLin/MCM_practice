#!/usr/bin/env python3
"""生成问题1—3精简工程的编程阶段复现清单。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results" / "q123"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(paths: list[Path]) -> list[dict[str, object]]:
    return [
        {"path": path.relative_to(PROJECT).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(paths)
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    attachment = PROJECT / "C题" / "附件.xlsx"
    code = [
        PROJECT / "question1" / "solve_q1.py",
        PROJECT / "question2" / "validate_q2_policy.py",
        PROJECT / "question3" / "nested_q3_audit.py",
        PROJECT / "question123" / "plot_q123.py",
        PROJECT / "question123" / "build_manifest.py",
        PROJECT / "question123" / "run_q123.py",
    ]
    result_files = [p for folder in (PROJECT / "results" / "q1", PROJECT / "results" / "q2", PROJECT / "results" / "q3") for p in folder.glob("*") if p.is_file()]
    figures = [p for p in (PROJECT / "figures" / "q123").rglob("*") if p.is_file()]
    packages = {}
    for name in ("numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "matplotlib", "xgboost", "openpyxl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "schema_version": 1,
        "scope": "C题问题1—3精简工程的编程阶段，不含Word排版产物",
        "input": {"path": "C题/附件.xlsx", "sha256": digest(attachment), "read_only": True},
        "random_seeds": {"q1": "无随机抽样", "q2": 2025, "q3": 20260824},
        "commands": ["python scripts/run_all.py"],
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "packages": packages},
        "code": records(code),
        "results": records(result_files),
        "figures": records(figures),
    }
    target = OUT / "复现清单.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest={target} files={len(code) + len(result_files) + len(figures)}")


if __name__ == "__main__":
    main()
