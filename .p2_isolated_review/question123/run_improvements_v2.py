"""按固定顺序复现C题四项改进实验并生成哈希清单。"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("C_PROJECT_PYTHON", sys.executable)
XGB_PYTHON = os.environ.get("C_PROJECT_XGB_PYTHON", PYTHON)
COMMANDS = [
    [XGB_PYTHON, "question3/extended_group_search_v3.py"],
    [XGB_PYTHON, "question4/patient_triage_v6.py"],
    [PYTHON, "question2/policy_sensitivity_v11.py"],
    [PYTHON, "question1/diagnostics_v2.py", "--repeats", "200"],
]
OUTPUT_DIRS = (
    ROOT / "results" / "q3_improved_v3",
    ROOT / "results" / "q4_improved_v6",
    ROOT / "results" / "q2_improved_v11",
    ROOT / "results" / "q1_improved_v2",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for command in COMMANDS:
        subprocess.run(command, cwd=ROOT, check=True)
    files = []
    for directory in OUTPUT_DIRS:
        for path in sorted(directory.glob("*")):
            if path.is_file() and "smoke" not in path.name:
                files.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    scripts = []
    for command in COMMANDS:
        path = ROOT / command[1]
        scripts.append({"path": command[1], "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema_version": 1,
        "scope": "C题改进实验v2；不覆盖原实验",
        "input": {"path": "C题/附件.xlsx", "sha256": sha256(ROOT / "C题" / "附件.xlsx"), "read_only": True},
        "commands": [subprocess.list2cmdline(command) for command in COMMANDS],
        "unique_reproduction_command": "python question123/run_improvements_v2.py",
        "runtime": {"launcher_python": sys.version, "platform": platform.platform()},
        "scripts": scripts,
        "results": files,
    }
    target = ROOT / "results" / "改进实验复现清单_v2.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "result_files": len(files), "manifest": str(target)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
