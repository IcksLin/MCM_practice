#!/usr/bin/env python3
"""问题1—3唯一轻量复现入口。

问题1重新拟合；问题2验证冻结结果与基线依赖；问题3重建无外层选参审计；随后
重绘九张证据图并生成SHA-256清单。历史大规模搜索不在默认命令中重复执行。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent

REQUIRED = [
    PROJECT / "results" / "q2" / "问题2核心方案对比.csv",
    PROJECT / "results" / "q2" / "问题2最终建议表.csv",
    PROJECT / "results" / "q2" / "参数AFT基线_冻结.json",
    PROJECT / "results" / "q3" / "问题3最终建议表_v2.csv",
    PROJECT / "results" / "q3" / "最终政策外层4折审计.csv",
]
EXPECTED = {
    "results/q2/问题2核心方案对比.csv": "8fd3ac9619acb9b25beaf53d80b4cf4bad47e058aa2ffced71dd10d5062f42cf",
    "results/q2/问题2最终建议表.csv": "40fd33ad112f618a80ded136fb061c8ea0b7bc2369544b2fa7d0592bd3297c64",
    "results/q3/问题3最终建议表_v2.csv": "da9fa655e03a69d525520c0b6b9625f89f743af7f1b90cb6edb2c2df60cb7226",
    "results/q3/最终政策外层4折审计.csv": "0dba0d5b5b97092ec55a4ace836e95a5be7cc7d1d195eb1d9fd31c166460e8cb",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(script: Path) -> None:
    subprocess.run([sys.executable, str(script)], cwd=REPO, check=True)


def main() -> None:
    missing = [str(path) for path in REQUIRED if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少冻结结果：" + "; ".join(missing))
    for relative, expected in EXPECTED.items():
        actual = sha(PROJECT / relative)
        if actual != expected:
            raise RuntimeError(f"冻结结果哈希不符 {relative}: {actual}")
    json.loads((PROJECT / "results" / "q2" / "参数AFT基线_冻结.json").read_text(encoding="utf-8"))
    run(PROJECT / "question1" / "solve_q1.py")
    run(PROJECT / "question2" / "validate_q2_policy.py")
    run(PROJECT / "question3" / "nested_q3_audit.py")
    run(PROJECT / "question123" / "plot_q123.py")
    run(PROJECT / "question123" / "build_manifest.py")
    print("q123_status=PASS")


if __name__ == "__main__":
    main()
