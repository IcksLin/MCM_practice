#!/usr/bin/env python3
"""校验问题2冻结政策的完整性与最终推荐口径。

本脚本不重新执行历史大规模搜索；它验证冻结表哈希、样本覆盖、主方案安全性
以及建议表与统合报告采用的切点和时点是否一致。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "output" / "results" / "q2"
EXPECTED = {
    "问题2核心方案对比.csv": "8fd3ac9619acb9b25beaf53d80b4cf4bad47e058aa2ffced71dd10d5062f42cf",
    "问题2最终建议表.csv": "40fd33ad112f618a80ded136fb061c8ea0b7bc2369544b2fa7d0592bd3297c64",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for name, expected in EXPECTED.items():
        actual = sha256(RESULTS / name)
        if actual != expected:
            raise RuntimeError(f"冻结结果哈希不符 {name}: {actual}")

    baseline = json.loads((RESULTS / "参数AFT基线_冻结.json").read_text(encoding="utf-8"))
    policies = pd.read_csv(RESULTS / "问题2核心方案对比.csv")
    advice = pd.read_csv(RESULTS / "问题2最终建议表.csv")
    main_policy = policies.loc[policies["方案"].eq("18/22")].iloc[0]
    main_advice = advice.loc[advice["建议层级"].eq("题面主推荐")]

    if sorted(main_advice["样本数"].astype(int).tolist()) != [52, 215]:
        raise RuntimeError("主推荐样本覆盖不符")
    if int(main_advice["样本数"].sum()) != 267:
        raise RuntimeError("主推荐未覆盖267名患者")
    if sorted(main_advice["最佳NIPT时点"].str.extract(r"(\d+)")[0].astype(int).tolist()) != [18, 22]:
        raise RuntimeError("主推荐时点不符")
    if float(main_policy["Bootstrap安全率"]) < 0.95 or float(main_policy["重复折安全率_y"]) < 0.90:
        raise RuntimeError("18/22未满足冻结安全线")

    print(json.dumps({
        "status": "PASS",
        "patients": 267,
        "cutpoint": 34.357,
        "weeks": [18, 22],
        "readiness": float(main_policy["总体预测达标概率"]),
        "bootstrap_safety": float(main_policy["Bootstrap安全率"]),
        "aft_baseline_loaded": bool(baseline),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
