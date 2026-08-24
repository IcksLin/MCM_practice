#!/usr/bin/env python3
"""用预先固定的 q=0.95 和仅开发折判优规则重建问题3嵌套外层审计。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "results" / "q3" / "问题3外层逐折分组政策_冻结.csv"
OUT = PROJECT / "results" / "q3"
EXPECTED_SHA256 = "a567ba748fe6c68e18c13203992505a2e549d4f0b220f91d1be9c09821d0e530"
DESIGN_Q = 0.95
NEAR_WEEK = 0.25


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actual = sha256(SOURCE)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"冻结外层政策表哈希不符: {actual}")
    data = pd.read_csv(SOURCE)
    data = data[(data["可求解"] == True) & data["开发可靠性阈值"].eq(DESIGN_Q)].copy()  # noqa: E712
    selected = []
    for fold, frame in data.groupby("outer_fold", sort=True):
        best = float(frame["开发平均孕周"].min())
        eligible = frame[frame["开发平均孕周"] <= best + NEAR_WEEK + 1e-12]
        winner = eligible.sort_values(["组数", "开发平均孕周"], kind="mergesort").iloc[0]
        selected.append(winner)
    result = pd.DataFrame(selected)
    keep = [
        "outer_fold", "组数", "切点", "推荐孕周", "开发平均孕周",
        "测试平均检测孕周", "测试总体达标概率", "测试审计层最低均值",
        "测试审计层最低保守下限", "测试均值安全", "测试保守下限安全",
    ]
    result[keep].to_csv(OUT / "问题3嵌套外层独立审计.csv", index=False, encoding="utf-8-sig")
    summary = {
        "schema_version": 1,
        "source_sha256": actual,
        "selection_contract": "q=0.95预先固定；每个外层折仅以开发平均孕周选最优，0.25周内取最少组；外层指标不参与选择",
        "outer_folds": int(result["outer_fold"].nunique()),
        "mean_safe_folds": int(result["测试均值安全"].sum()),
        "lcb_safe_folds": int(result["测试保守下限安全"].sum()),
        "mean_test_week": float(result["测试平均检测孕周"].mean()),
        "mean_test_readiness": float(result["测试总体达标概率"].mean()),
        "selected_group_counts": result["组数"].value_counts().sort_index().to_dict(),
        "interpretation": "用于无外层选参的模型选择能力估计；最终整数切点35与18/22固定政策仍属于开发后稳定性审计。",
    }
    (OUT / "问题3嵌套外层独立审计.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
