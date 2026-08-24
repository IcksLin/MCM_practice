"""问题2补充实验：风险权重、安全门槛与BMI切点敏感性。

不重新训练已冻结AFT；对最终候选政策执行三类决策稳健性审计：
1. 在准确性、连续等待、不稳定风险的单位单纯形上枚举权重；
2. 枚举Bootstrap、重复折、重复4/4比例门槛并记录可行政策；
3. 使用既有外层固定切点回放表，检查该表实际覆盖网格上的安全平台。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
POLICIES = ROOT / "output" / "results" / "q2" / "问题2核心方案对比.csv"
CUTS = ROOT / "output" / "results" / "q3" / "两组固定切点外层汇总.csv"
OUT = ROOT / "output" / "results" / "q2" / "sensitivity"


def main() -> None:
    policies = pd.read_csv(POLICIES).reset_index(drop=True)
    risk_rows = []
    grid = np.linspace(0, 1, 101)
    for wa in grid:
        for wt in grid:
            if wa + wt > 1 + 1e-12:
                continue
            ws = 1 - wa - wt
            score = wa * policies["准确性风险"] + wt * policies["连续时间风险"] + ws * policies["不稳定风险"]
            winner = policies.iloc[int(np.argmin(score.to_numpy(float)))]["方案"]
            risk_rows.append({"准确性权重": wa, "连续等待权重": wt, "不稳定权重": ws, "胜出方案": winner, "最小综合风险": float(score.min())})
    risk = pd.DataFrame(risk_rows)

    safety_rows = []
    for bootstrap_min in (0.95, 0.975, 0.99, 0.995, 1.0):
        for repeat_min in (0.85, 0.90, 0.95, 0.99, 1.0):
            for all4_min in (0.50, 0.70, 0.80, 0.90, 1.0):
                feasible = policies[
                    (policies["Bootstrap安全率"] >= bootstrap_min)
                    & (policies["重复折安全率_y"] >= repeat_min)
                    & (policies["重复4_4比例"] >= all4_min)
                ]
                safety_rows.append({
                    "Bootstrap门槛": bootstrap_min,
                    "重复折门槛": repeat_min,
                    "重复4_4门槛": all4_min,
                    "可行方案": "|".join(feasible["方案"]),
                    "最早可行方案": "" if feasible.empty else feasible.sort_values("患者平均孕周").iloc[0]["方案"],
                })
    safety = pd.DataFrame(safety_rows)

    cuts = pd.read_csv(CUTS)
    platform = cuts[
        cuts["最小组人数"].eq(30)
        & cuts["完成外层折数"].eq(4)
    ].copy()
    platform["均值与LCB均4折安全"] = (platform["均值安全折数"] == 4) & (platform["保守下限安全折数"] == 4)
    safe_platform = platform[platform["均值与LCB均4折安全"]]

    OUT.mkdir(parents=True, exist_ok=True)
    risk.to_csv(OUT / "风险权重单纯形.csv", index=False, encoding="utf-8-sig")
    safety.to_csv(OUT / "安全门槛敏感性.csv", index=False, encoding="utf-8-sig")
    platform.to_csv(OUT / "固定网格切点安全平台.csv", index=False, encoding="utf-8-sig")
    winner_rate = risk["胜出方案"].value_counts(normalize=True).sort_index()
    conclusion = {
        "schema_version": 1,
        "risk_grid_points": len(risk),
        "winner_rate_on_equal_grid": winner_rate.to_dict(),
        "safety_threshold_combinations": len(safety),
        "feasible_rate_by_policy": {
            policy: float(safety["可行方案"].str.split("|").map(lambda x: policy in x).mean())
            for policy in policies["方案"]
        },
        "cutpoint_evidence_scope": "两组固定切点外层汇总.csv中最小组人数30且完成4折的全部网格点；不声称为Bootstrap区间",
        "cutpoints_evaluated_min_group_30": len(platform),
        "cutpoints_safe_4_of_4": len(safe_platform),
        "safe_cutpoint_range": None if safe_platform.empty else [float(safe_platform["固定切点"].min()), float(safe_platform["固定切点"].max())],
        "algorithmic_cutpoint": 34.357,
        "deployment_rounding_35_in_safe_platform": bool(not safe_platform.empty and safe_platform["固定切点"].eq(35.0).any()),
        "interpretation": "权重网格点比例不是先验概率；用于说明结论依赖哪些风险偏好。34.357是既有算法切点；本表只证明列出的固定网格点，35不在4/4安全平台时不得据此声称安全。",
    }
    (OUT / "实验结论.json").write_text(json.dumps(conclusion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(conclusion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
