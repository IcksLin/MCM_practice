"""
C题 Optimal Binning 最小组人数敏感性实验
===========================================

实验方案：保持q=0.90、每日时间网格、BMI审计安全线、固定内外层折和1至4组不变，
仅比较最小开发组人数1（无实质固定下限）、15、20、30。每个设置均重新执行内层全局
切点搜索、外层测试和一标准误差简约组数选择，禁止用全样本结果选择人数下限。

第一指标：外层测试平均检测孕周；安全折数优先。
辅助指标：最终切点/人数/时间、外层最低审计概率、切点折间范围、是否产生少于15人的组。
解释边界：本实验判断固定30人约束是否影响结论，不以放宽后最早的表观结果替代验证。
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
GLOBAL_SCRIPT = ROOT / "C题" / "全局分组优化" / "global_grouping_search.py"
OUT = ROOT / "outputs" / "C题" / "全局分组优化_v1"


def load_global_module():
    spec = importlib.util.spec_from_file_location("global_grouping", GLOBAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载全局分组模块")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G = load_global_module()


def group_counts(bmi: np.ndarray, cuts: tuple[float, ...]) -> str:
    counts = np.bincount(np.digitize(bmi, cuts), minlength=len(cuts) + 1)
    return "|".join(str(int(value)) for value in counts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data, outer_probability = G.load_data_and_probabilities()
    inner_probability = G.train_inner_oof_probabilities(data)
    original_minimum = G.MIN_GROUP_SIZE
    result_rows = []
    detail_rows = []
    try:
        for minimum in (1, 15, 20, 30):
            print(f"[size-sensitivity] minimum={minimum}", flush=True)
            G.MIN_GROUP_SIZE = minimum
            outer_rows, summary, selected_groups = G.outer_nested_validation(
                data, outer_probability, inner_probability
            )
            outer_rows["最小开发组人数设置"] = minimum
            detail_rows.append(outer_rows)
            graph = G.make_graph(
                data["孕妇BMI"].to_numpy(float), outer_probability, G.Q_MAIN
            )
            solution = G.solve_graph(graph, selected_groups, G.Q_MAIN)
            if solution is None:
                raise RuntimeError(f"minimum={minimum}无最终解")
            selected_summary = summary[summary["组数"] == selected_groups].iloc[0]
            fold_selected = outer_rows[outer_rows["组数"] == selected_groups]
            cut_values = []
            for text in fold_selected["切点"].fillna("").astype(str):
                cut_values.extend(float(value) for value in text.split("|") if value)
            counts_text = group_counts(data["孕妇BMI"].to_numpy(float), solution.cuts)
            minimum_final_count = min(int(value) for value in counts_text.split("|"))
            result_rows.append({
                "最小开发组人数设置": minimum,
                "设置解释": "无实质固定下限" if minimum == 1 else f"每组至少{minimum}人",
                "外层选择组数": selected_groups,
                "外层测试平均孕周": float(selected_summary["外层测试平均孕周"]),
                "外层测试安全折数": int(selected_summary["外层测试安全折数"]),
                "平均测试最低审计概率": float(selected_summary["平均测试最低审计概率"]),
                "最终切点": "|".join(f"{v:.6f}" for v in solution.cuts),
                "最终各组人数": counts_text,
                "最终最小组人数": minimum_final_count,
                "最终推荐孕周": "|".join(f"{v:.6f}" for v in solution.times),
                "最终平均孕周": solution.average_week,
                "最终最低审计概率": solution.min_audit_readiness,
                "最终安全": solution.safe,
                "外层切点最小值": min(cut_values) if cut_values else np.nan,
                "外层切点最大值": max(cut_values) if cut_values else np.nan,
                "产生少于15人的最终组": minimum_final_count < 15,
            })
    finally:
        G.MIN_GROUP_SIZE = original_minimum

    result = pd.DataFrame(result_rows)
    details = pd.concat(detail_rows, ignore_index=True)
    result.to_csv(OUT / "最小组人数敏感性.csv", index=False, encoding="utf-8-sig")
    details.to_csv(OUT / "最小组人数敏感性_外层明细.csv", index=False, encoding="utf-8-sig")

    strict = result[result["最小开发组人数设置"] == 30].iloc[0]
    relaxed = result[result["最小开发组人数设置"] == 1].iloc[0]
    conclusion = {
        "settings": result.to_dict("records"),
        "removing_fixed_limit_changes_group_count": bool(
            relaxed["外层选择组数"] != strict["外层选择组数"]
        ),
        "removing_fixed_limit_creates_tiny_final_group": bool(
            relaxed["产生少于15人的最终组"]
        ),
        "recommendation": (
            "保留30人主约束"
            if bool(relaxed["产生少于15人的最终组"])
            else "组数结论对人数下限不敏感；保留30人用于精度保护，并报告放宽敏感性"
        ),
    }
    (OUT / "最小组人数敏感性结论.json").write_text(
        json.dumps(G.serializable(conclusion), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
