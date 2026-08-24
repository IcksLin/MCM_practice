"""问题4改进实验：患者级重复检测聚合与三分流策略。

实验方案与指标
--------------
基础概率来自既有患者隔离外层OOF预测。先把同一孕妇多次检测合并，再比较预先定义的
重复一致性规则。每个外折的规则只用其余三个外折的OOF患者选择，随后在留出外折评价。
判优先要求“直接阳性或复测”覆盖至少85%的异常孕妇、复测率不超过55%，再最大化
直接阳性的特异度和PPV。T21单独阳性统一进入复测，不直接宣称稳定识别。

患者级目标定义为“该孕妇是否曾在附件记录中出现筛查异常标签”（记录标签取最大值）；
它不是患者真实疾病状态，也不是临床确诊。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output" / "results" / "q4" / "q4_outer_predictions.csv"
OUT = ROOT / "output" / "results" / "q4" / "triage"


def safe_div(a: int, b: int) -> float:
    return float(a / b) if b else float("nan")


def patients(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient, frame in records.groupby("孕妇代码", sort=True):
        if frame["outer_fold"].nunique() != 1:
            raise RuntimeError(f"孕妇{patient}跨外折")
        rows.append({
            "孕妇代码": patient,
            "outer_fold": int(frame["outer_fold"].iloc[0]),
            "检测次数": len(frame),
            "y_any": int(frame["y_any"].max()),
            "标签是否跨记录一致": bool(frame["y_any"].nunique() == 1),
            "y_T21": int(frame["y_T21"].max()),
            "any阳性次数": int(frame["pred_any"].sum()),
            "T13阳性次数": int(frame["pred_T13"].sum()),
            "T18阳性次数": int(frame["pred_T18"].sum()),
            "T21阳性次数": int(frame["pred_T21"].sum()),
            "不确定次数": int(frame[["uncertain_T13", "uncertain_T18", "uncertain_T21", "uncertain_any"]].max(axis=1).sum()),
            "质量异常次数": int(frame["quality_extreme"].sum()),
            "最大any概率": float(frame["prob_any"].max()),
        })
    return pd.DataFrame(rows)


def assign(frame: pd.DataFrame, rule: str) -> np.ndarray:
    n = frame["检测次数"].to_numpy(int)
    pos = frame["any阳性次数"].to_numpy(int)
    t21_only = (
        (frame["T21阳性次数"].to_numpy(int) > 0)
        & (frame["T13阳性次数"].to_numpy(int) == 0)
        & (frame["T18阳性次数"].to_numpy(int) == 0)
    )
    thresholds = {
        "p30_r15": (0.30, 0.15),
        "p40_r20": (0.40, 0.20),
        "p50_r25": (0.50, 0.25),
        "p60_r30": (0.60, 0.30),
    }
    if rule not in thresholds:
        raise ValueError(rule)
    positive_threshold, retest_threshold = thresholds[rule]
    risk = frame["最大any概率"].to_numpy(float)
    # 概率达到阳性阈值，且重复检测时至少一半记录超过原OOF判定线。
    direct = (risk >= positive_threshold) & ((n == 1) | (pos / n >= 0.5))
    # 稀有T21单独信号降级为复测；质量异常、阈值不确定和未形成一致性的阳性也复测。
    retest = (
        t21_only
        | (frame["质量异常次数"].to_numpy(int) > 0)
        | ((frame["不确定次数"].to_numpy(int) > 0) & (risk >= retest_threshold))
        | ((risk >= retest_threshold) & ~direct)
    )
    direct &= ~t21_only
    return np.where(direct, "直接阳性", np.where(retest, "复测", "筛查阴性"))


def metrics(frame: pd.DataFrame, action: np.ndarray) -> dict[str, float]:
    y = frame["y_any"].to_numpy(int)
    direct = action == "直接阳性"
    retest = action == "复测"
    covered = direct | retest
    tp, fp = int(np.sum(direct & (y == 1))), int(np.sum(direct & (y == 0)))
    tn, fn = int(np.sum(~direct & (y == 0))), int(np.sum(~direct & (y == 1)))
    return {
        "患者数": len(frame),
        "异常患者数": int(y.sum()),
        "直接阳性数": int(direct.sum()),
        "复测数": int(retest.sum()),
        "阴性数": int((action == "筛查阴性").sum()),
        "直接阳性灵敏度": safe_div(tp, tp + fn),
        "直接阳性特异度": safe_div(tn, tn + fp),
        "直接阳性PPV": safe_div(tp, tp + fp),
        "三分流覆盖灵敏度": safe_div(int(np.sum(covered & (y == 1))), int(y.sum())),
        "复测率": float(retest.mean()),
        "阴性漏诊数": int(np.sum((action == "筛查阴性") & (y == 1))),
    }


def choose(development: pd.DataFrame, rules: tuple[str, ...]) -> str:
    candidates = []
    for rank, rule in enumerate(rules):
        result = metrics(development, assign(development, rule))
        feasible = result["三分流覆盖灵敏度"] >= 0.85 and result["复测率"] <= 0.55
        candidates.append((feasible, result["直接阳性特异度"], result["直接阳性PPV"], -result["复测率"], -rank, rule))
    feasible_candidates = [item for item in candidates if item[0]]
    if not feasible_candidates:
        raise RuntimeError("开发折没有同时满足覆盖灵敏度与复测率约束的候选规则")
    return max(feasible_candidates)[-1]


def main() -> None:
    records = pd.read_csv(SOURCE)
    frame = patients(records)
    if len(frame) != 147:
        raise RuntimeError("患者聚合后应为147人")
    rules = ("p30_r15", "p40_r20", "p50_r25", "p60_r30")
    all_rows, selected_rows, actions = [], [], []
    for outer in sorted(frame["outer_fold"].unique()):
        development = frame[frame["outer_fold"] != outer]
        test = frame[frame["outer_fold"] == outer].copy()
        selected = choose(development, rules)
        for rule in rules:
            audit = metrics(development, assign(development, rule))
            all_rows.append({
                "outer_fold": outer,
                "数据层": "development",
                "规则": rule,
                "满足可行约束": bool(audit["三分流覆盖灵敏度"] >= 0.85 and audit["复测率"] <= 0.55),
                **audit,
            })
        action = assign(test, selected)
        selected_rows.append({"outer_fold": outer, "选择规则": selected, **metrics(test, action)})
        test["最终分流"] = action
        test["选择规则"] = selected
        actions.append(test)
    selected = pd.DataFrame(selected_rows)
    patient_actions = pd.concat(actions, ignore_index=True).sort_values("孕妇代码")
    pooled = metrics(patient_actions, patient_actions["最终分流"].to_numpy())
    # T21单独信号不得直接进入阳性。
    t21_only = (patient_actions["T21阳性次数"] > 0) & (patient_actions["T13阳性次数"] == 0) & (patient_actions["T18阳性次数"] == 0)
    if patient_actions.loc[t21_only, "最终分流"].eq("直接阳性").any():
        raise RuntimeError("T21回退规则失效")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(OUT / "候选规则开发折比较.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT / "外层患者级审计.csv", index=False, encoding="utf-8-sig")
    patient_actions.to_csv(OUT / "患者级最终分流.csv", index=False, encoding="utf-8-sig")
    conclusion = {
        "schema_version": 1,
        "source": str(SOURCE.relative_to(ROOT)),
        "patients": len(frame),
        "patient_target": "同一孕妇任一记录出现附件筛查异常标签（记录y_any最大值）",
        "patients_with_inconsistent_record_labels": int((~frame["标签是否跨记录一致"]).sum()),
        "selection": "每个外折仅用其他三个外折的OOF患者选择规则",
        "feasibility_constraints": {"triage_sensitivity_min": 0.85, "retest_rate_max": 0.55},
        "selected_rules": selected["选择规则"].value_counts().to_dict(),
        "pooled_outer_results": pooled,
        "t21_policy": "T21单独阳性只进入复测，不直接判为稳定阳性",
        "interpretation": "患者级内部交叉拟合三分流；目标是附件筛查标签曾阳性，不代表真实疾病或临床确诊性能",
    }
    (OUT / "实验结论.json").write_text(json.dumps(conclusion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(conclusion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
