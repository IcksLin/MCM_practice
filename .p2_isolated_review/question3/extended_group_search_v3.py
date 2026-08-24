"""问题3改进实验：在全部可行组数中执行患者级嵌套分组搜索。

实验方案与指标
--------------
1. 输入固定为267名男胎孕妇及既有患者折号；外层测试折不参与模型、组数、切点和孕周选择。
2. 每个外层训练集内部使用3折OOF训练XGBoost-AFT，显式处理左/区间/右删失。
3. BMI排序后用动态规划搜索全部可行连续分组。每组至少30名孕妇，因此不硬编码1—4组；
   实际最大组数由每个开发折样本量自动计算。
4. 每段选择LCB达到0.95的最早整数周，主目标为患者加权平均推荐孕周；0.25周近优
   范围内优先选择组数更少的方案。
5. 外层报告总体达标概率、BMI四分位审计层均值及90%单侧LCB安全折数。

该实验评价的是建模流程的内部泛化能力，不是外部临床验证。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import gumbel_l


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "C题" / "附件.xlsx"
FOLDS = ROOT / "工作框架" / "男胎患者级折号.csv"
OUT = ROOT / "results" / "q3_improved_v3"
FEATURES = ["孕妇BMI", "年龄", "身高", "体重", "IVF", "孕次", "产次"]
WEEKS = np.arange(10.0, 26.0, 1.0)
MIN_GROUP_SIZE = 30
DESIGN_Q = 0.95
DEPLOY_Q = 0.90
LCB_Z = 1.645
NEAR_WEEK = 0.25
SEED = 20260825
# 由commit c4cb802中严格嵌套实验的各外折开发集内层早停确定；禁止使用全数据361轮。
OUTER_ROUNDS = (150, 64, 853, 113)


@dataclass
class Solution:
    groups: int
    cuts: tuple[float, ...]
    weeks: tuple[float, ...]
    counts: tuple[int, ...]
    average_week: float
    min_lcb: float


def historical_outer_params() -> list[dict[str, object]]:
    """从Git管理的既有AFT实验快照读取参数，不复制已清理的探索产物。"""
    path = "outputs/C题/传统机器学习重构/tree_results.json"
    text = subprocess.check_output(["git", "show", f"c4cb802:{path}"], cwd=ROOT.parent)
    return json.loads(text.decode("utf-8"))["aft"]["outer_params"]


def count_value(value: object) -> float:
    match = re.search(r"\d+", str(value))
    return float(match.group()) if match else np.nan


def load_data() -> pd.DataFrame:
    raw = pd.read_excel(INPUT, sheet_name="男胎检测数据")
    raw["IVF"] = (raw["IVF妊娠"] != "自然受孕").astype(float)
    raw["孕次"] = raw["怀孕次数"].map(count_value)
    raw["产次"] = raw["生产次数"].map(count_value)
    static = raw.groupby("孕妇代码", as_index=False)[FEATURES].median()
    folds = pd.read_csv(FOLDS)
    data = folds.merge(static, on="孕妇代码", validate="one_to_one")
    if len(data) != 267 or data[FEATURES + ["lower", "upper"]].isna().any().any():
        raise RuntimeError("患者级输入规模或必要字段异常")
    return data


def set_bounds(matrix: xgb.DMatrix, frame: pd.DataFrame) -> None:
    matrix.set_float_info("label_lower_bound", frame["lower"].to_numpy(np.float32))
    matrix.set_float_info("label_upper_bound", frame["upper"].to_numpy(np.float32))


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, params: dict[str, object], rounds: int) -> np.ndarray:
    dtrain = xgb.DMatrix(train[FEATURES], feature_names=FEATURES)
    dtest = xgb.DMatrix(test[FEATURES], feature_names=FEATURES)
    set_bounds(dtrain, train)
    model = xgb.train(params, dtrain, num_boost_round=rounds, verbose_eval=False)
    predicted_time = np.clip(model.predict(dtest), 1e-6, None)
    scale = float(params["aft_loss_distribution_scale"])
    return np.column_stack([
        gumbel_l.cdf((np.log(week) - np.log(predicted_time)) / scale) for week in WEEKS
    ])


def inner_oof(data: pd.DataFrame, outer: int, params: dict[str, object], smoke: bool) -> np.ndarray:
    outer_train = data["outer_fold"].to_numpy(int) != outer
    column = f"inner_fold_when_outer_{outer}"
    result = np.full((len(data), len(WEEKS)), np.nan)
    # 烟雾模式只减少外折数量，不降低轮数；每折轮数仅源于该外折开发集。
    rounds = OUTER_ROUNDS[outer]
    for inner in range(3):
        valid = outer_train & data[column].to_numpy(int).astype(int).__eq__(inner)
        train = outer_train & ~valid
        result[valid] = fit_predict(data.loc[train], data.loc[valid], params, rounds)
    if np.isnan(result[outer_train]).any():
        raise RuntimeError(f"外折{outer}内层OOF预测不完整")
    return result


def outer_probability(data: pd.DataFrame, outer: int, params: dict[str, object], smoke: bool) -> np.ndarray:
    test = data["outer_fold"].to_numpy(int) == outer
    return fit_predict(data.loc[~test], data.loc[test], params, OUTER_ROUNDS[outer])


def segment_week(probability: np.ndarray) -> tuple[int, float] | None:
    means = probability.mean(axis=0)
    if len(probability) == 1:
        lcb = means
    else:
        lcb = means - LCB_Z * probability.std(axis=0, ddof=1) / math.sqrt(len(probability))
    hit = np.flatnonzero(lcb >= DESIGN_Q)
    if not len(hit):
        return None
    index = int(hit[0])
    return index, float(lcb[index])


def solve(bmi: np.ndarray, probability: np.ndarray, groups: int) -> Solution | None:
    order = np.argsort(bmi, kind="mergesort")
    x, p = bmi[order], probability[order]
    internal = np.flatnonzero(x[:-1] < x[1:]) + 1
    nodes = np.concatenate(([0], internal, [len(x)])).astype(int)
    m = len(nodes)
    cost = np.full((m, m), np.inf)
    week_idx = np.full((m, m), -1, int)
    lcb_value = np.full((m, m), np.nan)
    for left in range(m - 1):
        for right in range(left + 1, m):
            a, b = int(nodes[left]), int(nodes[right])
            if b - a < MIN_GROUP_SIZE:
                continue
            selected = segment_week(p[a:b])
            if selected is None:
                continue
            wi, lcb = selected
            cost[left, right] = (b - a) * WEEKS[wi]
            week_idx[left, right] = wi
            lcb_value[left, right] = lcb
    dp = np.full((groups + 1, m), np.inf)
    back = np.full((groups + 1, m), -1, int)
    dp[0, 0] = 0.0
    for level in range(1, groups + 1):
        for right in range(1, m):
            values = dp[level - 1, :right] + cost[:right, right]
            left = int(np.argmin(values))
            dp[level, right] = values[left]
            back[level, right] = left
    if not np.isfinite(dp[groups, -1]):
        return None
    path = [m - 1]
    for level in range(groups, 0, -1):
        path.append(int(back[level, path[-1]]))
    path.reverse()
    positions = [int(nodes[node]) for node in path]
    cuts = tuple(float((x[pos - 1] + x[pos]) / 2) for pos in positions[1:-1])
    weeks = tuple(float(WEEKS[week_idx[a, b]]) for a, b in zip(path[:-1], path[1:]))
    counts = tuple(b - a for a, b in zip(positions[:-1], positions[1:]))
    min_lcb = min(float(lcb_value[a, b]) for a, b in zip(path[:-1], path[1:]))
    return Solution(groups, cuts, weeks, counts, float(dp[groups, -1] / len(x)), min_lcb)


def evaluate(bmi: np.ndarray, probability: np.ndarray, solution: Solution, audit_cuts: tuple[float, ...]) -> dict[str, object]:
    labels = np.digitize(bmi, solution.cuts)
    assigned_week = np.asarray(solution.weeks)[labels]
    indices = np.searchsorted(WEEKS, assigned_week).astype(int)
    readiness = probability[np.arange(len(bmi)), indices]
    audit = np.digitize(bmi, audit_cuts)
    audit_means, audit_lcbs = [], []
    for group in range(4):
        values = readiness[audit == group]
        if not len(values):
            continue
        mean = float(values.mean())
        lcb = mean if len(values) == 1 else mean - LCB_Z * float(values.std(ddof=1)) / math.sqrt(len(values))
        audit_means.append(mean)
        audit_lcbs.append(lcb)
    return {
        "测试平均孕周": float(assigned_week.mean()),
        "测试总体达标概率": float(readiness.mean()),
        "测试审计最低均值": min(audit_means),
        "测试审计最低LCB": min(audit_lcbs),
        "测试均值安全": bool(min(audit_means) >= DEPLOY_Q),
        "测试LCB安全": bool(min(audit_lcbs) >= DEPLOY_Q),
    }


def record_solution(solution: Solution) -> dict[str, object]:
    return {
        "组数": solution.groups,
        "切点": "|".join(f"{v:.6f}" for v in solution.cuts),
        "推荐孕周": "|".join(f"{v:.0f}" for v in solution.weeks),
        "开发组人数": "|".join(str(v) for v in solution.counts),
        "开发平均孕周": solution.average_week,
        "开发最低LCB": solution.min_lcb,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    data = load_data()
    params = historical_outer_params()
    outer_ids = [0] if args.smoke else list(range(4))
    rows, selected_rows = [], []
    for outer in outer_ids:
        train = data["outer_fold"].to_numpy(int) != outer
        test = ~train
        probabilities = inner_oof(data, outer, params[outer], args.smoke)
        max_groups = int(train.sum() // MIN_GROUP_SIZE)
        solutions = []
        for groups in range(1, max_groups + 1):
            solution = solve(data.loc[train, "孕妇BMI"].to_numpy(float), probabilities[train], groups)
            if solution is not None:
                solutions.append(solution)
                rows.append({"outer_fold": outer, "最大可行组数": max_groups, **record_solution(solution)})
        if not solutions:
            raise RuntimeError(f"外折{outer}没有可行方案")
        best = min(item.average_week for item in solutions)
        selected = min((s for s in solutions if s.average_week <= best + NEAR_WEEK), key=lambda s: s.groups)
        test_probability = outer_probability(data, outer, params[outer], args.smoke)
        audit_cuts = tuple(float(v) for v in np.quantile(data.loc[train, "孕妇BMI"], [0.25, 0.5, 0.75]))
        selected_rows.append({
            "outer_fold": outer,
            "最大可行组数": max_groups,
            **record_solution(selected),
            **evaluate(data.loc[test, "孕妇BMI"].to_numpy(float), test_probability, selected, audit_cuts),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "full"
    pd.DataFrame(rows).to_csv(OUT / f"全部可行组数开发比较_{suffix}.csv", index=False, encoding="utf-8-sig")
    selected_frame = pd.DataFrame(selected_rows)
    selected_frame.to_csv(OUT / f"严格嵌套外层审计_{suffix}.csv", index=False, encoding="utf-8-sig")
    summary = {
        "mode": suffix,
        "outer_folds": len(selected_frame),
        "group_search_rule": "1..floor(n_development/30)",
        "selected_group_counts": selected_frame["组数"].value_counts().sort_index().to_dict(),
        "mean_safe_folds": int(selected_frame["测试均值安全"].sum()),
        "lcb_safe_folds": int(selected_frame["测试LCB安全"].sum()),
        "mean_test_week": float(selected_frame["测试平均孕周"].mean()),
        "mean_test_readiness": float(selected_frame["测试总体达标概率"].mean()),
        "seed": SEED,
        "outer_rounds_from_development_inner_early_stopping": list(OUTER_ROUNDS),
    }
    (OUT / f"实验结论_{suffix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
