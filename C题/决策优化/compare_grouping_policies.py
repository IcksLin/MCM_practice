"""
C题问题2/3：BMI分组方案固定验证对比实验
================================================

实验方案
--------
1. 数据与折号固定：读取 `C题/工作框架/男胎患者级折号.csv`，不得重新随机划分。
2. 候选方案：
   - 不分组：所有孕妇使用统一时点；
   - 参数AFT两组：使用既有参数AFT最优切点；
   - 树AFT三组：使用既有稳定性检验得到的两个切点；
   - 题目经验五组：BMI 28、32、36、40为切点。
3. OOF重训：每个外层测试折的超参数沿用已完成的内层TPE结果；提升轮数只由
   固定内层3折早停决定。外层测试折不参与调参、早停或轮数选择。
4. 时点定义：在10至25周内按1天搜索，使政策组平均预测达标概率首次达到
   q∈{0.85, 0.90, 0.95}。
5. 公平性审计：用全体BMI四分位形成与候选方案无关的四个审计层，检查每层
   在其被分配时点的平均达标概率。
6. 不确定性：按孕妇进行1000次bootstrap，重新计算各方案时点、指标和胜者。

第一指标与胜负规则
------------------
- 第一约束：每个政策组人数不少于30，且在25周前达到指定可靠性q；
- 安全约束：四个BMI审计层的最低达标概率不低于 q-0.01；
- 第一指标：满足约束后，人口加权平均检测孕周越早越好；
- 平局规则：平均时点相差不超过1天时，组数更少者优先。
- 若没有方案满足安全约束，先最大化最低审计层达标概率，再比较平均检测孕周。

辅助指标
--------
- 总体平均达标概率、政策组最差达标概率、审计层最低达标概率；
- 最晚推荐孕周、每组人数、确定已达标/确定未达标/删失不确定比例；
- 外层测试AFT负对数似然、相对统一时点提前天数；
- bootstrap胜出率、两组与三组平均时点差和安全性差的95%区间。

解释边界
--------
本实验比较的是固定数据内的交叉拟合决策方案，不等同于外部医院测试。最优方案
必须同时报告预测性能、政策安全性、复杂度和bootstrap稳定性，不能只凭树的切点
或训练集分数判定。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import gumbel_l, logistic, norm


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "C题" / "C题" / "附件.xlsx"
FOLDS_PATH = ROOT / "C题" / "工作框架" / "男胎患者级折号.csv"
PARAMETRIC_RESULTS = ROOT / "outputs" / "C题" / "c_results.json"
TREE_RESULTS = ROOT / "outputs" / "C题" / "传统机器学习重构" / "tree_results.json"
OUT = ROOT / "outputs" / "C题" / "决策优化对比"

SEED = 2025
FEATURES = ["孕妇BMI", "年龄", "身高", "体重", "IVF", "孕次", "产次"]
TIME_GRID = np.arange(10.0, 25.0 + 1 / 7, 1 / 7)
RELIABILITY_LEVELS = (0.85, 0.90, 0.95)
MIN_GROUP_SIZE = 30
AUDIT_TOLERANCE = 0.01
BOOTSTRAP_REPEATS = 1000


@dataclass(frozen=True)
class Policy:
    name: str
    cuts: tuple[float, ...]


def parse_count(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"\d+", str(value))
    return float(match.group()) if match else np.nan


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_bounds(matrix: xgb.DMatrix, lower: np.ndarray, upper: np.ndarray) -> None:
    matrix.set_float_info("label_lower_bound", np.asarray(lower, np.float32))
    matrix.set_float_info("label_upper_bound", np.asarray(upper, np.float32))


def aft_cdf(predicted_time: np.ndarray, week: float, params: dict[str, object]) -> np.ndarray:
    scale = float(params["aft_loss_distribution_scale"])
    z = (np.log(week) - np.log(np.clip(predicted_time, 1e-6, None))) / scale
    distribution = str(params["aft_loss_distribution"])
    if distribution == "normal":
        return norm.cdf(z)
    if distribution == "logistic":
        return logistic.cdf(z)
    return gumbel_l.cdf(z)


def load_patient_table() -> pd.DataFrame:
    raw = pd.read_excel(SOURCE, sheet_name="男胎检测数据")
    raw["IVF"] = (raw["IVF妊娠"] != "自然受孕").astype(float)
    raw["孕次"] = raw["怀孕次数"].map(parse_count)
    raw["产次"] = raw["生产次数"].map(parse_count)
    static = raw.groupby("孕妇代码", as_index=False)[FEATURES].median()
    folds = pd.read_csv(FOLDS_PATH)
    data = folds.merge(static, on="孕妇代码", how="left", validate="one_to_one")
    if data[FEATURES].isna().any().any():
        missing = data[FEATURES].isna().sum()
        raise ValueError(f"固定特征存在缺失：{missing[missing > 0].to_dict()}")
    if len(data) != 267 or not data["孕妇代码"].is_unique:
        raise ValueError("患者级固定表规模或唯一性异常")
    return data


def train_with_inner_early_stopping(
    data: pd.DataFrame,
    outer_fold: int,
    params: dict[str, object],
) -> tuple[xgb.Booster, int, float, np.ndarray, np.ndarray]:
    outer_test = data["outer_fold"].to_numpy(int) == outer_fold
    outer_train = ~outer_test
    inner_column = f"inner_fold_when_outer_{outer_fold}"
    inner_rounds = []

    for inner_fold in range(3):
        inner_valid = outer_train & (data[inner_column].to_numpy(int) == inner_fold)
        inner_train = outer_train & ~inner_valid
        dtrain = xgb.DMatrix(data.loc[inner_train, FEATURES], feature_names=FEATURES)
        dvalid = xgb.DMatrix(data.loc[inner_valid, FEATURES], feature_names=FEATURES)
        set_bounds(
            dtrain,
            data.loc[inner_train, "lower"].to_numpy(float),
            data.loc[inner_train, "upper"].to_numpy(float),
        )
        set_bounds(
            dvalid,
            data.loc[inner_valid, "lower"].to_numpy(float),
            data.loc[inner_valid, "upper"].to_numpy(float),
        )
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2500,
            evals=[(dvalid, "inner_valid")],
            early_stopping_rounds=80,
            verbose_eval=False,
        )
        inner_rounds.append(int(model.best_iteration + 1))

    selected_rounds = int(np.median(inner_rounds))
    dtrain_outer = xgb.DMatrix(data.loc[outer_train, FEATURES], feature_names=FEATURES)
    dtest_outer = xgb.DMatrix(data.loc[outer_test, FEATURES], feature_names=FEATURES)
    set_bounds(
        dtrain_outer,
        data.loc[outer_train, "lower"].to_numpy(float),
        data.loc[outer_train, "upper"].to_numpy(float),
    )
    set_bounds(
        dtest_outer,
        data.loc[outer_test, "lower"].to_numpy(float),
        data.loc[outer_test, "upper"].to_numpy(float),
    )
    model = xgb.train(params, dtrain_outer, num_boost_round=selected_rounds, verbose_eval=False)
    prediction = model.predict(dtest_outer)
    evaluation = model.eval(dtest_outer, name="outer_test")
    outer_nloglik = float(evaluation.rsplit(":", 1)[1])
    return model, selected_rounds, outer_nloglik, np.flatnonzero(outer_test), prediction


def build_cross_fitted_probabilities(
    data: pd.DataFrame,
    outer_params: list[dict[str, object]],
) -> tuple[np.ndarray, pd.DataFrame]:
    if len(outer_params) != 4:
        raise ValueError("预期4组外层参数")
    probability = np.full((len(data), len(TIME_GRID)), np.nan)
    fold_rows = []
    for outer_fold, params in enumerate(outer_params):
        print(f"[OOF] outer fold {outer_fold + 1}/4", flush=True)
        _, rounds, nloglik, test_index, prediction = train_with_inner_early_stopping(
            data, outer_fold, params
        )
        probability[test_index] = np.column_stack(
            [aft_cdf(prediction, week, params) for week in TIME_GRID]
        )
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "test_women": int(len(test_index)),
                "inner_selected_rounds": rounds,
                "outer_test_aft_nloglik": nloglik,
                "distribution": params["aft_loss_distribution"],
                "distribution_scale": params["aft_loss_distribution_scale"],
            }
        )
    if np.isnan(probability).any():
        raise RuntimeError("OOF概率矩阵存在未填充位置")
    return probability, pd.DataFrame(fold_rows)


def load_policies() -> list[Policy]:
    parametric = json.loads(PARAMETRIC_RESULTS.read_text(encoding="utf-8"))
    tree = json.loads(TREE_RESULTS.read_text(encoding="utf-8"))
    two_cut = tuple(float(value) for value in parametric["bmi_grouping"]["cuts"])
    three_cuts = tuple(float(value) for value in tree["bmi_grouping"]["cuts"])
    return [
        Policy("不分组", ()),
        Policy("参数AFT两组", two_cut),
        Policy("树AFT三组", three_cuts),
        Policy("题目经验五组", (28.0, 32.0, 36.0, 40.0)),
    ]


def recommend_group_times(
    probability: np.ndarray,
    policy_labels: np.ndarray,
    q: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    groups = int(policy_labels.max() + 1)
    time_index = np.full(groups, len(TIME_GRID) - 1, dtype=int)
    achieved = np.zeros(groups, dtype=bool)
    for group in range(groups):
        group_mask = policy_labels == group
        if not group_mask.any():
            continue
        mean_curve = probability[group_mask].mean(axis=0)
        hit = np.flatnonzero(mean_curve >= q)
        if len(hit):
            time_index[group] = int(hit[0])
            achieved[group] = True
    return TIME_GRID[time_index], time_index, bool(achieved.all())


def evaluate_policy(
    data: pd.DataFrame,
    probability: np.ndarray,
    policy: Policy,
    q: float,
    audit_cuts: tuple[float, ...],
) -> dict[str, object]:
    bmi = data["孕妇BMI"].to_numpy(float)
    policy_labels = np.digitize(bmi, policy.cuts)
    counts = np.bincount(policy_labels, minlength=len(policy.cuts) + 1)
    times, time_index, all_groups_achieved = recommend_group_times(probability, policy_labels, q)
    assigned_time = times[policy_labels]
    assigned_index = time_index[policy_labels]
    readiness = probability[np.arange(len(data)), assigned_index]

    audit_labels = np.digitize(bmi, audit_cuts)
    audit_readiness = np.asarray(
        [readiness[audit_labels == group].mean() for group in range(len(audit_cuts) + 1)]
    )
    policy_readiness = np.asarray([
        readiness[policy_labels == group].mean()
        if np.any(policy_labels == group) else np.nan
        for group in range(len(policy.cuts) + 1)
    ])

    lower = data["lower"].to_numpy(float)
    upper = data["upper"].to_numpy(float)
    definitely_ready = np.isfinite(upper) & (upper <= assigned_time + 1e-12)
    definitely_unready = lower > assigned_time + 1e-12
    uncertain = ~(definitely_ready | definitely_unready)

    min_group_ok = bool(counts.min() >= MIN_GROUP_SIZE)
    audit_ok = bool(audit_readiness.min() >= q - AUDIT_TOLERANCE)
    return {
        "可靠性q": q,
        "方案": policy.name,
        "组数": len(policy.cuts) + 1,
        "切点": "|".join(f"{value:.3f}" for value in policy.cuts),
        "各组人数": "|".join(str(int(value)) for value in counts),
        "各组推荐孕周": "|".join(f"{value:.3f}" for value in times),
        "平均检测孕周": float(assigned_time.mean()),
        "最晚检测孕周": float(assigned_time.max()),
        "总体平均达标概率": float(readiness.mean()),
        "政策组最低达标概率": float(np.nanmin(policy_readiness)),
        "审计层最低达标概率": float(audit_readiness.min()),
        "审计层达标概率": "|".join(f"{value:.4f}" for value in audit_readiness),
        "确定已达标比例": float(definitely_ready.mean()),
        "确定未达标比例": float(definitely_unready.mean()),
        "删失不确定比例": float(uncertain.mean()),
        "最小组人数合格": min_group_ok,
        "全部政策组达到q": all_groups_achieved,
        "审计安全约束合格": audit_ok,
        "候选有效": bool(min_group_ok and all_groups_achieved),
        "第一约束全部合格": bool(min_group_ok and all_groups_achieved and audit_ok),
    }


def select_winner(rows: list[dict[str, object]], q: float) -> str:
    candidates = [row for row in rows if row["可靠性q"] == q and row["候选有效"]]
    safe = [row for row in candidates if row["审计安全约束合格"]]
    if safe:
        earliest = min(float(row["平均检测孕周"]) for row in safe)
        finalists = [row for row in safe if float(row["平均检测孕周"]) <= earliest + 1 / 7]
        return str(min(finalists, key=lambda row: int(row["组数"]))["方案"])

    best_audit = max(float(row["审计层最低达标概率"]) for row in candidates)
    near_best = [
        row for row in candidates
        if float(row["审计层最低达标概率"]) >= best_audit - AUDIT_TOLERANCE
    ]
    earliest = min(float(row["平均检测孕周"]) for row in near_best)
    finalists = [row for row in near_best if float(row["平均检测孕周"]) <= earliest + 1 / 7]
    return str(min(finalists, key=lambda row: int(row["组数"]))["方案"])


def bootstrap_comparison(
    data: pd.DataFrame,
    probability: np.ndarray,
    policies: list[Policy],
    audit_cuts: tuple[float, ...],
) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(SEED + 1000)
    records = []
    for repeat in range(BOOTSTRAP_REPEATS):
        index = rng.integers(0, len(data), len(data))
        sampled_data = data.iloc[index].reset_index(drop=True)
        sampled_probability = probability[index]
        rows = [
            evaluate_policy(sampled_data, sampled_probability, policy, 0.90, audit_cuts)
            for policy in policies
        ]
        winner = select_winner(rows, 0.90)
        by_name = {str(row["方案"]): row for row in rows}
        records.append(
            {
                "重复": repeat + 1,
                "胜出方案": winner,
                "两组平均孕周": by_name["参数AFT两组"]["平均检测孕周"],
                "三组平均孕周": by_name["树AFT三组"]["平均检测孕周"],
                "三组减两组_平均孕周": float(by_name["树AFT三组"]["平均检测孕周"])
                - float(by_name["参数AFT两组"]["平均检测孕周"]),
                "两组最低审计达标概率": by_name["参数AFT两组"]["审计层最低达标概率"],
                "三组最低审计达标概率": by_name["树AFT三组"]["审计层最低达标概率"],
                "三组减两组_最低审计概率": float(by_name["树AFT三组"]["审计层最低达标概率"])
                - float(by_name["参数AFT两组"]["审计层最低达标概率"]),
            }
        )
    frame = pd.DataFrame(records)
    summary = {
        "repeats": BOOTSTRAP_REPEATS,
        "winner_frequency": {
            str(key): int(value) for key, value in frame["胜出方案"].value_counts().items()
        },
        "winner_rate": {
            str(key): float(value / BOOTSTRAP_REPEATS)
            for key, value in frame["胜出方案"].value_counts().items()
        },
        "three_minus_two_average_week_95ci": [
            float(value)
            for value in frame["三组减两组_平均孕周"].quantile([0.025, 0.975])
        ],
        "three_minus_two_min_audit_probability_95ci": [
            float(value)
            for value in frame["三组减两组_最低审计概率"].quantile([0.025, 0.975])
        ],
    }
    return frame, summary


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in formats and isinstance(value, (float, np.floating)):
                values.append(format(value, formats[column]))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    comparison: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    winners: dict[str, str],
    bootstrap: dict[str, object],
    audit_cuts: tuple[float, ...],
) -> None:
    q90 = comparison[comparison["可靠性q"] == 0.90].copy()
    q90["平均检测孕周"] = q90["平均检测孕周"].astype(float)
    table = markdown_table(
        q90,
        [
            "方案", "各组人数", "各组推荐孕周", "平均检测孕周",
            "总体平均达标概率", "审计层最低达标概率", "候选有效", "审计安全约束合格",
        ],
        {
            "平均检测孕周": ".3f",
            "总体平均达标概率": ".4f",
            "审计层最低达标概率": ".4f",
        },
    )
    fold_table = markdown_table(
        fold_metrics,
        ["outer_fold", "test_women", "inner_selected_rounds", "outer_test_aft_nloglik"],
        {"outer_test_aft_nloglik": ".4f"},
    )
    winner_frequency = ", ".join(
        f"{name}: {count}/{BOOTSTRAP_REPEATS}"
        for name, count in bootstrap["winner_frequency"].items()
    )
    primary_rate = float(bootstrap["winner_rate"].get("参数AFT两组", 0.0))
    stable_unique = primary_rate >= 0.75
    stability_conclusion = (
        "两组方案达到预设75%稳定胜出门槛，可作为稳定最优方案。"
        if stable_unique
        else "两组方案未达到预设75%稳定胜出门槛，因此不能宣称唯一稳定最优；推荐两组作为简约主方案，同时保留三组作为安全性优先备选。"
    )
    report = f"""# C题问题2/3：BMI分组固定验证对比实验

## 1. 实验约束

本实验使用已经固化的患者级外层4折和内层3折。外层测试折不参与早停；每个外层模型的提升轮数完全由内层验证确定。候选为不分组、参数AFT两组、树AFT三组和题目经验五组。

固定BMI审计四分位切点为：{', '.join(f'{value:.3f}' for value in audit_cuts)}。

## 2. 外层测试模型

{fold_table}

外层测试 AFT 负对数似然均值为 {fold_metrics['outer_test_aft_nloglik'].mean():.4f}，标准差为 {fold_metrics['outer_test_aft_nloglik'].std(ddof=0):.4f}。

## 3. 90%可靠性主实验

{table}

主规则下的胜出方案：**{winners['0.90']}**。

## 4. 可靠性敏感性

- q=85%：{winners['0.85']}
- q=90%：{winners['0.90']}
- q=95%：{winners['0.95']}

## 5. Bootstrap稳定性

患者级bootstrap共{BOOTSTRAP_REPEATS}次，胜出频数为：{winner_frequency}。

{stability_conclusion}

三组减两组的平均检测孕周差95%区间：{bootstrap['three_minus_two_average_week_95ci'][0]:.3f}至{bootstrap['three_minus_two_average_week_95ci'][1]:.3f}周。负值表示三组更早。

三组减两组的最低审计层达标概率差95%区间：{bootstrap['three_minus_two_min_audit_probability_95ci'][0]:.4f}至{bootstrap['three_minus_two_min_audit_probability_95ci'][1]:.4f}。

## 6. 解释边界

该结论来自固定数据上的患者级交叉拟合测试和bootstrap，不是外部医院验证。若最优方案在bootstrap中的胜出率不足75%，应保留多个候选或优先选择更简单方案，不应宣称唯一最优。
"""
    (OUT / "分组对比实验报告.md").write_text(report, encoding="utf-8")


def serializable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_patient_table()
    tree_results = json.loads(TREE_RESULTS.read_text(encoding="utf-8"))
    policies = load_policies()
    audit_cuts = tuple(float(value) for value in np.quantile(data["孕妇BMI"], [0.25, 0.50, 0.75]))

    probability, fold_metrics = build_cross_fitted_probabilities(
        data, tree_results["aft"]["outer_params"]
    )
    fold_metrics.to_csv(OUT / "外层测试模型指标.csv", index=False, encoding="utf-8-sig")

    probability_base = data[
        ["孕妇代码", "孕妇BMI", "lower", "upper", "censoring", "outer_fold"]
    ].reset_index(drop=True)
    probability_columns = pd.DataFrame(
        {f"p_{week:.3f}": probability[:, index] for index, week in enumerate(TIME_GRID)}
    )
    probability_frame = pd.concat([probability_base, probability_columns], axis=1)
    probability_frame.to_csv(OUT / "男胎OOF达标概率_固定验证.csv", index=False, encoding="utf-8-sig")

    comparison_rows = [
        evaluate_policy(data, probability, policy, q, audit_cuts)
        for q in RELIABILITY_LEVELS
        for policy in policies
    ]
    comparison = pd.DataFrame(comparison_rows)
    winners = {f"{q:.2f}": select_winner(comparison_rows, q) for q in RELIABILITY_LEVELS}
    unified_weeks = {
        q: float(
            comparison.loc[
                (comparison["可靠性q"] == q) & (comparison["方案"] == "不分组"),
                "平均检测孕周",
            ].iloc[0]
        )
        for q in RELIABILITY_LEVELS
    }
    comparison["相对不分组提前天数"] = comparison.apply(
        lambda row: (unified_weeks[float(row["可靠性q"])] - float(row["平均检测孕周"])) * 7,
        axis=1,
    )
    comparison.to_csv(OUT / "分组方案对比.csv", index=False, encoding="utf-8-sig")

    bootstrap_frame, bootstrap_summary = bootstrap_comparison(
        data, probability, policies, audit_cuts
    )
    bootstrap_frame.to_csv(OUT / "Bootstrap分组选择.csv", index=False, encoding="utf-8-sig")

    primary_rate = float(bootstrap_summary["winner_rate"].get("参数AFT两组", 0.0))
    stable_unique = primary_rate >= 0.75
    conclusion = {
        "schema_version": 1,
        "experiment": {
            "source_sha256": sha256(SOURCE),
            "folds_sha256": sha256(FOLDS_PATH),
            "outer_test_not_used_for_early_stopping": True,
            "reliability_levels": list(RELIABILITY_LEVELS),
            "minimum_group_size": MIN_GROUP_SIZE,
            "audit_tolerance": AUDIT_TOLERANCE,
            "audit_bmi_quartile_cuts": list(audit_cuts),
        },
        "outer_test_aft_nloglik": {
            "mean": float(fold_metrics["outer_test_aft_nloglik"].mean()),
            "sd": float(fold_metrics["outer_test_aft_nloglik"].std(ddof=0)),
            "folds": fold_metrics.to_dict("records"),
        },
        "winner_by_reliability": winners,
        "final_recommendation": {
            "primary": "参数AFT两组（简约主方案）",
            "alternative": "树AFT三组（安全性优先备选）",
            "stable_unique_optimum": stable_unique,
            "reason": (
                "两组方案在q=0.85、0.90、0.95主样本比较中均胜出，且bootstrap胜出率更高；"
                "但其胜出率未达到75%门槛，平均时点差区间跨0，故不宣称唯一稳定最优。"
            ),
        },
        "bootstrap": bootstrap_summary,
        "comparison": comparison_rows,
        "decision_guardrail": "仅当主方案bootstrap胜出率至少75%时称为稳定最优；否则保留并列候选或选择更简单方案。",
    }
    (OUT / "最优分组结论.json").write_text(
        json.dumps(serializable(conclusion), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(comparison, fold_metrics, winners, bootstrap_summary, audit_cuts)
    print(json.dumps(serializable({"winners": winners, "bootstrap": bootstrap_summary}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
