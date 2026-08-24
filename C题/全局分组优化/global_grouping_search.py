"""
C题问题2/3：受约束 Optimal Binning 全局BMI分组
==================================================

实验方案
--------
1. 使用已固化的患者级4折外层、3折内层；外层测试折不参与组数、切点、时点选择。
2. BMI按数值排序，只允许在相邻不同BMI的中点切分；搜索1至4组，每个开发组至少30人。
3. 对每个连续区间，用交叉拟合AFT概率确定10至25周内首次达到q的推荐时间。
4. 主目标是在组规模、25周达标及BMI四分位审计安全约束下，最小化人口加权平均检测孕周。
5. 动态规划求无审计约束目标的全局下界；若下界方案不安全，对全部合法完整路径进行
   向量化穷举，精确求解安全约束问题。算法不使用树生成候选切点。
6. 外层测试比较1至4组；安全折数优先，并用一标准误差规则选择最少组数。
7. 最终组数锁定后，在267人固定OOF概率上重新全局搜索最终切点。
8. 1000次患者级bootstrap每次重新搜索1至4组和切点；一天内性能相当则选更少组。

第一指标与约束
--------------
- 第一指标：外层测试患者加权平均检测孕周，越早越好。
- 可靠性：主分析q=0.90，敏感性q=0.85/0.95。
- 最小组人数：开发样本中每组至少30人。
- 安全性：BMI审计四分位最低平均达标概率不低于q-0.01。
- 稳定性：bootstrap组数入选率至少75%、安全率至少95%才称稳定最优。

解释边界
--------
这是固定数据上的内部验证，不是外部医院验证。连续BMI预测保留为信息上限参照；分组是为了
形成可执行策略。全局最优仅指预设的连续BMI分段、1至4组及上述约束构成的候选空间。
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "C题" / "决策优化" / "compare_grouping_policies.py"
TREE_RESULTS = ROOT / "outputs" / "C题" / "传统机器学习重构" / "tree_results.json"
BASE_OOF = ROOT / "outputs" / "C题" / "决策优化对比" / "男胎OOF达标概率_固定验证.csv"
FOLD_METRICS = ROOT / "outputs" / "C题" / "决策优化对比" / "外层测试模型指标.csv"
PREVIOUS_COMPARISON = ROOT / "outputs" / "C题" / "决策优化对比" / "分组方案对比.csv"
OUT = ROOT / "outputs" / "C题" / "全局分组优化_v1"

Q_MAIN = 0.90
Q_LEVELS = (0.85, 0.90, 0.95)
MAX_GROUPS = 4
MIN_GROUP_SIZE = 30
AUDIT_TOLERANCE = 0.01
BOOTSTRAP_REPEATS = 1000
SEED = 2025


def load_base_module():
    spec = importlib.util.spec_from_file_location("c_grouping_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载上一阶段固定验证模块")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()
TIME_GRID = np.asarray(BASE.TIME_GRID, float)
FEATURES = list(BASE.FEATURES)


@dataclass
class Graph:
    bmi: np.ndarray
    probability: np.ndarray
    audit_labels: np.ndarray
    audit_counts: np.ndarray
    nodes: np.ndarray
    cost: np.ndarray
    time_index: np.ndarray
    audit_sum: np.ndarray
    legal_edges: int


@dataclass
class Solution:
    groups: int
    cuts: tuple[float, ...]
    times: tuple[float, ...]
    average_week: float
    min_audit_readiness: float
    audit_readiness: tuple[float, ...]
    safe: bool
    exact_safe_optimum: bool
    candidate_count: int
    solver: str


def load_data_and_probabilities() -> tuple[pd.DataFrame, np.ndarray]:
    data = BASE.load_patient_table().reset_index(drop=True)
    frame = pd.read_csv(BASE_OOF)
    if not frame["孕妇代码"].equals(data["孕妇代码"]):
        raise ValueError("OOF概率文件与固定患者表顺序不一致")
    columns = [column for column in frame.columns if column.startswith("p_")]
    probability = frame[columns].to_numpy(float)
    if probability.shape != (len(data), len(TIME_GRID)) or np.isnan(probability).any():
        raise ValueError("固定OOF概率矩阵异常")
    return data, probability


def train_inner_oof_probabilities(data: pd.DataFrame) -> dict[int, np.ndarray]:
    tree = json.loads(TREE_RESULTS.read_text(encoding="utf-8"))
    params_by_outer = tree["aft"]["outer_params"]
    rounds_frame = pd.read_csv(FOLD_METRICS).set_index("outer_fold")
    result: dict[int, np.ndarray] = {}
    for outer_fold in range(4):
        print(f"[inner-OOF] outer fold {outer_fold + 1}/4", flush=True)
        outer_train = data["outer_fold"].to_numpy(int) != outer_fold
        inner_column = f"inner_fold_when_outer_{outer_fold}"
        matrix = np.full((len(data), len(TIME_GRID)), np.nan)
        params = params_by_outer[outer_fold]
        rounds = int(rounds_frame.loc[outer_fold, "inner_selected_rounds"])
        for inner_fold in range(3):
            valid = outer_train & (data[inner_column].to_numpy(int) == inner_fold)
            train = outer_train & ~valid
            dtrain = xgb.DMatrix(data.loc[train, FEATURES], feature_names=FEATURES)
            dvalid = xgb.DMatrix(data.loc[valid, FEATURES], feature_names=FEATURES)
            BASE.set_bounds(
                dtrain,
                data.loc[train, "lower"].to_numpy(float),
                data.loc[train, "upper"].to_numpy(float),
            )
            model = xgb.train(params, dtrain, num_boost_round=rounds, verbose_eval=False)
            prediction = model.predict(dvalid)
            matrix[np.flatnonzero(valid)] = np.column_stack(
                [BASE.aft_cdf(prediction, week, params) for week in TIME_GRID]
            )
        if np.isnan(matrix[outer_train]).any():
            raise RuntimeError(f"outer={outer_fold} 的inner-OOF概率不完整")
        result[outer_fold] = matrix
    return result


def make_graph(
    bmi: np.ndarray,
    probability: np.ndarray,
    q: float,
    audit_cuts: tuple[float, ...] | None = None,
) -> Graph:
    order = np.argsort(bmi, kind="mergesort")
    x = np.asarray(bmi, float)[order]
    p = np.asarray(probability, float)[order]
    if audit_cuts is None:
        audit_cuts = tuple(float(v) for v in np.quantile(x, [0.25, 0.50, 0.75]))
    audit = np.digitize(x, audit_cuts)
    audit_counts = np.bincount(audit, minlength=4).astype(float)
    internal = np.flatnonzero(x[:-1] < x[1:]) + 1
    nodes = np.concatenate(([0], internal, [len(x)])).astype(int)
    m = len(nodes)
    cost = np.full((m, m), np.inf)
    time_index = np.full((m, m), -1, dtype=np.int16)
    audit_sum = np.zeros((m, m, 4), dtype=float)
    prefix = np.vstack([np.zeros((1, p.shape[1])), np.cumsum(p, axis=0)])
    legal_edges = 0
    for left_node in range(m - 1):
        left = int(nodes[left_node])
        for right_node in range(left_node + 1, m):
            right = int(nodes[right_node])
            size = right - left
            if size < MIN_GROUP_SIZE:
                continue
            curve = (prefix[right] - prefix[left]) / size
            hit = np.flatnonzero(curve >= q)
            if not len(hit):
                continue
            t_index = int(hit[0])
            readiness = p[left:right, t_index]
            cost[left_node, right_node] = size * TIME_GRID[t_index]
            time_index[left_node, right_node] = t_index
            audit_sum[left_node, right_node] = np.bincount(
                audit[left:right], weights=readiness, minlength=4
            )
            legal_edges += 1
    return Graph(x, p, audit, audit_counts, nodes, cost, time_index, audit_sum, legal_edges)


def count_segmentations(graph: Graph, groups: int) -> int:
    m = len(graph.nodes)
    ways = np.zeros((groups + 1, m), dtype=object)
    ways[0, 0] = 1
    for level in range(1, groups + 1):
        for right in range(1, m):
            total = 0
            for left in range(right):
                if np.isfinite(graph.cost[left, right]):
                    total += ways[level - 1, left]
            ways[level, right] = total
    return int(ways[groups, m - 1])


def dynamic_programming_path(graph: Graph, groups: int) -> list[int] | None:
    m = len(graph.nodes)
    dp = np.full((groups + 1, m), np.inf)
    back = np.full((groups + 1, m), -1, dtype=int)
    dp[0, 0] = 0.0
    for level in range(1, groups + 1):
        values = dp[level - 1, :, None] + graph.cost
        back[level] = np.argmin(values, axis=0)
        dp[level] = values[back[level], np.arange(m)]
    if not np.isfinite(dp[groups, -1]):
        return None
    path = [m - 1]
    right = m - 1
    for level in range(groups, 0, -1):
        right = int(back[level, right])
        path.append(right)
    return list(reversed(path))


def path_metrics(graph: Graph, path: list[int], q: float) -> tuple[float, np.ndarray]:
    total_cost = 0.0
    audit_sum = np.zeros(4)
    for left, right in zip(path[:-1], path[1:]):
        total_cost += graph.cost[left, right]
        audit_sum += graph.audit_sum[left, right]
    audit_readiness = np.divide(
        audit_sum,
        graph.audit_counts,
        out=np.full(4, np.nan),
        where=graph.audit_counts > 0,
    )
    return total_cost / len(graph.bmi), audit_readiness


def milp_safe_path(graph: Graph, groups: int, q: float) -> list[int] | None:
    edges = np.argwhere(np.isfinite(graph.cost))
    if not len(edges):
        return None
    edge_count = len(edges)
    node_count = len(graph.nodes)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for edge_id, (left, right) in enumerate(edges):
        rows.extend([int(left), int(right), node_count])
        cols.extend([edge_id, edge_id, edge_id])
        values.extend([1.0, -1.0, 1.0])
        for audit_group in range(4):
            rows.append(node_count + 1 + audit_group)
            cols.append(edge_id)
            values.append(float(graph.audit_sum[left, right, audit_group]))
    matrix = coo_matrix(
        (values, (rows, cols)), shape=(node_count + 5, edge_count)
    ).tocsc()
    lower = np.zeros(node_count + 5)
    upper = np.zeros(node_count + 5)
    lower[0] = upper[0] = 1.0
    lower[node_count - 1] = upper[node_count - 1] = -1.0
    lower[node_count] = upper[node_count] = groups
    lower[node_count + 1:] = (q - AUDIT_TOLERANCE) * graph.audit_counts
    upper[node_count + 1:] = np.inf
    objective = np.asarray([graph.cost[left, right] for left, right in edges], float)
    result = milp(
        c=objective,
        integrality=np.ones(edge_count, dtype=np.uint8),
        bounds=Bounds(np.zeros(edge_count), np.ones(edge_count)),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": 30.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        return None
    selected = edges[result.x > 0.5]
    next_node = {int(left): int(right) for left, right in selected}
    path = [0]
    while path[-1] != len(graph.nodes) - 1:
        if path[-1] not in next_node:
            return None
        path.append(next_node[path[-1]])
    return path if len(path) == groups + 1 else None


def exhaustive_safe_path(graph: Graph, groups: int, q: float) -> list[int] | None:
    """完整检查所有合法分组；最后一个内部切点向量化，适用于预设的最多4组。"""
    end = len(graph.nodes) - 1
    threshold = (q - AUDIT_TOLERANCE) * graph.audit_counts
    best_cost = np.inf
    best_path: list[int] | None = None

    if groups == 1:
        if not np.isfinite(graph.cost[0, end]):
            return None
        audit = graph.audit_sum[0, end]
        return [0, end] if np.all(audit >= threshold - 1e-12) else None

    def finish(prefix: list[int], prefix_cost: float, prefix_audit: np.ndarray) -> None:
        nonlocal best_cost, best_path
        current = prefix[-1]
        middle = np.arange(current + 1, end, dtype=int)
        if not len(middle):
            return
        costs = prefix_cost + graph.cost[current, middle] + graph.cost[middle, end]
        finite = np.isfinite(costs) & (costs < best_cost)
        if not finite.any():
            return
        audit = (
            prefix_audit[None, :]
            + graph.audit_sum[current, middle]
            + graph.audit_sum[middle, end]
        )
        feasible = finite & np.all(audit >= threshold[None, :] - 1e-12, axis=1)
        if not feasible.any():
            return
        candidates = np.flatnonzero(feasible)
        chosen_local = candidates[int(np.argmin(costs[candidates]))]
        best_cost = float(costs[chosen_local])
        best_path = [*prefix, int(middle[chosen_local]), end]

    # groups-2条前缀边逐层展开，剩余两条边由finish批量检查。
    def expand(prefix: list[int], prefix_cost: float, prefix_audit: np.ndarray, edges_left: int) -> None:
        if edges_left == 2:
            finish(prefix, prefix_cost, prefix_audit)
            return
        current = prefix[-1]
        for next_node in range(current + 1, end):
            edge_cost = graph.cost[current, next_node]
            if not np.isfinite(edge_cost) or prefix_cost + edge_cost >= best_cost:
                continue
            expand(
                [*prefix, next_node],
                prefix_cost + float(edge_cost),
                prefix_audit + graph.audit_sum[current, next_node],
                edges_left - 1,
            )

    expand([0], 0.0, np.zeros(4), groups)
    return best_path


def solve_graph(graph: Graph, groups: int, q: float, allow_milp: bool = True) -> Solution | None:
    candidate_count = count_segmentations(graph, groups)
    if candidate_count == 0:
        return None
    path = dynamic_programming_path(graph, groups)
    if path is None:
        return None
    average_week, audit = path_metrics(graph, path, q)
    safe = bool(np.nanmin(audit) >= q - AUDIT_TOLERANCE)
    solver = "dynamic_programming_global"
    exact_safe = safe
    # 单组只有一条完整路径；不安全时不存在其他切点可供MILP替换。
    if not safe and allow_milp and groups > 1:
        safe_path = exhaustive_safe_path(graph, groups, q)
        if safe_path is not None:
            path = safe_path
            average_week, audit = path_metrics(graph, path, q)
            safe = bool(np.nanmin(audit) >= q - AUDIT_TOLERANCE - 1e-9)
            exact_safe = safe
            solver = "vectorized_exhaustive_global_safe"
    positions = [int(graph.nodes[node]) for node in path]
    cuts = tuple(
        float((graph.bmi[position - 1] + graph.bmi[position]) / 2)
        for position in positions[1:-1]
    )
    times = tuple(
        float(TIME_GRID[graph.time_index[left, right]])
        for left, right in zip(path[:-1], path[1:])
    )
    return Solution(
        groups=groups,
        cuts=cuts,
        times=times,
        average_week=float(average_week),
        min_audit_readiness=float(np.nanmin(audit)),
        audit_readiness=tuple(float(value) for value in audit),
        safe=safe,
        exact_safe_optimum=exact_safe,
        candidate_count=candidate_count,
        solver=solver,
    )


def evaluate_on_test(
    bmi: np.ndarray,
    probability: np.ndarray,
    solution: Solution,
    q: float,
    audit_cuts: tuple[float, ...],
) -> dict[str, object]:
    labels = np.digitize(bmi, solution.cuts)
    time_values = np.asarray(solution.times)
    assigned_times = time_values[labels]
    time_indices = np.asarray([int(np.argmin(np.abs(TIME_GRID - value))) for value in assigned_times])
    readiness = probability[np.arange(len(bmi)), time_indices]
    audit_labels = np.digitize(bmi, audit_cuts)
    audit_values = np.asarray([
        readiness[audit_labels == group].mean() if np.any(audit_labels == group) else np.nan
        for group in range(4)
    ])
    counts = np.bincount(labels, minlength=solution.groups)
    return {
        "测试平均检测孕周": float(assigned_times.mean()),
        "测试总体达标概率": float(readiness.mean()),
        "测试最低审计达标概率": float(np.nanmin(audit_values)),
        "测试安全合格": bool(np.nanmin(audit_values) >= q - AUDIT_TOLERANCE),
        "测试各组人数": "|".join(str(int(value)) for value in counts),
        "测试审计达标概率": "|".join(f"{value:.4f}" for value in audit_values),
    }


def solution_record(solution: Solution) -> dict[str, object]:
    return {
        "组数": solution.groups,
        "切点": "|".join(f"{value:.6f}" for value in solution.cuts),
        "推荐孕周": "|".join(f"{value:.6f}" for value in solution.times),
        "开发平均检测孕周": solution.average_week,
        "开发最低审计达标概率": solution.min_audit_readiness,
        "开发审计达标概率": "|".join(f"{value:.6f}" for value in solution.audit_readiness),
        "开发安全合格": solution.safe,
        "安全全局最优已证明": solution.exact_safe_optimum,
        "合法完整分组数": solution.candidate_count,
        "求解器": solution.solver,
    }


def outer_nested_validation(
    data: pd.DataFrame,
    outer_probability: np.ndarray,
    inner_probabilities: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    rows = []
    for outer_fold in range(4):
        train = data["outer_fold"].to_numpy(int) != outer_fold
        test = ~train
        train_bmi = data.loc[train, "孕妇BMI"].to_numpy(float)
        audit_cuts = tuple(float(v) for v in np.quantile(train_bmi, [0.25, 0.50, 0.75]))
        graph = make_graph(train_bmi, inner_probabilities[outer_fold][train], Q_MAIN, audit_cuts)
        for groups in range(1, MAX_GROUPS + 1):
            print(f"[nested] outer={outer_fold}, groups={groups}", flush=True)
            solution = solve_graph(graph, groups, Q_MAIN)
            if solution is None:
                rows.append({"outer_fold": outer_fold, "组数": groups, "可求解": False})
                continue
            record = {"outer_fold": outer_fold, "可求解": True, **solution_record(solution)}
            record.update(
                evaluate_on_test(
                    data.loc[test, "孕妇BMI"].to_numpy(float),
                    outer_probability[test],
                    solution,
                    Q_MAIN,
                    audit_cuts,
                )
            )
            rows.append(record)
    frame = pd.DataFrame(rows)
    valid = frame[frame["可求解"] == True].copy()
    summary = valid.groupby("组数", as_index=False).agg(
        外层测试平均孕周=("测试平均检测孕周", "mean"),
        外层测试孕周标准差=("测试平均检测孕周", lambda x: x.std(ddof=1)),
        外层测试安全折数=("测试安全合格", "sum"),
        开发安全折数=("开发安全合格", "sum"),
        平均测试最低审计概率=("测试最低审计达标概率", "mean"),
    )
    summary["外层测试孕周标准误"] = summary["外层测试孕周标准差"] / math.sqrt(4)
    max_safe = int(summary["外层测试安全折数"].max())
    eligible = summary[summary["外层测试安全折数"] == max_safe].copy()
    best_row = eligible.loc[eligible["外层测试平均孕周"].idxmin()]
    threshold = float(best_row["外层测试平均孕周"] + best_row["外层测试孕周标准误"])
    within_one_se = eligible[eligible["外层测试平均孕周"] <= threshold + 1e-12]
    selected_groups = int(within_one_se["组数"].min())
    summary["达到最大安全折数"] = summary["外层测试安全折数"] == max_safe
    summary["落入最佳一标准误"] = (
        summary["达到最大安全折数"]
        & (summary["外层测试平均孕周"] <= threshold + 1e-12)
    )
    summary["最终选择"] = summary["组数"] == selected_groups
    return frame, summary, selected_groups


def brute_force_small(graph: Graph, groups: int, q: float) -> Solution | None:
    best: tuple[float, list[int], np.ndarray] | None = None
    internal = range(1, len(graph.nodes) - 1)
    for cuts in itertools.combinations(internal, groups - 1):
        path = [0, *cuts, len(graph.nodes) - 1]
        if any(not np.isfinite(graph.cost[left, right]) for left, right in zip(path[:-1], path[1:])):
            continue
        average, audit = path_metrics(graph, path, q)
        if np.nanmin(audit) < q - AUDIT_TOLERANCE:
            continue
        if best is None or average < best[0]:
            best = (average, path, audit)
    if best is None:
        return None
    average, path, audit = best
    positions = [int(graph.nodes[node]) for node in path]
    return Solution(
        groups,
        tuple(float((graph.bmi[p - 1] + graph.bmi[p]) / 2) for p in positions[1:-1]),
        tuple(float(TIME_GRID[graph.time_index[l, r]]) for l, r in zip(path[:-1], path[1:])),
        float(average), float(np.nanmin(audit)), tuple(float(v) for v in audit), True, True,
        count_segmentations(graph, groups), "brute_force_test",
    )


def self_test() -> None:
    rng = np.random.default_rng(47)
    bmi = np.arange(40, dtype=float)
    latent = 12.0 + 0.08 * bmi + rng.normal(0, 0.1, len(bmi))
    probability = 1 / (1 + np.exp(-(TIME_GRID[None, :] - latent[:, None]) / 0.8))
    original_min = globals()["MIN_GROUP_SIZE"]
    globals()["MIN_GROUP_SIZE"] = 5
    try:
        graph = make_graph(bmi, probability, 0.90)
        for groups in (1, 2, 3):
            solved = solve_graph(graph, groups, 0.90)
            brute = brute_force_small(graph, groups, 0.90)
            solved_safe = solved is not None and solved.safe
            brute_safe = brute is not None and brute.safe
            if solved_safe != brute_safe:
                raise AssertionError("全局求解器与暴力枚举可行性不一致")
            if solved_safe and not np.isclose(solved.average_week, brute.average_week):
                raise AssertionError("全局求解器未复现暴力枚举最优值")
    finally:
        globals()["MIN_GROUP_SIZE"] = original_min


def choose_bootstrap_winner(solutions: list[Solution | None]) -> Solution | None:
    safe = [solution for solution in solutions if solution is not None and solution.safe]
    if not safe:
        return None
    best = min(solution.average_week for solution in safe)
    finalists = [solution for solution in safe if solution.average_week <= best + 1 / 7]
    return min(finalists, key=lambda solution: solution.groups)


def bootstrap_global_search(
    data: pd.DataFrame,
    probability: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(SEED + 3000)
    rows = []
    for repeat in range(BOOTSTRAP_REPEATS):
        if repeat % 100 == 0:
            print(f"[bootstrap] {repeat}/{BOOTSTRAP_REPEATS}", flush=True)
        index = rng.integers(0, len(data), len(data))
        bmi = data.loc[index, "孕妇BMI"].to_numpy(float)
        graph = make_graph(bmi, probability[index], Q_MAIN)
        solutions = [solve_graph(graph, groups, Q_MAIN) for groups in range(1, MAX_GROUPS + 1)]
        winner = choose_bootstrap_winner(solutions)
        record: dict[str, object] = {
            "重复": repeat + 1,
            "胜出组数": winner.groups if winner else 0,
            "胜出切点": "|".join(f"{v:.6f}" for v in winner.cuts) if winner else "",
            "胜出推荐孕周": "|".join(f"{v:.6f}" for v in winner.times) if winner else "",
            "胜出平均孕周": winner.average_week if winner else np.nan,
            "胜出最低审计概率": winner.min_audit_readiness if winner else np.nan,
            "存在安全方案": winner is not None,
        }
        for solution in solutions:
            if solution is None:
                continue
            record[f"{solution.groups}组平均孕周"] = solution.average_week
            record[f"{solution.groups}组安全"] = solution.safe
        rows.append(record)
    frame = pd.DataFrame(rows)
    counts = frame["胜出组数"].value_counts().sort_index()
    cut_intervals: dict[str, list[list[float]]] = {}
    for groups, count in counts.items():
        groups = int(groups)
        if groups <= 1 or groups == 0 or count == 0:
            continue
        parsed = [
            [float(value) for value in text.split("|") if value]
            for text in frame.loc[frame["胜出组数"] == groups, "胜出切点"].astype(str)
        ]
        parsed = [values for values in parsed if len(values) == groups - 1]
        if parsed:
            array = np.asarray(parsed, float)
            cut_intervals[str(groups)] = [
                [float(v) for v in np.quantile(array[:, index], [0.025, 0.50, 0.975])]
                for index in range(groups - 1)
            ]
    summary = {
        "repeats": BOOTSTRAP_REPEATS,
        "group_frequency": {str(int(k)): int(v) for k, v in counts.items()},
        "group_rate": {str(int(k)): float(v / BOOTSTRAP_REPEATS) for k, v in counts.items()},
        "safe_solution_rate": float(frame["存在安全方案"].mean()),
        "cut_2.5_50_97.5_percentiles_by_group": cut_intervals,
        "winner_average_week_95ci": [
            float(v) for v in frame["胜出平均孕周"].quantile([0.025, 0.975])
        ],
        "winner_min_audit_probability_95ci": [
            float(v) for v in frame["胜出最低审计概率"].quantile([0.025, 0.975])
        ],
    }
    return frame, summary


def continuous_reference(probability: np.ndarray, q: float) -> dict[str, float]:
    hit = probability >= q
    achieved = hit.any(axis=1)
    indices = np.where(achieved, hit.argmax(axis=1), len(TIME_GRID) - 1)
    readiness = probability[np.arange(len(probability)), indices]
    return {
        "平均个体化孕周": float(TIME_GRID[indices].mean()),
        "最晚个体化孕周": float(TIME_GRID[indices].max()),
        "平均达标概率": float(readiness.mean()),
        "25周内达标比例": float(achieved.mean()),
    }


def write_report(
    outer_summary: pd.DataFrame,
    selected_groups: int,
    final_solution: Solution,
    sensitivity: pd.DataFrame,
    bootstrap: dict[str, object],
    continuous: dict[str, float],
) -> None:
    selected_rate = float(bootstrap["group_rate"].get(str(selected_groups), 0.0))
    selected_outer_safety = int(
        outer_summary.loc[outer_summary["组数"] == selected_groups, "外层测试安全折数"].iloc[0]
    )
    stable = (
        selected_rate >= 0.75
        and bootstrap["safe_solution_rate"] >= 0.95
        and selected_outer_safety == 4
    )
    safety_statement = (
        "所选组数通过4/4外层测试安全审计。"
        if selected_outer_safety == 4
        else f"所选组数仅通过{selected_outer_safety}/4外层测试安全审计，因此属于最佳候选，尚非完全验证的稳定最优。"
    )
    summary_table = outer_summary.to_markdown(index=False, floatfmt=".4f")
    sensitivity_table = sensitivity.to_markdown(index=False, floatfmt=".4f")
    report = f"""# C题BMI全局最优分组报告 v3

## 1. 方法

本报告不使用树分裂生成切点。全部相邻不同BMI中点均进入候选空间，在每组至少{MIN_GROUP_SIZE}人的条件下，通过动态规划求人口加权平均检测孕周的全局下界；下界不满足审计安全时，对全部合法完整路径进行向量化穷举，精确寻找安全最优解。组数和切点选择均纳入固定外层测试。

## 2. 外层测试组数选择

{summary_table}

按最大安全折数优先及一标准误差简约规则选择 **{selected_groups}组**。{safety_statement}

## 3. 最终全样本交叉拟合方案

- 切点：{', '.join(f'{v:.3f}' for v in final_solution.cuts) or '无'}
- 各组推荐孕周：{', '.join(f'{v:.3f}' for v in final_solution.times)}
- 平均检测孕周：{final_solution.average_week:.3f}
- 最低BMI审计层达标概率：{final_solution.min_audit_readiness:.4f}
- 搜索的合法完整分组数：{final_solution.candidate_count}
- 求解方法：{final_solution.solver}

## 4. 可靠性敏感性

{sensitivity_table}

## 5. Bootstrap稳定性

1000次患者级bootstrap组数胜出率：{bootstrap['group_rate']}。存在安全方案的比例为{bootstrap['safe_solution_rate']:.1%}。

最终所选组数的入选率为{selected_rate:.1%}；稳定最优判定为 **{stable}**。判定要求组数入选率至少75%、bootstrap安全方案比例至少95%，且通过4/4外层测试安全审计。

所选组数的切点bootstrap分位数（2.5%、50%、97.5%）为：{bootstrap['cut_2.5_50_97.5_percentiles_by_group'].get(str(selected_groups), [])}。

## 6. 连续BMI参照

连续个体化策略平均孕周为{continuous['平均个体化孕周']:.3f}周。它保留全部BMI和个体预测信息，是分组策略的理论参照，不直接作为题目所需的分组答案。

## 7. 解释边界

本结论是内部嵌套验证结果，不是外部医院验证。所谓全局最优仅限于预先规定的1至4个连续BMI组、每组至少30人、每日时间网格及安全约束。
"""
    (OUT / "C题分组优化报告_v3.md").write_text(report, encoding="utf-8")


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    self_test()
    print("[self-test] global solver matches brute force", flush=True)
    data, outer_probability = load_data_and_probabilities()
    inner_probabilities = train_inner_oof_probabilities(data)
    outer_rows, outer_summary, selected_groups = outer_nested_validation(
        data, outer_probability, inner_probabilities
    )
    outer_rows.to_csv(OUT / "外层折全局搜索.csv", index=False, encoding="utf-8-sig")
    outer_summary.to_csv(OUT / "组数选择.csv", index=False, encoding="utf-8-sig")

    final_rows = []
    selected_solution: Solution | None = None
    for q in Q_LEVELS:
        graph = make_graph(data["孕妇BMI"].to_numpy(float), outer_probability, q)
        solution = solve_graph(graph, selected_groups, q)
        if solution is None:
            raise RuntimeError(f"q={q}下最终组数无可行分组")
        final_rows.append({"可靠性q": q, **solution_record(solution)})
        if np.isclose(q, Q_MAIN):
            selected_solution = solution
    if selected_solution is None:
        raise RuntimeError("未生成主可靠性最终方案")
    sensitivity = pd.DataFrame(final_rows)
    sensitivity.to_csv(OUT / "最终切点与时点.csv", index=False, encoding="utf-8-sig")

    bootstrap_frame, bootstrap_summary = bootstrap_global_search(data, outer_probability)
    bootstrap_frame.to_csv(OUT / "Bootstrap稳定性.csv", index=False, encoding="utf-8-sig")

    previous = pd.read_csv(PREVIOUS_COMPARISON)
    previous[previous["可靠性q"] == Q_MAIN].to_csv(
        OUT / "全局方案对照.csv", index=False, encoding="utf-8-sig"
    )
    continuous = continuous_reference(outer_probability, Q_MAIN)
    selected_rate = float(bootstrap_summary["group_rate"].get(str(selected_groups), 0.0))
    selected_outer_safety = int(
        outer_summary.loc[outer_summary["组数"] == selected_groups, "外层测试安全折数"].iloc[0]
    )
    stable = (
        selected_rate >= 0.75
        and bootstrap_summary["safe_solution_rate"] >= 0.95
        and selected_outer_safety == 4
    )
    conclusion = {
        "schema_version": 1,
        "method": "constrained_global_optimal_binning",
        "outer_test_not_used_for_group_selection": True,
        "selected_groups": selected_groups,
        "final_solution_q90": solution_record(selected_solution),
        "outer_summary": outer_summary.to_dict("records"),
        "bootstrap": bootstrap_summary,
        "continuous_reference": continuous,
        "stable_unique_optimum": stable,
        "recommendation_status": (
            "stable_global_optimum" if stable else "best_candidate_not_fully_safety_validated"
        ),
        "selected_group_outer_safety_folds": selected_outer_safety,
        "any_group_count_passes_all_outer_safety_folds": bool(
            (outer_summary["外层测试安全折数"] == 4).any()
        ),
        "stability_rule": "组数入选率>=75%、bootstrap安全方案率>=95%且外层测试安全4/4",
    }
    (OUT / "全局最优分组结论.json").write_text(
        json.dumps(serializable(conclusion), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        outer_summary, selected_groups, selected_solution, sensitivity,
        bootstrap_summary, continuous,
    )
    print(json.dumps(serializable(conclusion), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
