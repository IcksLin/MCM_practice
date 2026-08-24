"""C题问题4完整实验：女胎T13/T18/T21多标签异常判定。

实验方案：4折患者分组外层验证，外层训练集内再做3折患者分组选参、
Platt概率校准及阈值选择；测试折全程隔离。XGBoost与|Z|规则比较。
主指标：PR-AUC；安全性指标：灵敏度、F2；误报指标：特异度、精确率；
稳定性：患者级2000次Bootstrap 95%区间。

本程序输出的是对附件AB列筛查标签的复现能力，不是临床确诊模型。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from xgboost import XGBClassifier

from solve_q4 import (
    LABELS, PROJECT_ROOT, RESULTS, SEED, apply_platt, balanced_group_folds,
    binary_metrics, build_feature_frame, candidate_thresholds, file_sha256,
    fit_platt, load_data, patient_equal_weights, select_threshold,
)


CANDIDATES = [
    {"candidate": "A_shallow", "n_estimators": 80, "max_depth": 1, "learning_rate": 0.07,
     "min_child_weight": 2.0, "reg_lambda": 5.0, "reg_alpha": 0.0},
    {"candidate": "B_conservative", "n_estimators": 120, "max_depth": 2, "learning_rate": 0.04,
     "min_child_weight": 4.0, "reg_lambda": 8.0, "reg_alpha": 0.5},
    {"candidate": "C_slow", "n_estimators": 160, "max_depth": 2, "learning_rate": 0.03,
     "min_child_weight": 3.0, "reg_lambda": 10.0, "reg_alpha": 0.2},
]


def make_candidate(params: Dict[str, object], seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_jobs=1,
        tree_method="hist", subsample=0.85, colsample_bytree=0.85,
        random_state=seed,
        **{k: v for k, v in params.items() if k != "candidate"},
    )


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if int(np.sum(y)) else math.nan


def inner_oof(train: pd.DataFrame, label: str, folds, params, seed: int) -> np.ndarray:
    out = np.full(len(train), np.nan)
    for inner_id, (fit_idx, val_idx) in enumerate(folds):
        fit = train.iloc[fit_idx]
        val = train.iloc[val_idx]
        x_fit, medians = build_feature_frame(fit)
        x_val, _ = build_feature_frame(val, medians)
        y_fit = fit[f"y_{label}"].to_numpy(int) if label != "any" else fit["y_any"].to_numpy(int)
        model = make_candidate(params, seed + inner_id)
        model.fit(x_fit, y_fit, sample_weight=patient_equal_weights(fit, y_fit))
        out[val_idx] = model.predict_proba(x_val)[:, 1]
    if not np.isfinite(out).all():
        raise RuntimeError(f"{label}的内层OOF预测不完整")
    return out


def quality_bounds(train: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    cols = ["GC含量", "原始读段数", "唯一比对的读段数", "在参考基因组上比对的比例",
            "重复读段的比例", "被过滤掉读段数的比例"]
    return {c: dict(zip(("p005", "p01", "p99"),
                        pd.to_numeric(train[c], errors="coerce").quantile([0.005, 0.01, 0.99]).to_numpy(float)))
            for c in cols}


def quality_flag(frame: pd.DataFrame, bounds: Dict[str, Dict[str, float]]) -> np.ndarray:
    extreme_count = np.zeros(len(frame), dtype=int)
    low_read = np.zeros(len(frame), dtype=bool)
    for col, q in bounds.items():
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
        extreme_count += (~np.isfinite(values) | (values < q["p01"]) | (values > q["p99"])).astype(int)
        if col in {"原始读段数", "唯一比对的读段数"}:
            low_read |= ~np.isfinite(values) | (values < q["p005"])
    return (extreme_count >= 2) | low_read


def threshold_curve(y: np.ndarray, p: np.ndarray, outer_fold: int, label: str, source: str) -> List[Dict[str, object]]:
    thresholds = candidate_thresholds(p)
    if len(thresholds) > 250:
        thresholds = thresholds[np.unique(np.linspace(0, len(thresholds) - 1, 250).astype(int))]
    rows = []
    for t in thresholds:
        m = binary_metrics(y, (p >= t).astype(int), p)
        rows.append({"outer_fold": outer_fold, "label": label, "source": source, "threshold": float(t),
                     "sensitivity": m["sensitivity"], "specificity": m["specificity"], "f2": m["f2"]})
    return rows


def uncertainty_delta(probability: np.ndarray, threshold: float) -> float:
    return min(float(np.quantile(np.abs(probability - threshold), 0.15, method="lower")), 0.15)


def uncertainty_flags(probability: np.ndarray, threshold: float, delta: float, sequence: np.ndarray) -> np.ndarray:
    distance = np.abs(np.asarray(probability, float) - threshold)
    candidates = np.flatnonzero(distance <= delta)
    limit = int(math.floor(0.15 * len(distance)))
    order = candidates[np.lexsort((np.asarray(sequence)[candidates], distance[candidates]))]
    flags = np.zeros(len(distance), dtype=int)
    flags[order[:limit]] = 1
    return flags


def fold_audit_rows(df: pd.DataFrame, folds, level: str, outer_fold: int = -1) -> List[Dict[str, object]]:
    rows = []
    for fold, (train_idx, test_idx) in enumerate(folds):
        tr, te = df.iloc[train_idx], df.iloc[test_idx]
        trp, tep = set(tr["孕妇代码"].astype(str)), set(te["孕妇代码"].astype(str))
        row = {"level": level, "outer_fold": outer_fold, "fold": fold, "patient_overlap": len(trp & tep),
               "train_rows": len(tr), "test_rows": len(te), "train_patients": len(trp), "test_patients": len(tep)}
        for lab in (*LABELS, "any"):
            col = f"y_{lab}" if lab != "any" else "y_any"
            row[f"train_positive_records_{lab}"] = int(tr[col].sum())
            row[f"test_positive_records_{lab}"] = int(te[col].sum())
            row[f"train_positive_patients_{lab}"] = int(tr.groupby("孕妇代码")[col].max().sum())
            row[f"test_positive_patients_{lab}"] = int(te.groupby("孕妇代码")[col].max().sum())
        rows.append(row)
    return rows


def patient_bootstrap(pred: pd.DataFrame, repeats: int = 2000) -> pd.DataFrame:
    patients = np.array(sorted(pred["孕妇代码"].astype(str).unique()))
    by_patient = {p: pred[pred["孕妇代码"].astype(str) == p] for p in patients}
    rng = np.random.default_rng(SEED + 90000)
    rows = []
    targets = [(lab, f"y_{lab}", f"pred_{lab}", f"prob_{lab}") for lab in LABELS]
    targets += [("any", "y_any", "pred_any", "prob_any")]
    targets += [(f"z3_{lab}", f"y_{lab}", f"z3_pred_{lab}", f"zscore_{lab}") for lab in LABELS]
    for rep in range(repeats):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        boot = pd.concat([by_patient[p] for p in sampled], ignore_index=True)
        for name, ycol, predcol, probcol in targets:
            m = binary_metrics(boot[ycol].to_numpy(int), boot[predcol].to_numpy(int), boot[probcol].to_numpy(float))
            for metric in ("sensitivity", "specificity", "precision", "f2", "pr_auc", "roc_auc", "brier"):
                rows.append({"repeat": rep, "target": name, "metric": metric, "value": m[metric]})
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["target", "metric"])["value"].agg(
        ci_low=lambda s: s.quantile(0.025), ci_high=lambda s: s.quantile(0.975), valid="count"
    ).reset_index()
    point_rows = []
    for name, ycol, predcol, probcol in targets:
        point = binary_metrics(pred[ycol].to_numpy(int), pred[predcol].to_numpy(int), pred[probcol].to_numpy(float))
        for metric in ("sensitivity", "specificity", "precision", "f2", "pr_auc", "roc_auc", "brier"):
            point_rows.append({"target": name, "metric": metric, "point": point[metric]})
    summary = summary.merge(pd.DataFrame(point_rows), on=["target", "metric"], how="left")
    summary = summary[["target", "metric", "point", "ci_low", "ci_high", "valid"]]
    raw.to_csv(RESULTS / "q4_bootstrap_raw.csv", index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_data().reset_index(drop=True)
    outer_folds = balanced_group_folds(df, 4, SEED, repeats=2000, min_positive_patients=2)
    audits = fold_audit_rows(df, outer_folds, "outer")
    predictions, metric_rows, selection_rows, threshold_rows, importance_rows = [], [], [], [], []
    any_source_by_outer = []

    for outer_id, (train_idx, test_idx) in enumerate(outer_folds):
        train = df.iloc[train_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        inner_folds = balanced_group_folds(train, 3, SEED + 1000 + outer_id * 100, repeats=2000, min_positive_patients=1)
        audits += fold_audit_rows(train, inner_folds, "inner", outer_id)
        x_train, medians = build_feature_frame(train)
        x_test, _ = build_feature_frame(test, medians)
        bounds = quality_bounds(train)
        out = test[["序号", "孕妇代码", "检测孕周", "孕周_周", "孕妇BMI", "染色体的非整倍体", "y_T13", "y_T18", "y_T21", "y_any"]].copy()
        out["outer_fold"] = outer_id
        out["quality_extreme"] = quality_flag(test, bounds).astype(int)
        train_oof_probability = {}

        for label in (*LABELS, "any"):
            y_train = train[f"y_{label}"].to_numpy(int) if label != "any" else train["y_any"].to_numpy(int)
            y_test = test[f"y_{label}"].to_numpy(int) if label != "any" else test["y_any"].to_numpy(int)
            candidate_runs = []
            for rank, params in enumerate(CANDIDATES):
                raw_oof = inner_oof(train, label, inner_folds, params, SEED + outer_id * 10000 + rank * 100)
                ap = safe_ap(y_train, raw_oof)
                raw_threshold, raw_info = select_threshold(y_train, raw_oof)
                candidate_runs.append((ap, raw_info["f2"], -rank, params, raw_oof))
                selection_rows.append({"outer_fold": outer_id, "label": label, "candidate": params["candidate"],
                                       "inner_pr_auc": ap, "inner_f2": raw_info["f2"], "inner_threshold_raw": raw_threshold})
            _, _, _, best, best_raw_oof = max(candidate_runs, key=lambda z: (z[0], z[1], z[2]))
            platt = fit_platt(y_train, best_raw_oof)
            oof = apply_platt(platt, best_raw_oof)
            train_oof_probability[label] = oof
            threshold, threshold_info = select_threshold(y_train, oof)
            delta = uncertainty_delta(oof, threshold)
            threshold_rows += threshold_curve(y_train, oof, outer_id, label, "xgb_inner_oof")

            model = make_candidate(best, SEED + outer_id * 10000 + 900 + list((*LABELS, "any")).index(label))
            weights = patient_equal_weights(train, y_train)
            model.fit(x_train, y_train, sample_weight=weights)
            p = apply_platt(platt, model.predict_proba(x_test)[:, 1])
            pred = (p >= threshold).astype(int)
            uncertain = uncertainty_flags(p, threshold, delta, test["序号"].to_numpy())
            out[f"prob_{label}"] = p
            out[f"pred_{label}"] = pred
            out[f"uncertain_{label}"] = uncertain
            out[f"threshold_{label}"] = threshold
            for feature, value in zip(x_train.columns, model.feature_importances_):
                importance_rows.append({"outer_fold": outer_id, "label": label, "feature": feature, "importance": float(value)})
            metric_rows.append({"outer_fold": outer_id, "label": label, "method": "xgboost",
                                "threshold": threshold, "uncertainty_delta": delta,
                                "threshold_mode": threshold_info["mode"], "platt_used": platt is not None,
                                "candidate": best["candidate"], "train_weight_min": float(weights.min()),
                                "train_weight_max": float(weights.max()), **binary_metrics(y_test, pred, p)})

            if label in LABELS:
                z = pd.to_numeric(test[f"{label[1:]}号染色体的Z值"], errors="coerce").abs().to_numpy(float)
                z_train = pd.to_numeric(train[f"{label[1:]}号染色体的Z值"], errors="coerce").abs().to_numpy(float)
                z_threshold, z_info = select_threshold(y_train, z_train)
                z_pred = (z >= z_threshold).astype(int)
                out[f"zscore_{label}"] = z
                out[f"z3_pred_{label}"] = (z >= 3.0).astype(int)
                out[f"ztuned_pred_{label}"] = z_pred
                metric_rows.append({"outer_fold": outer_id, "label": label, "method": "z_tuned",
                                    "threshold": z_threshold, "uncertainty_delta": math.nan,
                                    "threshold_mode": z_info["mode"], "platt_used": False, "candidate": "abs_z",
                                    "train_weight_min": math.nan, "train_weight_max": math.nan,
                                    **binary_metrics(y_test, z_pred, z)})
                metric_rows.append({"outer_fold": outer_id, "label": label, "method": "z_fixed_3",
                                    "threshold": 3.0, "uncertainty_delta": math.nan,
                                    "threshold_mode": "fixed", "platt_used": False, "candidate": "abs_z",
                                    "train_weight_min": math.nan, "train_weight_max": math.nan,
                                    **binary_metrics(y_test, out[f"z3_pred_{label}"].to_numpy(int), z)})
        # “任一异常”来源仅用当前外层训练集的OOF结果决定，绝不读取外层测试标签。
        y_train_any = train["y_any"].to_numpy(int)
        union_oof = 1.0 - np.prod(1.0 - np.column_stack([train_oof_probability[x] for x in LABELS]), axis=1)
        direct_oof = train_oof_probability["any"]
        inner_compare = {
            "direct_pr_auc": safe_ap(y_train_any, direct_oof), "union_pr_auc": safe_ap(y_train_any, union_oof),
            "direct_brier": float(brier_score_loss(y_train_any, direct_oof)),
            "union_brier": float(brier_score_loss(y_train_any, union_oof)),
        }
        outer_source = "direct" if (
            inner_compare["direct_pr_auc"] - inner_compare["union_pr_auc"] > 0.02
            or inner_compare["union_brier"] - inner_compare["direct_brier"] > 0.02
        ) else "union"
        any_source_by_outer.append({"outer_fold": outer_id, "selected": outer_source, **inner_compare})
        out["prob_any_direct"] = out["prob_any"]
        out["pred_any_direct"] = out["pred_any"]
        out["uncertain_any_direct"] = out["uncertain_any"]
        out["prob_any_union"] = 1.0 - np.prod(1.0 - out[[f"prob_{x}" for x in LABELS]].to_numpy(float), axis=1)
        if outer_source == "union":
            union_threshold, union_info = select_threshold(y_train_any, union_oof)
            union_delta = uncertainty_delta(union_oof, union_threshold)
            out["prob_any"] = out["prob_any_union"]
            out["pred_any"] = (out["prob_any"] >= union_threshold).astype(int)
            out["uncertain_any"] = uncertainty_flags(out["prob_any"].to_numpy(float), union_threshold, union_delta,
                                                      out["序号"].to_numpy())
            threshold_rows += threshold_curve(y_train_any, union_oof, outer_id, "any", "union_inner_oof")
            for row in reversed(metric_rows):
                if row["outer_fold"] == outer_id and row["label"] == "any" and row["method"] == "xgboost":
                    row.update({"threshold": union_threshold, "uncertainty_delta": union_delta,
                                "threshold_mode": union_info["mode"], "candidate": "probability_union",
                                **binary_metrics(test["y_any"].to_numpy(int), out["pred_any"].to_numpy(int), out["prob_any"].to_numpy(float))})
                    break
        out["any_source"] = outer_source
        predictions.append(out)
        print(f"outer fold {outer_id + 1}/4 complete", flush=True)

    pred = pd.concat(predictions, ignore_index=True).sort_values("序号")
    # 三个染色体概率的并集，与直接any模型作无泄漏OOF比较。
    pred["prob_any_union"] = 1.0 - np.prod(1.0 - pred[[f"prob_{x}" for x in LABELS]].to_numpy(float), axis=1)
    y_any = pred["y_any"].to_numpy(int)
    direct_ap = safe_ap(y_any, pred["prob_any_direct"].to_numpy(float))
    union_ap = safe_ap(y_any, pred["prob_any_union"].to_numpy(float))
    direct_brier = float(brier_score_loss(y_any, pred["prob_any_direct"].to_numpy(float)))
    union_brier = float(brier_score_loss(y_any, pred["prob_any_union"].to_numpy(float)))
    # 联合概率只有在排序至少提高0.02且校准不变差时才替代直接any模型；
    # 小样本下不为极小的PR-AUC波动牺牲可操作性。

    pred["recommended_action"] = np.where(
        pred["quality_extreme"].eq(1), "质量异常：建议重抽复检",
        np.where(pred[[f"uncertain_{x}" for x in LABELS]].max(axis=1).eq(1), "阈值附近：建议复检",
                 np.where(pred["pred_any"].eq(1), "筛查阳性：建议遗传咨询及诊断性检查", "筛查阴性：常规随访")),
    )

    pred.to_csv(RESULTS / "q4_outer_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(RESULTS / "q4_outer_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selection_rows).to_csv(RESULTS / "q4_model_selection.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(threshold_rows).to_csv(RESULTS / "q4_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audits).to_csv(RESULTS / "q4_fold_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(importance_rows).to_csv(RESULTS / "q4_feature_importance.csv", index=False, encoding="utf-8-sig")
    bootstrap = patient_bootstrap(pred, repeats=2000)
    bootstrap.to_csv(RESULTS / "q4_bootstrap_summary.csv", index=False, encoding="utf-8-sig")

    recommendation_cols = ["序号", "孕妇代码", "检测孕周", "孕妇BMI", "prob_T13", "pred_T13", "prob_T18", "pred_T18",
                           "prob_T21", "pred_T21", "prob_any", "pred_any", "uncertain_any", "quality_extreme", "recommended_action"]
    pred[recommendation_cols].to_csv(RESULTS / "q4_final_recommendations.csv", index=False, encoding="utf-8-sig")
    profile_cols = ["序号", "孕妇代码", "孕周_周", "孕妇BMI", "GC含量", "原始读段数", "在参考基因组上比对的比例",
                    "重复读段的比例", "13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值",
                    "y_T13", "y_T18", "y_T21", "y_any"]
    df[profile_cols].to_csv(RESULTS / "q4_profile_input.csv", index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame(metric_rows)
    summary_metrics = metrics.groupby(["label", "method"])[["sensitivity", "specificity", "precision", "f2", "roc_auc", "pr_auc", "brier"]].mean().reset_index()
    summary_metrics.to_csv(RESULTS / "q4_metrics_summary.csv", index=False, encoding="utf-8-sig")
    final = {
        "status": "complete_nested_group_cv", "input_sha256": file_sha256(PROJECT_ROOT / "data" / "raw" / "附件.xlsx"),
        "rows": len(df), "patients": int(df["孕妇代码"].nunique()), "outer_folds": 4, "inner_folds": 3,
        "split_search_repeats": 2000, "bootstrap_repeats": 2000, "labels": list(LABELS),
        "any_probability_comparison_descriptive_only": {"direct_pr_auc": direct_ap, "union_pr_auc": union_ap,
                                       "direct_brier": direct_brier, "union_brier": union_brier},
        "any_selection_policy": "each outer fold selected using only its training OOF probabilities",
        "any_source_by_outer": any_source_by_outer,
        "patient_overlap_max": int(pd.DataFrame(audits)["patient_overlap"].max()),
        "important_limit": "附件AE列均为‘是’，只能验证AB筛查标签复现，不能声称预测真实胎儿疾病。",
    }
    (RESULTS / "q4_final_method.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
