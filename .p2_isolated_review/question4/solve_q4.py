"""C题问题4：女胎T13/T18/T21多标签异常判定。

实验合同：
1. 同一孕妇的全部检测始终位于同一折，禁止按记录随机拆分。
2. 训练折内完成缺失填补、患者等权、正类权重、概率校准和阈值选择。
3. 外层测试折只评价，不参与参数、阈值或质量边界选择。
4. --smoke 运行一个真实外层折和T18标签，用于P1最小门禁。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "C题" / "附件.xlsx"
RESULTS = PROJECT_ROOT / "results"
EXPECTED_SHA256 = "14827156218bd4f7e4f16db4aa6d9f757c6648379e038ae6c6b58383648614af"
SEED = 20250824
LABELS = ("T13", "T18", "T21")

BASE_NUMERIC = [
    "年龄", "身高", "体重", "孕妇BMI", "孕周_周", "检测抽血次数",
    "原始读段数", "在参考基因组上比对的比例", "重复读段的比例", "唯一比对的读段数",
    "GC含量", "13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值",
    "X染色体的Z值", "X染色体浓度", "13号染色体的GC含量", "18号染色体的GC含量",
    "21号染色体的GC含量", "被过滤掉读段数的比例", "怀孕次数", "生产次数",
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_week(value: object) -> float:
    text = str(value).strip().lower()
    if "w" not in text:
        raise ValueError(f"无法解析孕周: {value!r}")
    week, rest = text.split("w", 1)
    days = int(rest.replace("+", "")) if rest.replace("+", "") else 0
    return float(week) + days / 7.0


def load_data() -> pd.DataFrame:
    digest = file_sha256(INPUT)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"附件哈希异常: {digest}")
    df = pd.read_excel(INPUT, sheet_name="女胎检测数据", header=0)
    if df.shape != (605, 31) or df["孕妇代码"].nunique() != 147:
        raise RuntimeError(f"女胎数据规模异常: {df.shape}, patients={df['孕妇代码'].nunique()}")
    if int(df.iloc[0]["序号"]) != 1 or int(df.iloc[-1]["序号"]) != 605:
        raise RuntimeError("首末行序号异常")
    df = df.copy()
    df["孕周_周"] = df["检测孕周"].map(parse_week)
    raw_label = df["染色体的非整倍体"].fillna("").astype(str)
    for label in LABELS:
        df[f"y_{label}"] = raw_label.str.contains(label, regex=False).astype(int)
    df["y_any"] = df[[f"y_{x}" for x in LABELS]].max(axis=1)
    return df


def build_feature_frame(df: pd.DataFrame, medians: pd.Series | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    x = df[BASE_NUMERIC].apply(pd.to_numeric, errors="coerce").copy()
    x["BMI缺失"] = x["孕妇BMI"].isna().astype(int)
    if medians is None:
        medians = x.median(numeric_only=True)
    x = x.fillna(medians)
    for chrom in ("13", "18", "21"):
        x[f"abs_Z{chrom}"] = x[f"{chrom}号染色体的Z值"].abs()
        x[f"GC偏离_{chrom}"] = x[f"{chrom}号染色体的GC含量"] - x["GC含量"]
    x["最大绝对目标Z"] = x[["abs_Z13", "abs_Z18", "abs_Z21"]].max(axis=1)
    x["目标Z极差"] = x[["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值"]].max(axis=1) - x[["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值"]].min(axis=1)
    denom = x["原始读段数"].clip(lower=1)
    x["唯一读段占比"] = x["唯一比对的读段数"] / denom
    x["IVF_试管"] = df["IVF妊娠"].astype(str).str.contains("IVF", regex=False).astype(int).to_numpy()
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise RuntimeError("特征中存在NaN或Inf")
    return x.astype(float), medians


def patient_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, group in df.groupby("孕妇代码", sort=True):
        rows.append({
            "孕妇代码": code,
            "records": len(group),
            **{f"records_{lab}": int(group[f"y_{lab}"].sum()) for lab in LABELS},
            **{f"patient_{lab}": int(group[f"y_{lab}"].max()) for lab in LABELS},
            "patient_any": int(group["y_any"].max()),
        })
    return pd.DataFrame(rows).set_index("孕妇代码")


def balanced_group_folds(
    df: pd.DataFrame,
    n_splits: int,
    seed: int,
    repeats: int = 2000,
    min_positive_patients: int = 1,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    summary = patient_summary(df)
    codes = summary.index.to_list()
    metric_cols = ["records", "records_T13", "records_T18", "records_T21", "patient_any"]
    target = summary[metric_cols].sum().to_numpy(float) / n_splits
    patient_rates = {lab: max(summary[f"patient_{lab}"].sum(), 1) for lab in LABELS}
    rarity = {
        code: sum(summary.loc[code, f"patient_{lab}"] / patient_rates[lab] for lab in LABELS)
        for code in codes
    }
    best = None
    best_key = None
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        jitter = {code: float(rng.random()) for code in codes}
        order = sorted(codes, key=lambda c: (-rarity[c], -summary.loc[c, "records"], jitter[c], c))
        fold_metrics = np.zeros((n_splits, len(metric_cols)), dtype=float)
        assignments: Dict[str, int] = {}
        for code in order:
            vec = summary.loc[code, metric_cols].to_numpy(float)
            scores = []
            for fold in range(n_splits):
                trial = fold_metrics.copy()
                trial[fold] += vec
                score = float(np.sum(((trial - target) / (target + 1.0)) ** 2))
                scores.append((score, fold))
            chosen = min(scores)[1]
            fold_metrics[chosen] += vec
            assignments[code] = chosen
        feasible = True
        for fold in range(n_splits):
            fold_codes = [c for c, f in assignments.items() if f == fold]
            for lab in LABELS:
                if int(summary.loc[fold_codes, f"patient_{lab}"].sum()) < min_positive_patients:
                    feasible = False
        if not feasible:
            continue
        objective = float(np.sum(((fold_metrics - target) / (target + 1.0)) ** 2))
        lex = tuple(assignments[c] for c in sorted(codes))
        key = (round(objective, 14), lex)
        if best_key is None or key < best_key:
            best_key = key
            best = assignments.copy()
    if best is None:
        raise RuntimeError(f"无法构造满足阳性患者约束的{n_splits}折")
    folds = []
    groups = df["孕妇代码"].astype(str).to_numpy()
    for fold in range(n_splits):
        test_mask = np.array([best[g] == fold for g in groups])
        folds.append((np.flatnonzero(~test_mask), np.flatnonzero(test_mask)))
    return folds


def patient_equal_weights(df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    counts = df.groupby("孕妇代码")["孕妇代码"].transform("size").to_numpy(float)
    base = 1.0 / counts
    base /= base.mean()
    positives = max(float(y.sum()), 1.0)
    negatives = max(float(len(y) - y.sum()), 1.0)
    alpha = negatives / positives
    return base * np.where(y == 1, alpha, 1.0)


def make_model(seed: int, n_estimators: int = 100) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators,
        max_depth=2,
        learning_rate=0.05,
        min_child_weight=3.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        reg_alpha=0.2,
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
    )


def candidate_thresholds(score: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(score, dtype=float))
    if len(unique) == 1:
        return np.array([np.nextafter(unique[0], -np.inf), np.nextafter(unique[0], np.inf)])
    mids = (unique[:-1] + unique[1:]) / 2.0
    return np.r_[np.nextafter(unique[0], -np.inf), mids, np.nextafter(unique[-1], np.inf)]


def select_threshold(y: np.ndarray, score: np.ndarray, min_sensitivity: float = 0.85) -> Tuple[float, Dict[str, float]]:
    rows = []
    for threshold in candidate_thresholds(score):
        pred = (score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f2 = fbeta_score(y, pred, beta=2, zero_division=0)
        rows.append((threshold, sensitivity, specificity, f2))
    feasible = [r for r in rows if r[1] >= min_sensitivity] if int(y.sum()) >= 8 else []
    if feasible:
        chosen = max(feasible, key=lambda r: (r[2], r[3], r[0]))
        mode = "sensitivity_constrained"
    else:
        chosen = max(rows, key=lambda r: (r[3], r[1], -r[0]))
        mode = "f2_fallback"
    return float(chosen[0]), {"sensitivity": chosen[1], "specificity": chosen[2], "f2": chosen[3], "mode": mode}


def fit_platt(y: np.ndarray, probability: np.ndarray):
    if int(y.sum()) < 8 or int((1 - y).sum()) < 20:
        return None
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1.0, solver="lbfgs", random_state=SEED)
    try:
        model.fit(logit, y)
    except (ValueError, FloatingPointError):
        return None
    return model


def apply_platt(model, probability: np.ndarray) -> np.ndarray:
    if model is None:
        return np.asarray(probability, dtype=float)
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return model.predict_proba(logit)[:, 1]


def binary_metrics(y: np.ndarray, pred: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "positive": int(y.sum()), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else math.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else math.nan,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else math.nan,
        "pr_auc": float(average_precision_score(y, probability)) if int(y.sum()) else math.nan,
        "brier": float(brier_score_loss(y, probability)) if np.nanmin(probability) >= 0 and np.nanmax(probability) <= 1 else math.nan,
    }


def run_smoke() -> Dict[str, object]:
    df = load_data()
    outer = balanced_group_folds(df, n_splits=4, seed=SEED, repeats=2000, min_positive_patients=2)
    train_idx, test_idx = outer[0]
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    inner = balanced_group_folds(train, n_splits=3, seed=SEED + 1000, repeats=500, min_positive_patients=1)
    label = "T18"
    oof = np.full(len(train), np.nan)
    for inner_id, (fit_idx, val_idx) in enumerate(inner):
        fit = train.iloc[fit_idx]
        val = train.iloc[val_idx]
        x_fit, medians = build_feature_frame(fit)
        x_val, _ = build_feature_frame(val, medians)
        y_fit = fit[f"y_{label}"].to_numpy(int)
        model = make_model(SEED + inner_id, n_estimators=80)
        model.fit(x_fit, y_fit, sample_weight=patient_equal_weights(fit, y_fit))
        oof[val_idx] = model.predict_proba(x_val)[:, 1]
    if not np.isfinite(oof).all():
        raise RuntimeError("内层OOF预测不完整")
    y_train = train[f"y_{label}"].to_numpy(int)
    platt = fit_platt(y_train, oof)
    oof_cal = apply_platt(platt, oof)
    threshold, threshold_info = select_threshold(y_train, oof_cal)
    x_train, medians = build_feature_frame(train)
    x_test, _ = build_feature_frame(test, medians)
    final_model = make_model(SEED + 99, n_estimators=80)
    final_model.fit(x_train, y_train, sample_weight=patient_equal_weights(train, y_train))
    test_probability = apply_platt(platt, final_model.predict_proba(x_test)[:, 1])
    test_pred = (test_probability >= threshold).astype(int)
    result = {
        "mode": "smoke",
        "input_sha256": file_sha256(INPUT),
        "rows": len(df),
        "patients": int(df["孕妇代码"].nunique()),
        "label": label,
        "outer_fold": 0,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_patients": int(train["孕妇代码"].nunique()),
        "test_patients": int(test["孕妇代码"].nunique()),
        "threshold": threshold,
        "threshold_selection": threshold_info,
        "platt_used": platt is not None,
        "test_metrics": binary_metrics(test[f"y_{label}"].to_numpy(int), test_pred, test_probability),
        "probability_range": [float(test_probability.min()), float(test_probability.max())],
        "feature_count": int(x_train.shape[1]),
        "seed": SEED,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "q4_smoke_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="运行P1最小真实数据纵向切片")
    args = parser.parse_args()
    if not args.smoke:
        raise SystemExit("全量模式将在P1门禁通过后启用；当前请使用 --smoke")
    result = run_smoke()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
