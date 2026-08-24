from __future__ import annotations

import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "C题" / "C题" / "附件.xlsx"
OUT = ROOT / "C题" / "output" / "archive" / "legacy_analysis" / "机器学习扩展"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfcfe",
        "axes.edgecolor": "#667085",
        "axes.titleweight": "bold",
        "grid.alpha": 0.30,
    }
)


def parse_week(value) -> float:
    m = re.fullmatch(r"\s*(\d+)[wW](?:\+(\d+))?\s*", str(value))
    return float(m.group(1)) + float(m.group(2) or 0) / 7.0


def numeric_count(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value)
    m = re.search(r"\d+", s)
    return float(m.group()) if m else np.nan


def prepare(sheet: str) -> pd.DataFrame:
    d = pd.read_excel(SOURCE, sheet_name=sheet)
    d["孕周"] = d["检测孕周"].map(parse_week)
    d["IVF"] = (d["IVF妊娠"] != "自然受孕").astype(float)
    d["孕次"] = d["怀孕次数"].map(numeric_count)
    d["产次"] = d["生产次数"].map(numeric_count)
    d["log原始读段"] = np.log1p(d["原始读段数"])
    unique_col = "唯一比对的读段数  " if "唯一比对的读段数  " in d.columns else "唯一比对的读段数"
    d["log唯一读段"] = np.log1p(d[unique_col])
    return d


BIO_FEATURES = ["孕周", "孕妇BMI", "年龄", "身高", "IVF", "孕次", "产次"]
QC_FEATURES = [
    "log原始读段",
    "在参考基因组上比对的比例",
    "重复读段的比例",
    "log唯一读段",
    "GC含量",
    "被过滤掉读段数的比例",
    "X染色体浓度",
]
FEATURE_LABELS = {
    "孕周": "孕周",
    "孕妇BMI": "BMI",
    "年龄": "年龄",
    "身高": "身高",
    "IVF": "IVF",
    "孕次": "孕次",
    "产次": "产次",
    "log原始读段": "log原始读段",
    "在参考基因组上比对的比例": "比对率",
    "重复读段的比例": "重复率",
    "log唯一读段": "log唯一读段",
    "GC含量": "GC含量",
    "被过滤掉读段数的比例": "过滤率",
    "X染色体浓度": "X浓度",
    "13号染色体的Z值": "Z13",
    "18号染色体的Z值": "Z18",
    "21号染色体的Z值": "Z21",
    "X染色体的Z值": "ZX",
    "13号染色体的GC含量": "GC13",
    "18号染色体的GC含量": "GC18",
    "21号染色体的GC含量": "GC21",
}


def regression_estimators():
    elastic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", ElasticNet(max_iter=20000, random_state=2025)),
        ]
    )
    elastic_grid = {
        "model__alpha": [0.002, 0.01, 0.05, 0.15],
        "model__l1_ratio": [0.15, 0.5, 0.85],
    }
    hgb = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=220,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=2025,
                ),
            ),
        ]
    )
    hgb_grid = {
        "model__max_leaf_nodes": [7, 15],
        "model__min_samples_leaf": [25, 45],
        "model__l2_regularization": [1.0, 5.0],
    }
    return {"ElasticNet": (elastic, elastic_grid), "HGBR": (hgb, hgb_grid)}


def classifier_estimators(class_weight=None):
    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    max_iter=5000,
                    class_weight=class_weight,
                    random_state=2025,
                ),
            ),
        ]
    )
    logistic_grid = {"model__C": [0.03, 0.1, 0.3, 1.0]}
    hgb = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=180,
                    early_stopping=True,
                    validation_fraction=0.15,
                    class_weight=class_weight,
                    random_state=2025,
                ),
            ),
        ]
    )
    hgb_grid = {
        "model__max_leaf_nodes": [7, 15],
        "model__min_samples_leaf": [25, 45],
        "model__l2_regularization": [1.0, 5.0],
    }
    return {"RegularizedLogit": (logistic, logistic_grid), "HGBC": (hgb, hgb_grid)}


def nested_regression(d: pd.DataFrame, features: list[str], model_name: str) -> dict:
    X = d[features]
    y = logit(np.clip(d["Y染色体浓度"].to_numpy(float), 1e-5, 1 - 1e-5))
    groups = d["孕妇代码"].to_numpy()
    estimator, grid = regression_estimators()[model_name]
    outer = GroupKFold(5)
    pred = np.full(len(d), np.nan)
    importances = []
    best_params = []
    for train, test in outer.split(X, y, groups):
        search = GridSearchCV(
            clone(estimator), grid, scoring="neg_root_mean_squared_error", cv=GroupKFold(3), n_jobs=-1
        )
        search.fit(X.iloc[train], y[train], groups=groups[train])
        pred[test] = search.predict(X.iloc[test])
        best_params.append(search.best_params_)
        pi = permutation_importance(
            search.best_estimator_, X.iloc[test], y[test], scoring="neg_root_mean_squared_error",
            n_repeats=8, random_state=2025, n_jobs=-1,
        )
        importances.append(pi.importances_mean)
    final_search = GridSearchCV(
        clone(estimator), grid, scoring="neg_root_mean_squared_error", cv=GroupKFold(5), n_jobs=-1
    )
    final_search.fit(X, y, groups=groups)
    return {
        "model": model_name,
        "features": features,
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "pred": pred,
        "importance_mean": np.mean(importances, axis=0),
        "importance_sd": np.std(importances, axis=0),
        "best_params_frequency": dict(Counter(json.dumps(v, sort_keys=True) for v in best_params)),
        "final_best_params": final_search.best_params_,
        "final_model": final_search.best_estimator_,
    }


def nested_classifier(
    d: pd.DataFrame,
    features: list[str],
    y: np.ndarray,
    model_name: str,
    scoring: str,
    class_weight=None,
    woman_level: bool = False,
) -> dict:
    X = d[features]
    groups = d["孕妇代码"].to_numpy()
    estimator, grid = classifier_estimators(class_weight)[model_name]
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=2025)
    pred = np.full(len(d), np.nan)
    importances = []
    best_params = []
    fold_models = []
    for fold, (train, test) in enumerate(outer.split(X, y, groups)):
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=2025 + fold)
        search = GridSearchCV(clone(estimator), grid, scoring=scoring, cv=inner, n_jobs=-1)
        search.fit(X.iloc[train], y[train], groups=groups[train])
        pred[test] = search.predict_proba(X.iloc[test])[:, 1]
        best_params.append(search.best_params_)
        fold_models.append(search.best_estimator_)
        pi = permutation_importance(
            search.best_estimator_, X.iloc[test], y[test], scoring="neg_brier_score",
            n_repeats=8, random_state=2025 + fold, n_jobs=-1,
        )
        importances.append(pi.importances_mean)
    final_inner = StratifiedGroupKFold(5, shuffle=True, random_state=2030)
    final_search = GridSearchCV(clone(estimator), grid, scoring=scoring, cv=final_inner, n_jobs=-1)
    final_search.fit(X, y, groups=groups)

    if woman_level:
        eval_df = pd.DataFrame({"group": groups, "y": y, "p": pred})
        agg = eval_df.groupby("group").agg(y=("y", "max"), p=("p", "max"))
        y_eval, p_eval = agg["y"].to_numpy(), agg["p"].to_numpy()
        eval_groups = agg.index.to_numpy()
    else:
        y_eval, p_eval = y, pred
        eval_groups = groups
    auc_samples = []
    rng = np.random.default_rng(2025)
    unique_groups = np.unique(eval_groups)
    group_rows = {g: np.flatnonzero(eval_groups == g) for g in unique_groups}
    for _ in range(600):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        rows = np.concatenate([group_rows[g] for g in sampled])
        if np.unique(y_eval[rows]).size == 2:
            auc_samples.append(roc_auc_score(y_eval[rows], p_eval[rows]))
    return {
        "model": model_name,
        "features": features,
        "roc_auc": float(roc_auc_score(y_eval, p_eval)),
        "roc_auc_cluster_bootstrap_95ci": [float(v) for v in np.quantile(auc_samples, [0.025, 0.975])],
        "pr_auc": float(average_precision_score(y_eval, p_eval)),
        "brier": float(brier_score_loss(y_eval, p_eval)),
        "log_loss": float(log_loss(y_eval, np.clip(p_eval, 1e-6, 1 - 1e-6))),
        "pred": pred,
        "eval_y": y_eval,
        "eval_p": p_eval,
        "importance_mean": np.mean(importances, axis=0),
        "importance_sd": np.std(importances, axis=0),
        "best_params_frequency": dict(Counter(json.dumps(v, sort_keys=True) for v in best_params)),
        "final_best_params": final_search.best_params_,
        "final_model": final_search.best_estimator_,
        "fold_models": fold_models,
    }


def timing_from_model(model, women: pd.DataFrame, features: list[str], cut: float = 34.5) -> list[dict]:
    grid = np.arange(10.0, 25.0 + 1 / 7, 1 / 7)
    out = []
    for idx, (label, mask) in enumerate(
        [("BMI<34.5", women["孕妇BMI"] < cut), ("BMI≥34.5", women["孕妇BMI"] >= cut)], start=1
    ):
        base = women.loc[mask, features].copy()
        probs = []
        for week in grid:
            x = base.copy()
            x["孕周"] = week
            probs.append(float(model.predict_proba(x)[:, 1].mean()))
        probs = np.asarray(probs)
        hits = np.flatnonzero(probs >= 0.90)
        t90 = float(grid[hits[0]]) if len(hits) else None
        out.append(
            {
                "组": idx,
                "BMI组": label,
                "人数": int(mask.sum()),
                "ML_90%时点": t90,
                "25周达标概率": float(probs[-1]),
                "grid": grid,
                "prob": probs,
            }
        )
    return out


def plot_male_results(d: pd.DataFrame, regressions: list[dict], classifiers: list[dict], timing: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)
    fig.suptitle("C题男胎：可解释机器学习与统计基线比较", fontsize=18, weight="bold")

    ax = axes[0, 0]
    names = [r["model"] + ("+QC" if len(r["features"]) > len(BIO_FEATURES) else "") for r in regressions]
    vals = [r["rmse"] for r in regressions]
    bars = ax.bar(names, vals, color=["#667085", "#2e90a5", "#f79009"][: len(vals)])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(title="Y浓度 logit 回归：患者级交叉验证", ylabel="RMSE（越低越好）")
    ax.grid(True, axis="y")

    best_reg = min(regressions, key=lambda x: x["rmse"])
    order = np.argsort(best_reg["importance_mean"])[-8:]
    ax = axes[0, 1]
    ax.barh(
        [FEATURE_LABELS.get(best_reg["features"][i], best_reg["features"][i]) for i in order],
        best_reg["importance_mean"][order], xerr=best_reg["importance_sd"][order],
        color="#2e90a5", alpha=0.88,
    )
    ax.set(title=f"最佳回归模型特征重要性：{best_reg['model']}", xlabel="置换后 RMSE 增量")
    ax.grid(True, axis="x")

    ax = axes[1, 0]
    for res, color in zip(classifiers, ["#667085", "#2e90a5", "#f79009"]):
        frac, mean = calibration_curve(res["eval_y"], res["eval_p"], n_bins=8, strategy="quantile")
        ax.plot(mean, frac, "o-", lw=1.8, label=f"{res['model']} (Brier={res['brier']:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "--", color="#344054")
    ax.set(title="4%达标概率校准", xlabel="预测概率", ylabel="实际达标比例", xlim=(0, 1), ylim=(0, 1))
    ax.grid(True)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    for item, color in zip(timing, ["#2e90a5", "#f79009"]):
        ax.plot(item["grid"], item["prob"], lw=2.2, label=item["BMI组"], color=color)
        if item["ML_90%时点"] is not None:
            ax.axvline(item["ML_90%时点"], color=color, alpha=0.45)
    ax.axhline(0.9, ls="--", color="#344054")
    ax.set(title="梯度提升模型的达标概率曲线", xlabel="孕周 / 周", ylabel="平均达标概率", ylim=(0, 1.02))
    ax.grid(True)
    ax.legend(frameon=False)
    fig.savefig(OUT / "男胎_模型比较_重要性_校准_时点.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Two-dimensional partial dependence is the most relevant hidden interaction.
    best_hgb = next((r for r in regressions if r["model"] == "HGBR" and len(r["features"]) == len(BIO_FEATURES)), None)
    if best_hgb is not None:
        fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
        PartialDependenceDisplay.from_estimator(
            best_hgb["final_model"], d[best_hgb["features"]], [("孕周", "孕妇BMI")],
            grid_resolution=35, ax=ax,
        )
        ax.set_title("男胎 Y 浓度：孕周×BMI 二维部分依赖")
        fig.savefig(OUT / "男胎_孕周_BMI_二维部分依赖.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def female_features() -> list[str]:
    return BIO_FEATURES + [
        "13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值",
        "log原始读段", "在参考基因组上比对的比例", "重复读段的比例", "log唯一读段", "GC含量",
        "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量", "被过滤掉读段数的比例",
        "X染色体浓度",
    ]


def plot_female_results(results: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)
    fig.suptitle("C题女胎：AB标签的机器学习检验与解释", fontsize=18, weight="bold")

    ax = axes[0, 0]
    targets = list(results)
    x = np.arange(len(targets))
    width = 0.34
    for j, model in enumerate(["RegularizedLogit", "HGBC"]):
        vals = [next(r for r in results[t] if r["model"] == model)["roc_auc"] for t in targets]
        ax.bar(x + (j - 0.5) * width, vals, width, label=model, color=["#667085", "#2e90a5"][j])
    ax.axhline(0.5, ls="--", color="#d92d20")
    ax.set_xticks(x, targets)
    ax.set_ylim(0.4, 0.75)
    ax.set(title="孕妇级 ROC-AUC", ylabel="ROC-AUC")
    ax.grid(True, axis="y")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for target, color in zip(targets, ["#2e90a5", "#f79009", "#7f56a3"]):
        best = max(results[target], key=lambda r: r["roc_auc"])
        frac, mean = calibration_curve(best["eval_y"], best["eval_p"], n_bins=5, strategy="quantile")
        ax.plot(mean, frac, "o-", label=f"{target} {best['model']}", color=color)
    ax.plot([0, 1], [0, 1], "--", color="#344054")
    ax.set(title="各标签最佳模型的概率校准", xlabel="预测概率", ylabel="实际阳性比例", xlim=(0, 1), ylim=(0, 1))
    ax.grid(True)
    ax.legend(frameon=False, fontsize=9)

    for ax, target in zip(axes[1], targets[:2]):
        best = max(results[target], key=lambda r: r["roc_auc"])
        order = np.argsort(best["importance_mean"])[-9:]
        ax.barh(
            [FEATURE_LABELS.get(best["features"][i], best["features"][i]) for i in order],
            best["importance_mean"][order], xerr=best["importance_sd"][order],
            color="#2e90a5" if target == "T13" else "#f79009", alpha=0.88,
        )
        ax.set(title=f"{target}最佳模型重要性", xlabel="置换后 Brier 损失增量")
        ax.grid(True, axis="x")
    fig.savefig(OUT / "女胎_模型比较_校准_重要性.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # T21 is shown separately because it has the fewest positive women.
    target = "T21"
    best = max(results[target], key=lambda r: r["roc_auc"])
    order = np.argsort(best["importance_mean"])[-10:]
    fig, ax = plt.subplots(figsize=(7.8, 5.7), constrained_layout=True)
    ax.barh(
        [FEATURE_LABELS.get(best["features"][i], best["features"][i]) for i in order],
        best["importance_mean"][order], xerr=best["importance_sd"][order], color="#7f56a3", alpha=0.88,
    )
    ax.set(title=f"T21最佳模型特征重要性（{best['model']}）", xlabel="置换后 Brier 损失增量")
    ax.grid(True, axis="x")
    fig.savefig(OUT / "女胎_T21_特征重要性.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def serializable(res: dict) -> dict:
    keep = {k: v for k, v in res.items() if k not in {"pred", "eval_y", "eval_p", "final_model", "fold_models"}}
    for key in ("importance_mean", "importance_sd"):
        if key in keep:
            keep[key] = keep[key].tolist()
    return keep


def main() -> None:
    male = prepare("男胎检测数据")
    print("[1/4] 男胎 Y 浓度回归", flush=True)
    regressions = [
        nested_regression(male, BIO_FEATURES, "ElasticNet"),
        nested_regression(male, BIO_FEATURES, "HGBR"),
        nested_regression(male, BIO_FEATURES + QC_FEATURES, "HGBR"),
    ]

    print("[2/4] 男胎 4% 达标概率", flush=True)
    y_hit = (male["Y染色体浓度"].to_numpy(float) >= 0.04).astype(int)
    male_classifiers = [
        nested_classifier(male, BIO_FEATURES, y_hit, "RegularizedLogit", "neg_log_loss"),
        nested_classifier(male, BIO_FEATURES, y_hit, "HGBC", "neg_log_loss"),
    ]
    best_timing_model = min(male_classifiers, key=lambda r: r["brier"])
    women = male.sort_values("孕周").drop_duplicates("孕妇代码")
    timing = timing_from_model(best_timing_model["final_model"], women, BIO_FEATURES)
    plot_male_results(male, regressions, male_classifiers, timing)

    print("[3/4] 女胎 AB 标签模型", flush=True)
    female = prepare("女胎检测数据")
    ffeatures = female_features()
    label_text = female["染色体的非整倍体"].fillna("").astype(str)
    female_results: dict[str, list[dict]] = {}
    for target in ("T13", "T18", "T21"):
        y = label_text.str.contains(target).astype(int).to_numpy()
        female_results[target] = [
            nested_classifier(
                female, ffeatures, y, "RegularizedLogit", "average_precision", class_weight="balanced", woman_level=True
            ),
            nested_classifier(
                female, ffeatures, y, "HGBC", "average_precision", class_weight="balanced", woman_level=True
            ),
        ]
    plot_female_results(female_results)

    print("[4/4] 导出结果", flush=True)
    summary_rows = []
    for r in regressions:
        summary_rows.append({"任务": "男胎Y浓度回归", "标签": "Y_logit", "模型": r["model"], "特征集": "BIO+QC" if len(r["features"]) > len(BIO_FEATURES) else "BIO", "RMSE": r["rmse"], "MAE": r["mae"], "R2": r["r2"]})
    for r in male_classifiers:
        summary_rows.append({"任务": "男胎达标概率", "标签": "Y>=4%", "模型": r["model"], "特征集": "BIO", "ROC_AUC": r["roc_auc"], "PR_AUC": r["pr_auc"], "Brier": r["brier"], "LogLoss": r["log_loss"]})
    for target, values in female_results.items():
        for r in values:
            summary_rows.append({"任务": "女胎AB标签", "标签": target, "模型": r["model"], "特征集": "BIO+Z+QC", "ROC_AUC": r["roc_auc"], "PR_AUC": r["pr_auc"], "Brier": r["brier"], "LogLoss": r["log_loss"]})
    pd.DataFrame(summary_rows).to_csv(OUT / "模型比较.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{k: v for k, v in t.items() if k not in {"grid", "prob"}} for t in timing]).to_csv(
        OUT / "机器学习时点.csv", index=False, encoding="utf-8-sig"
    )

    output = {
        "male_regression": [serializable(r) for r in regressions],
        "male_threshold_classification": [serializable(r) for r in male_classifiers],
        "male_timing_model": best_timing_model["model"],
        "male_timing": [{k: v for k, v in t.items() if k not in {"grid", "prob"}} for t in timing],
        "female": {target: [serializable(r) for r in values] for target, values in female_results.items()},
        "validation": "All outer and inner splits are grouped by woman code; female primary metrics aggregate repeated records to woman level.",
    }
    (OUT / "ml_results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
