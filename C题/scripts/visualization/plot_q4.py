"""问题4科研图：原始数据、建模过程、验证结果各三张。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "output" / "results" / "q4"
FIGURES = PROJECT_ROOT / "output" / "figures"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from utils.plot_style import PALETTE, add_panel_labels, apply_publication_style, publication_subplots

EXPORTER = Path(r"C:\Users\admin\.codex\skills\math-modeling\tools\figure\scripts\export_figure.py")
spec = importlib.util.spec_from_file_location("q4_export_figure", EXPORTER)
export_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(export_module)
export_figure = export_module.export_figure

LABELS = ("T13", "T18", "T21")
COLORS = [PALETTE["primary"], PALETTE["secondary"], PALETTE["positive"]]

CONTRACTS = {
    "raw_q4_label_distribution": {"type": "raw", "claim": "AB标签极度不平衡", "data": "q4_profile_input.csv", "statistic": "记录计数与占比", "size_in": [6.3, 3.9]},
    "raw_q4_zscore_distribution": {"type": "raw", "claim": "固定Z阈值与AB标签分离有限", "data": "q4_profile_input.csv", "statistic": "箱线图与全部散点", "size_in": [6.3, 3.1]},
    "raw_q4_quality_relationships": {"type": "raw", "claim": "测序指标与Z值存在相关结构", "data": "q4_profile_input.csv", "statistic": "Pearson相关", "size_in": [6.3, 4.7]},
    "process_q4_fold_balance": {"type": "process", "claim": "四折均覆盖少数类患者", "data": "q4_fold_audit.csv", "statistic": "患者计数", "size_in": [6.3, 3.1]},
    "process_q4_model_selection": {"type": "process", "claim": "复杂度胜负随标签和折变化", "data": "q4_model_selection.csv", "statistic": "内层PR-AUC", "size_in": [6.3, 2.8]},
    "process_q4_threshold_tradeoff": {"type": "process", "claim": "高灵敏度阈值牺牲特异度", "data": "q4_threshold_sensitivity.csv", "statistic": "灵敏度/特异度/F2", "size_in": [6.3, 3.9]},
    "result_q4_pr_curves": {"type": "result", "claim": "T18与任一异常排序能力最强", "data": "q4_outer_predictions.csv", "statistic": "外层OOF PR曲线与AP", "size_in": [6.3, 3.9]},
    "result_q4_metrics_comparison": {"type": "result", "claim": "多变量模型总体优于固定Z规则", "data": "q4_outer_metrics.csv", "statistic": "四折均值", "size_in": [6.3, 3.1]},
    "result_q4_feature_importance": {"type": "result", "claim": "质量、GC和BMI共同贡献", "data": "q4_feature_importance.csv", "statistic": "12个模型均值与SD", "size_in": [6.3, 4.5]},
    "result_q4_calibration": {"type": "result", "claim": "概率校准质量存在标签差异", "data": "q4_outer_predictions.csv", "statistic": "5等频箱与Brier", "size_in": [6.3, 3.9]},
}


def save(fig, name, size=(6.3, 3.9)):
    export_figure(fig, str(FIGURES / name), formats=("svg", "png"), dpi=300,
                  size_inches=size, grayscale_preview=True)
    gray = FIGURES / f"{name}_grayscale.png"
    if gray.exists():
        with Image.open(gray) as im:
            im.save(gray, dpi=(300, 300))
        gray_dir = FIGURES / "grayscale"
        gray_dir.mkdir(parents=True, exist_ok=True)
        gray.replace(gray_dir / gray.name)
    plt.close(fig)


def raw_figures(profile: pd.DataFrame):
    counts = [int(profile[f"y_{x}"].sum()) for x in LABELS] + [int(profile["y_any"].sum())]
    total = len(profile)
    fig, ax = publication_subplots(width="report", aspect=0.55)
    names = [*LABELS, "任一异常"]
    y = np.arange(4)
    ax.barh(y, counts, color=[*COLORS, PALETTE["contrast"]], label="阳性")
    ax.barh(y, np.array([total] * 4) - counts, left=counts, color="#D9D9D9", label="阴性")
    for i, value in enumerate(counts):
        ax.text(value + 5, i, f"{value} ({value/total:.1%})", va="center", fontsize=7)
    ax.set_yticks(y, names); ax.set_xlabel("记录数"); ax.set_title("异常标签分布（n=605）")
    ax.legend(frameon=False, ncol=2, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "raw_q4_label_distribution")

    fig, axes = publication_subplots(1, 3, width="report", aspect=0.48)
    for ax, lab, color in zip(axes, LABELS, COLORS):
        chrom = lab[1:]
        z = profile[f"{chrom}号染色体的Z值"].abs()
        groups = [z[profile[f"y_{lab}"] == 0], z[profile[f"y_{lab}"] == 1]]
        bp = ax.boxplot(groups, tick_labels=["阴性", "阳性"], widths=.55, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], ["#D9D9D9", color]): patch.set_facecolor(c)
        rng = np.random.default_rng(100 + int(chrom))
        for j, values in enumerate(groups, 1):
            ax.scatter(rng.normal(j, .045, len(values)), values, s=4, alpha=.22, color="#333333", linewidth=0)
        ax.axhline(3, ls="--", color=PALETTE["contrast"], lw=1, label="|Z|=3")
        ax.set_title(lab); ax.set_ylabel("|Z值|" if lab == "T13" else "")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].legend(frameon=False, loc="upper right")
    add_panel_labels(axes)
    save(fig, "raw_q4_zscore_distribution", (6.3, 3.1))

    cols = ["孕周_周", "孕妇BMI", "GC含量", "原始读段数", "在参考基因组上比对的比例", "重复读段的比例",
            "13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值"]
    short = ["孕周", "BMI", "GC", "原始读段", "比对率", "重复率", "Z13", "Z18", "Z21", "ZX"]
    corr = profile[cols].corr().to_numpy()
    fig, ax = publication_subplots(width="report", aspect=.75)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(short)), short, rotation=45, ha="right"); ax.set_yticks(range(len(short)), short)
    for i in range(len(short)):
        for j in range(len(short)):
            if abs(corr[i, j]) >= .35: ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, shrink=.72, label="Pearson r"); ax.set_title("质量指标与染色体Z值相关结构")
    save(fig, "raw_q4_quality_relationships", (6.3, 4.7))


def process_figures(audit, selection, curves):
    outer = audit[audit.level == "outer"].sort_values("fold")
    fig, axes = publication_subplots(1, 2, width="report", aspect=.48)
    axes[0].bar(outer.fold + 1, outer.test_patients, color=PALETTE["primary"])
    axes[0].set(xlabel="外层折", ylabel="测试患者数", title="患者规模")
    bottom = np.zeros(len(outer))
    for lab, color in zip(LABELS, COLORS):
        val = outer[f"test_positive_patients_{lab}"].to_numpy()
        axes[1].bar(outer.fold + 1, val, bottom=bottom, color=color, label=lab)
        bottom += val
    axes[1].set(xlabel="外层折", ylabel="阳性患者计数（可重叠）", title="少数类覆盖")
    axes[1].legend(frameon=False, ncol=3)
    for ax in axes: ax.set_xticks([1, 2, 3, 4]); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "process_q4_fold_balance", (6.3, 3.1))

    fig, axes = publication_subplots(1, 4, width="report", aspect=.42)
    order = ["A_shallow", "B_conservative", "C_slow"]
    for ax, lab in zip(axes, (*LABELS, "any")):
        part = selection[selection.label == lab]
        for fold, g in part.groupby("outer_fold"):
            g = g.set_index("candidate").loc[order]
            ax.scatter(range(3), g.inner_pr_auc, alpha=.72, s=20, label=f"折{fold+1}")
        ax.set_xticks(range(3), ["A", "B", "C"]); ax.set_title(lab if lab != "any" else "任一异常")
        ax.set_xlabel("候选复杂度"); ax.set_ylabel("内层PR-AUC" if lab == "T13" else "")
        ax.spines[["top", "right"]].set_visible(False)
    add_panel_labels(axes)
    save(fig, "process_q4_model_selection", (6.3, 2.8))

    part = curves[(curves.label == "any") & (curves.source == "xgb_inner_oof")].copy()
    part["q"] = part.groupby("outer_fold")["threshold"].rank(pct=True)
    part["bin"] = pd.cut(part.q, bins=np.linspace(0, 1, 31), labels=False, include_lowest=True)
    mean = part.groupby("bin")[["threshold", "sensitivity", "specificity", "f2"]].mean().reset_index()
    fig, ax = publication_subplots(width="report", aspect=.58)
    for metric, color, style in [("sensitivity", PALETTE["contrast"], "-"), ("specificity", PALETTE["primary"], "--"), ("f2", PALETTE["positive"], "-.")]:
        ax.plot(mean.threshold, mean[metric], color=color, ls=style, label=metric)
    ax.set(xlabel="候选阈值（校准概率）", ylabel="指标值", title="任一异常模型的阈值权衡（训练折内）")
    ax.set_ylim(0, 1.03); ax.legend(frameon=False, ncol=3); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "process_q4_threshold_tradeoff")


def result_figures(pred, metrics, importance):
    fig, ax = publication_subplots(width="report", aspect=.62)
    for lab, color in zip((*LABELS, "any"), [*COLORS, PALETTE["contrast"]]):
        y = pred[f"y_{lab}"] if lab != "any" else pred.y_any
        p = pred[f"prob_{lab}"]
        precision, recall, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(recall, precision, color=color, label=f"{lab if lab != 'any' else '任一异常'} (AP≈{ap:.3f})")
        ax.axhline(y.mean(), color=color, lw=.5, ls=":", alpha=.55)
    ax.set(xlabel="灵敏度（Recall）", ylabel="精确率（Precision）", title="外层留出预测的PR曲线")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "result_q4_pr_curves")

    fig, ax = publication_subplots(width="report", aspect=.62)
    for lab, color in zip((*LABELS, "any"), [*COLORS, PALETTE["contrast"]]):
        y = pred[f"y_{lab}"] if lab != "any" else pred.y_any
        p = pred[f"prob_{lab}"]
        observed, estimated = calibration_curve(y, p, n_bins=5, strategy="quantile")
        brier = np.mean((y.to_numpy(float) - p.to_numpy(float)) ** 2)
        ax.plot(estimated, observed, marker="o", color=color,
                label=f"{lab if lab != 'any' else '任一异常'} (Brier={brier:.3f})")
    ax.plot([0, 1], [0, 1], color="#777777", ls="--", lw=1, label="理想校准")
    ax.set(xlabel="平均预测概率", ylabel="实际阳性率", title="外层留出概率校准")
    ax.set_xlim(0, .6); ax.set_ylim(0, .6); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "result_q4_calibration")

    view = metrics[metrics.label.isin(LABELS)].groupby(["label", "method"])[["sensitivity", "f2"]].mean().reset_index()
    fig, axes = publication_subplots(1, 2, width="report", aspect=.48)
    methods = ["xgboost", "z_tuned", "z_fixed_3"]
    mnames = ["多变量模型", "训练折Z阈值", "固定|Z|≥3"]
    x = np.arange(3); width = .23
    for i, (method, name, color) in enumerate(zip(methods, mnames, [PALETTE["primary"], PALETTE["secondary"], "#999999"])):
        g = view[view.method == method].set_index("label").reindex(LABELS)
        axes[0].bar(x + (i-1)*width, g.sensitivity, width, label=name, color=color)
        axes[1].bar(x + (i-1)*width, g.f2, width, label=name, color=color)
        for panel, values in zip(axes, (g.sensitivity.to_numpy(), g.f2.to_numpy())):
            for xpos, value in zip(x + (i-1)*width, values):
                if np.isfinite(value) and value == 0:
                    panel.text(xpos, .008, "0", ha="center", va="bottom", fontsize=6, color=color)
    for ax, title in zip(axes, ["灵敏度", "F2（更重视漏检）"]):
        ax.set_xticks(x, LABELS); ax.set_ylim(0, 1); ax.set_title(title); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("四折均值"); axes[1].legend(frameon=False, fontsize=6)
    add_panel_labels(axes)
    save(fig, "result_q4_metrics_comparison", (6.3, 3.1))

    imp = importance[importance.label.isin(LABELS)].groupby("feature").importance.agg(["mean", "std"]).sort_values("mean").tail(12)
    fig, ax = publication_subplots(width="report", aspect=.72)
    ax.barh(np.arange(len(imp)), imp["mean"], xerr=imp["std"], color=PALETTE["primary"], alpha=.9,
            error_kw={"elinewidth": .7, "capsize": 2})
    ax.set_yticks(np.arange(len(imp)), imp.index); ax.set_xlabel("平均特征重要性（误差线：染色体×外层折SD）")
    ax.set_title("T13/T18/T21模型的主要变量"); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "result_q4_feature_importance", (6.3, 4.5))


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    apply_publication_style("zh", "report")
    profile = pd.read_csv(RESULTS / "q4_profile_input.csv")
    audit = pd.read_csv(RESULTS / "q4_fold_audit.csv")
    selection = pd.read_csv(RESULTS / "q4_model_selection.csv")
    curves = pd.read_csv(RESULTS / "q4_threshold_sensitivity.csv")
    pred = pd.read_csv(RESULTS / "q4_outer_predictions.csv")
    metrics = pd.read_csv(RESULTS / "q4_outer_metrics.csv")
    importance = pd.read_csv(RESULTS / "q4_feature_importance.csv")
    raw_figures(profile); process_figures(audit, selection, curves); result_figures(pred, metrics, importance)
    (RESULTS / "q4_figure_contracts.json").write_text(json.dumps(CONTRACTS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"created {len(CONTRACTS)} figure families")


if __name__ == "__main__":
    main()
