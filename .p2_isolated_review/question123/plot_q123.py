#!/usr/bin/env python3
"""问题1—3九张证据图：原始数据、建模过程、最终结果各三张。

结果图读取 C题/results/q1—q3 的冻结表；原始数据图允许读取已校验哈希的附件，
统一导出 PNG、SVG 和灰度预览。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT))
from utils.plot_style import (  # noqa: E402
    PALETTE,
    add_panel_labels,
    apply_publication_style,
    export_figure,
    publication_subplots,
)

OUT = PROJECT / "figures" / "q123"
Q1 = PROJECT / "results" / "q1"
Q2 = PROJECT / "results" / "q2"
Q3 = PROJECT / "results" / "q3"
apply_publication_style("zh", "report")


def save(fig, name: str) -> None:
    export_figure(fig, OUT / name, dpi=300, grayscale_preview=True)
    plt.close(fig)


def raw_q1() -> None:
    d = pd.read_csv(Q1 / "q1_record_fit.csv")
    fig, ax = publication_subplots(aspect=0.62)
    pts = ax.scatter(d["孕周数"], d["Y染色体浓度"] * 100, c=d["孕妇BMI"],
                     s=8, alpha=0.38, cmap="viridis", linewidths=0)
    cb = fig.colorbar(pts, ax=ax, pad=0.02)
    cb.set_label("记录级BMI (kg/m²)")
    ax.axhline(4, color=PALETTE["contrast"], ls="--", lw=1, label="4%达标线")
    ax.set(xlabel="检测孕周（周）", ylabel="Y染色体浓度（%）", title="孕周、BMI与Y浓度")
    ax.legend(loc="upper left")
    save(fig, "raw_q1_y_week_bmi")


def raw_q2() -> None:
    d = pd.read_csv(Q2 / "男胎患者级精简数据.csv")
    cut = 34.357
    fig, ax = publication_subplots(aspect=0.58)
    ax.hist(d["孕妇BMI"], bins=np.arange(20, 48, 1), color=PALETTE["sky"], edgecolor="white")
    ax.axvline(cut, color=PALETTE["contrast"], lw=1.4, label=f"Optimal Binning切点 {cut:.3f}")
    ax.set(xlabel="患者级BMI中位数 (kg/m²)", ylabel="孕妇人数", title="BMI切点与分布")
    ax.legend()
    save(fig, "raw_q2_bmi_distribution")


def raw_q3() -> None:
    raw = pd.read_excel(PROJECT / "C题" / "附件.xlsx", sheet_name=0)
    def parse_count(value):
        match = re.search(r"\d+", str(value)) if pd.notna(value) else None
        return float(match.group()) if match else np.nan
    raw["孕次"] = raw["怀孕次数"].map(parse_count)
    raw["产次"] = raw["生产次数"].map(parse_count)
    cols = ["年龄", "身高", "体重", "孕妇BMI", "孕次", "产次"]
    patient = raw.groupby("孕妇代码", as_index=False)[cols].median(numeric_only=True)
    corr = patient[cols].corr().to_numpy()
    labels = ["年龄", "身高", "体重", "BMI", "孕次", "产次"]
    fig, ax = publication_subplots(aspect=0.74)
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    cb = fig.colorbar(image, ax=ax, pad=0.02)
    cb.set_label("Pearson相关系数")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set(title="患者级原始协变量相关")
    save(fig, "raw_q3_feature_evidence")


def process_q1() -> None:
    d = pd.read_csv(Q1 / "q1_record_fit.csv")
    r = d["基础模型残差_logit"].to_numpy()
    f = d["基础模型拟合_logit"].to_numpy()
    fig, axes = publication_subplots(1, 2, aspect=0.47)
    axes[0].scatter(f, r, s=7, alpha=0.28, color=PALETTE["primary"], linewidths=0)
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set(xlabel="拟合值（logit）", ylabel="残差（logit）", title="残差—拟合值")
    (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
    axes[1].scatter(osm, osr, s=7, alpha=0.35, color=PALETTE["secondary"], linewidths=0)
    axes[1].plot(osm, slope * osm + intercept, color="black", lw=0.8)
    axes[1].set(xlabel="正态理论分位数", ylabel="残差分位数", title="残差Q-Q图")
    add_panel_labels(axes)
    save(fig, "process_q1_diagnostics")


def process_q2() -> None:
    d = pd.read_csv(Q2 / "问题2核心方案对比.csv")
    fig, ax = publication_subplots(aspect=0.62)
    x = np.arange(len(d))
    metrics = [("Bootstrap安全率", "Bootstrap安全率", "o"),
               ("重复折安全率_y", "重复折安全率", "s"),
               ("重复4_4比例", "重复4/4比例", "^")]
    for offset, (col, label, marker) in zip((-0.08, 0.0, 0.08), metrics):
        ax.scatter(x + offset, d[col], marker=marker, label=label, s=28)
    ax.axhline(0.95, color=PALETTE["neutral"], ls=":", lw=0.8)
    ax.set_xticks(x, d["方案"])
    ax.set_ylim(0.35, 1.03)
    ax.set(xlabel="固定时点方案（低BMI/高BMI）", ylabel="安全比例", title="三方案稳定性")
    ax.legend(ncols=3, loc="lower right")
    save(fig, "process_q2_policy_stability")


def process_q3() -> None:
    d = pd.read_csv(Q3 / "问题3嵌套外层独立审计.csv")
    fig, axes = publication_subplots(1, 2, aspect=0.48)
    x = d["outer_fold"] + 1
    axes[0].scatter(x, d["测试平均检测孕周"], marker="o", color=PALETTE["primary"], s=28)
    for xi, week, groups in zip(x, d["测试平均检测孕周"], d["组数"]):
        axes[0].annotate(f"{groups}组", (xi, week), xytext=(0, 5), textcoords="offset points", ha="center")
    axes[0].set(xlabel="外层测试折", ylabel="测试平均推荐孕周", title="折内独立选模")
    axes[0].set_xticks(x)
    axes[1].scatter(x - 0.04, d["测试审计层最低均值"], marker="o", label="审计层最低均值", s=28)
    axes[1].scatter(x + 0.04, d["测试审计层最低保守下限"], marker="s", label="审计层最低LCB", s=28)
    axes[1].axhline(0.90, color=PALETTE["contrast"], ls="--", lw=0.8, label="可靠性门槛")
    axes[1].set(xlabel="外层测试折", ylabel="安全性指标", title="未见孕妇审计")
    axes[1].set_xticks(x)
    axes[1].set_ylim(0.88, 0.97)
    axes[1].legend()
    add_panel_labels(axes)
    save(fig, "process_q3_group_selection")


def result_q1() -> None:
    d = pd.read_csv(Q1 / "q1_model_coefficients.csv")
    d = d[(d["模型"] == "基础模型") & (d["变量"] != "Intercept")].copy()
    labels = {"week_c": "孕周", "week_c2": "孕周²", "bmi_c": "BMI", "week_c:bmi_c": "孕周×BMI"}
    d["标签"] = d["变量"].map(labels)
    y = np.arange(len(d))[::-1]
    fig, ax = publication_subplots(aspect=0.58)
    ax.errorbar(d["系数"], y, xerr=[d["系数"] - d["95%CI下限"], d["95%CI上限"] - d["系数"]],
                fmt="o", capsize=3, color=PALETTE["primary"])
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(y, d["标签"])
    ax.set(xlabel="标准化系数及95%置信区间（logit尺度）", ylabel="", title="混合模型效应")
    save(fig, "result_q1_effects")


def result_q2() -> None:
    d = pd.read_csv(Q2 / "问题2核心方案对比.csv")
    fig, ax = publication_subplots(aspect=0.62)
    ax.scatter(d["患者平均孕周"], d["总体预测达标概率"], s=48, c=[PALETTE["neutral"], PALETTE["secondary"], PALETTE["positive"]])
    for _, row in d.iterrows():
        ax.annotate(row["方案"], (row["患者平均孕周"], row["总体预测达标概率"]), xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="患者平均推荐孕周（周）", ylabel="总体预测达标概率", title="准确性—等待权衡")
    save(fig, "result_q2_tradeoff")


def result_q3() -> None:
    audit = pd.read_csv(Q3 / "最终政策外层4折审计.csv")
    fig, axes = publication_subplots(1, 2, aspect=0.49)
    axes[0].hlines([0, 1], [20, 35], [35, 47], color=[PALETTE["primary"], PALETTE["secondary"]], lw=8)
    axes[0].scatter([27.5, 41], [0, 1], s=55, color="white", edgecolor="black", zorder=3)
    axes[0].text(27.5, 0, "18周", ha="center", va="center", fontsize=8)
    axes[0].text(41, 1, "22周", ha="center", va="center", fontsize=8)
    axes[0].axvline(35, color="black", ls="--", lw=0.8)
    axes[0].set_yticks([0, 1], ["BMI<35（n=227）", "BMI≥35（n=40）"])
    axes[0].set(xlim=(20, 47), xlabel="BMI (kg/m²)", ylabel="", title="最终分组与推荐时点")
    x = np.arange(1, 5)
    axes[1].scatter(x - 0.04, audit["政策组最低均值"], marker="o", label="组内最低均值", s=28)
    axes[1].scatter(x + 0.04, audit["政策组最低保守下限"], marker="s", label="组内最低LCB", s=28)
    axes[1].axhline(0.90, color=PALETTE["contrast"], ls="--", lw=0.8, label="可靠性门槛")
    axes[1].set_xticks(x)
    axes[1].set_ylim(0.89, 0.965)
    axes[1].set(xlabel="外层测试折", ylabel="安全性指标", title="固定政策内部回放")
    axes[1].legend()
    add_panel_labels(axes)
    save(fig, "result_q3_final_policy")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    contracts = [
        {"id": "raw_q1", "层级": "原始数据", "结论": "孕周、BMI与Y浓度存在结构性关系", "文件": "raw_q1_y_week_bmi"},
        {"id": "raw_q2", "层级": "原始数据", "结论": "问题2切点位于高BMI尾部但组内仍有52人", "文件": "raw_q2_bmi_distribution"},
        {"id": "raw_q3", "层级": "原始数据", "结论": "患者级身高、体重与BMI存在相关结构", "文件": "raw_q3_feature_evidence"},
        {"id": "process_q1", "层级": "建模过程", "结论": "残差总体可用但尾部偏差需披露", "文件": "process_q1_diagnostics"},
        {"id": "process_q2", "层级": "建模过程", "结论": "18/22在三类稳定性审计中最稳健", "文件": "process_q2_policy_stability"},
        {"id": "process_q3", "层级": "建模过程", "结论": "无外层选参审计为均值4/4、LCB 3/4折安全", "文件": "process_q3_group_selection"},
        {"id": "result_q1", "层级": "最终结果", "结论": "孕周正向、BMI负向，交互不显著", "文件": "result_q1_effects"},
        {"id": "result_q2", "层级": "最终结果", "结论": "18/22以等待时间换取最高可靠性", "文件": "result_q2_tradeoff"},
        {"id": "result_q3", "层级": "最终结果", "结论": "BMI 35分组的18/22固定政策在内部回放中4/4折通过", "文件": "result_q3_final_policy"},
    ]
    (OUT / "q123_figure_contracts.json").write_text(json.dumps(contracts, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_q1(); raw_q2(); raw_q3()
    process_q1(); process_q2(); process_q3()
    result_q1(); result_q2(); result_q3()
    print(f"generated={len(contracts)} output={OUT}")


if __name__ == "__main__":
    main()
