from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[1]
B_DIR = ROOT / "B题" / "B题" / "附件"
C_FILE = ROOT / "C题" / "data" / "raw" / "附件.xlsx"
OUT_B = ROOT / "B题" / "output" / "figures" / "attachments"
OUT_C = ROOT / "C题" / "output" / "figures" / "attachments"
OUT_B.mkdir(parents=True, exist_ok=True)
OUT_C.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfcfe",
        "axes.edgecolor": "#667085",
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "grid.color": "#d0d5dd",
        "grid.alpha": 0.38,
    }
)


def parse_week(value) -> float:
    m = re.fullmatch(r"\s*(\d+)[wW](?:\+(\d+))?\s*", str(value))
    return float(m.group(1)) + float(m.group(2) or 0) / 7.0


def save(fig: plt.Figure, directory: Path, name: str) -> None:
    fig.savefig(directory / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_b_attachment(index: int, material: str, angle: int) -> None:
    d = pd.read_excel(B_DIR / f"附件{index}.xlsx")
    x = d.iloc[:, 0].to_numpy(float)
    y = d.iloc[:, 1].to_numpy(float)

    fig, ax = plt.subplots(figsize=(11.2, 5.8), constrained_layout=True)
    ax.plot(x, y, color="#245b88", lw=1.0, label="实测反射率")
    if material == "碳化硅":
        ax.axvspan(1500, 4000, color="#e9a23b", alpha=0.12, label="稳定干涉分析频段")
    else:
        ax.axvspan(1000, 4000, color="#4ba3a6", alpha=0.10, label="干涉分析频段")
    ax.set_title(f"B题附件{index}：{material}晶圆红外反射光谱（入射角 {angle}°）", pad=14)
    ax.set_xlabel("波数 / cm$^{-1}$")
    ax.set_ylabel("反射率 / %")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.grid(True, lw=0.7)
    ax.legend(frameon=False, loc="best")
    ax.text(
        0.995,
        0.02,
        f"观测点：{len(d):,}  |  波数范围：{x.min():.1f}–{x.max():.1f} cm$^{{-1}}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#475467",
        fontsize=9,
    )
    save(fig, OUT_B, f"B题_附件{index}_{material}_{angle}度.png")


def add_panel_label(ax, text: str) -> None:
    ax.text(-0.08, 1.06, text, transform=ax.transAxes, fontsize=12, weight="bold", color="#344054")


def male_overview() -> None:
    d = pd.read_excel(C_FILE, sheet_name="男胎检测数据")
    d["孕周数"] = d["检测孕周"].map(parse_week)
    women = d.sort_values("孕周数").drop_duplicates("孕妇代码")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), constrained_layout=True)
    fig.suptitle("C题附件：男胎检测数据可视化总览", fontsize=18, weight="bold")

    ax = axes[0, 0]
    sc = ax.scatter(
        d["孕周数"], d["Y染色体浓度"] * 100,
        c=d["孕妇BMI"], cmap="viridis", s=18, alpha=0.52, linewidths=0,
        norm=Normalize(d["孕妇BMI"].quantile(0.01), d["孕妇BMI"].quantile(0.99)),
    )
    ax.axhline(4, color="#d92d20", ls="--", lw=1.5, label="4% 达标线")
    ax.set(title="Y 染色体浓度与检测孕周", xlabel="检测孕周 / 周", ylabel="Y 染色体浓度 / %")
    ax.grid(True)
    ax.legend(frameon=False)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("孕妇 BMI")
    add_panel_label(ax, "A")

    ax = axes[0, 1]
    ax.hist(women["孕妇BMI"], bins=np.arange(20, 48.1, 1.5), color="#3b82a0", alpha=0.88, edgecolor="white")
    ax.axvline(34.5, color="#d92d20", ls="--", lw=1.5, label="模型分界：34.5")
    ax.set(title="孕妇层 BMI 分布", xlabel="BMI", ylabel="孕妇人数")
    ax.grid(True, axis="y")
    ax.legend(frameon=False)
    add_panel_label(ax, "B")

    ax = axes[1, 0]
    bins = pd.cut(d["孕周数"], bins=np.arange(10, 30.01, 2), right=False)
    q = d.groupby(bins, observed=True).agg(
        week=("孕周数", "mean"),
        median=("Y染色体浓度", "median"),
        q25=("Y染色体浓度", lambda x: x.quantile(0.25)),
        q75=("Y染色体浓度", lambda x: x.quantile(0.75)),
        n=("Y染色体浓度", "size"),
    )
    ax.fill_between(q["week"], q["q25"] * 100, q["q75"] * 100, color="#84c5d6", alpha=0.35, label="四分位区间")
    ax.plot(q["week"], q["median"] * 100, "o-", color="#155e75", lw=2, label="分箱中位数")
    ax.axhline(4, color="#d92d20", ls="--", lw=1.2)
    ax.set(title="孕周分箱后的 Y 浓度趋势", xlabel="检测孕周 / 周", ylabel="Y 染色体浓度 / %")
    ax.grid(True)
    ax.legend(frameon=False)
    add_panel_label(ax, "C")

    ax = axes[1, 1]
    abnormal = d["染色体的非整倍体"].notna()
    ax.scatter(
        d.loc[~abnormal, "GC含量"] * 100,
        d.loc[~abnormal, "在参考基因组上比对的比例"] * 100,
        s=16, alpha=0.35, color="#667085", label="AB 空白",
    )
    ax.scatter(
        d.loc[abnormal, "GC含量"] * 100,
        d.loc[abnormal, "在参考基因组上比对的比例"] * 100,
        s=22, alpha=0.72, color="#d97706", marker="^", label="AB 标记异常",
    )
    ax.axvspan(40, 60, color="#12b76a", alpha=0.06, label="GC 正常范围")
    ax.set(title="测序质量：GC 含量与比对率", xlabel="GC 含量 / %", ylabel="参考基因组比对率 / %")
    ax.grid(True)
    ax.legend(frameon=False, fontsize=9)
    add_panel_label(ax, "D")

    fig.text(0.5, -0.01, f"记录数 {len(d):,}；独立孕妇 {d['孕妇代码'].nunique():,}。重复检测按孕妇代码识别。", ha="center", color="#475467")
    save(fig, OUT_C, "C题_男胎检测数据_可视化总览.png")


def female_overview() -> None:
    d = pd.read_excel(C_FILE, sheet_name="女胎检测数据")
    d["孕周数"] = d["检测孕周"].map(parse_week)
    women = d.sort_values("孕周数").drop_duplicates("孕妇代码")
    ab = d["染色体的非整倍体"].fillna("").astype(str)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), constrained_layout=True)
    fig.suptitle("C题附件：女胎检测数据可视化总览", fontsize=18, weight="bold")

    ax = axes[0, 0]
    zcols = ["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值"]
    labels = ["13号", "18号", "21号", "X"]
    values = [d[c].dropna().to_numpy() for c in zcols]
    parts = ax.violinplot(values, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#4c9bb0")
        body.set_edgecolor("#245b88")
        body.set_alpha(0.55)
    parts["cmedians"].set_color("#d92d20")
    ax.axhline(3, color="#d97706", ls="--", lw=1.2)
    ax.axhline(-3, color="#d97706", ls="--", lw=1.2, label="|Z|=3 警戒线")
    ax.set_xticks(range(1, 5), labels)
    ax.set(title="染色体 Z 值分布", xlabel="染色体", ylabel="Z 值")
    ax.grid(True, axis="y")
    ax.legend(frameon=False)
    add_panel_label(ax, "A")

    ax = axes[0, 1]
    ax.hist(women["孕妇BMI"].dropna(), bins=np.arange(24, 47.1, 1.5), color="#7f56a3", alpha=0.86, edgecolor="white")
    ax.set(title="孕妇层 BMI 分布", xlabel="BMI", ylabel="孕妇人数")
    ax.grid(True, axis="y")
    add_panel_label(ax, "B")

    ax = axes[1, 0]
    categories = ["T13", "T18", "T21"]
    counts = [int(ab.str.contains(c).sum()) for c in categories]
    women_counts = [int(d.loc[ab.str.contains(c), "孕妇代码"].nunique()) for c in categories]
    xpos = np.arange(3)
    width = 0.36
    bars1 = ax.bar(xpos - width / 2, counts, width, color="#2e90a5", label="异常记录数")
    bars2 = ax.bar(xpos + width / 2, women_counts, width, color="#f79009", label="涉及孕妇数")
    ax.bar_label(bars1, padding=3, fontsize=9)
    ax.bar_label(bars2, padding=3, fontsize=9)
    ax.set_xticks(xpos, categories)
    ax.set(title="AB 列异常标签构成", xlabel="异常类型", ylabel="数量")
    ax.grid(True, axis="y")
    ax.legend(frameon=False)
    add_panel_label(ax, "C")

    ax = axes[1, 1]
    abnormal = ab.ne("")
    ax.scatter(
        d.loc[~abnormal, "GC含量"] * 100,
        d.loc[~abnormal, "在参考基因组上比对的比例"] * 100,
        s=17, alpha=0.35, color="#667085", label="AB 空白",
    )
    ax.scatter(
        d.loc[abnormal, "GC含量"] * 100,
        d.loc[abnormal, "在参考基因组上比对的比例"] * 100,
        s=25, alpha=0.75, color="#d92d20", marker="^", label="AB 标记异常",
    )
    ax.axvspan(40, 60, color="#12b76a", alpha=0.06, label="GC 正常范围")
    ax.set(title="测序质量与 AB 标签", xlabel="GC 含量 / %", ylabel="参考基因组比对率 / %")
    ax.grid(True)
    ax.legend(frameon=False, fontsize=9)
    add_panel_label(ax, "D")

    fig.text(0.5, -0.01, f"记录数 {len(d):,}；独立孕妇 {d['孕妇代码'].nunique():,}；出生结局列全部为“健康”。", ha="center", color="#475467")
    save(fig, OUT_C, "C题_女胎检测数据_可视化总览.png")


def main() -> None:
    for args in [
        (1, "碳化硅", 10),
        (2, "碳化硅", 15),
        (3, "硅", 10),
        (4, "硅", 15),
    ]:
        plot_b_attachment(*args)
    male_overview()
    female_overview()
    generated = sorted(OUT_B.glob("*.png")) + sorted(OUT_C.glob("*.png"))
    print("\n".join(str(p) for p in generated))


if __name__ == "__main__":
    main()
