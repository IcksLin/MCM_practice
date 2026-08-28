from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

from solve_b import (
    ROOT,
    harmonic_design,
    load_record,
    optical_coordinate,
    peak_thickness,
    preprocess,
)


SEED = 20250825
BOOTSTRAP_REPEATS = 400
BLOCK_LENGTH = 2
FIG_ROOT = ROOT / "figures"
RESULT_ROOT = ROOT / "results"
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def moving_block_sample(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    starts = rng.integers(0, n - BLOCK_LENGTH + 1, size=int(np.ceil(n / BLOCK_LENGTH)))
    return np.concatenate([values[s : s + BLOCK_LENGTH] for s in starts])[:n]


def local_thicknesses(record: dict) -> np.ndarray:
    """Extract physically meaningful adjacent-order thickness estimates."""
    x, y = record["x"], record["y"]
    z, _ = preprocess(x, y)
    dx = float(np.median(np.diff(x)))
    spectrum = np.abs(np.fft.rfft((z - z.mean()) * np.hanning(len(z)))) ** 2
    frequencies = np.fft.rfftfreq(len(z), dx)
    valid = (frequencies > 0.002) & (frequencies < 0.25)
    f0 = float(frequencies[valid][np.argmax(spectrum[valid])])
    period_points = max(8, int(round(1.0 / (f0 * dx))))
    base = float(peak_thickness(x, y, record["angle"], record["material"])["thickness_um"])
    values: list[float] = []
    for sign in (1.0, -1.0):
        peaks, _ = find_peaks(sign * z, distance=max(5, int(0.55 * period_points)), prominence=max(0.18, 0.12 * float(np.std(z))))
        q = optical_coordinate(x[peaks], record["angle"], record["material"])
        local = 1e4 / np.diff(q)
        values.extend(local[(local > 0.60 * base) & (local < 1.40 * base)])
    return np.asarray(values)


def bootstrap_record(record: dict, repeats: int = BOOTSTRAP_REPEATS) -> np.ndarray:
    rng = np.random.default_rng(SEED + int(record["index"]))
    local = local_thicknesses(record)
    estimates: list[float] = []
    for _ in range(repeats):
        estimates.append(float(np.median(moving_block_sample(local, rng))))
    return np.asarray(estimates)


def fixed_design_error(record: dict, harmonics: int, initial: float, folds: int = 5) -> float:
    n = len(record["x"])
    fold_ids = np.array_split(np.arange(n), folds)
    squared, count = 0.0, 0
    for valid in fold_ids:
        train = np.ones(n, dtype=bool)
        train[valid] = False

        def objective(d_um: float) -> float:
            phi = 2 * np.pi * (d_um / 1e4) * record["q"]
            design = harmonic_design(phi, harmonics)
            coef, *_ = np.linalg.lstsq(design[train], record["y"][train], rcond=None)
            residual = record["y"][train] - design[train] @ coef
            return float(residual @ residual)

        fit = minimize_scalar(objective, bounds=(0.8 * initial, 1.2 * initial), method="bounded")
        phi = 2 * np.pi * (float(fit.x) / 1e4) * record["q"]
        design = harmonic_design(phi, harmonics)
        coef, *_ = np.linalg.lstsq(design[train], record["y"][train], rcond=None)
        error = record["y"][valid] - design[valid] @ coef
        squared += float(error @ error)
        count += len(valid)
    return float(np.sqrt(squared / count) / np.std(record["y"]))


def sensitivity(record: dict, base: float) -> list[dict]:
    rows = []
    x, y = record["x"], record["y"]
    material = record["material"]
    for delta_angle in (-0.2, 0.0, 0.2):
        estimate = peak_thickness(x, y, record["angle"] + delta_angle, material)["thickness_um"]
        rows.append({"factor": "angle_deg", "level": delta_angle, "thickness_um": estimate})
    for scale in (0.995, 1.0, 1.005):
        # q is nearly proportional to n in the selected transparent bands.
        rows.append({"factor": "refractive_index_scale", "level": scale, "thickness_um": base / scale})
    lo, hi = float(x.min()), float(x.max())
    width = hi - lo
    for shift in (-0.08, 0.0, 0.08):
        mask = (x >= lo + shift * width) & (x <= hi + shift * width)
        estimate = peak_thickness(x[mask], y[mask], record["angle"], material)["thickness_um"]
        rows.append({"factor": "band_shift_fraction", "level": shift, "thickness_um": estimate})
    clipped = np.clip(y, 0.0, 1.0)
    rows.append({"factor": "reflectance_clip", "level": 1.0, "thickness_um": peak_thickness(x, clipped, record["angle"], material)["thickness_um"]})
    return rows


def save_figure(fig: plt.Figure, category: str, name: str) -> None:
    folder = FIG_ROOT / category
    folder.mkdir(parents=True, exist_ok=True)
    png = folder / f"{name}.png"
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    with Image.open(png) as source:
        source.convert("L").save(folder / f"{name}_grayscale.png", dpi=(320, 320))


def make_figures(records: list[dict], boot: dict[int, np.ndarray], summary: pd.DataFrame, sensitivity_df: pd.DataFrame) -> None:
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False, "figure.figsize": (6.3, 4.0)})
    # Raw evidence: three distinct questions/views.
    fig, ax = plt.subplots()
    for r, color in zip(records[:2], COLORS):
        ax.plot(r["x_all"], 100 * r["y_all"], color=color, lw=0.8, label=f"{r['angle']:.0f}°")
    ax.axvspan(1500, 4000, color="0.7", alpha=0.18, label="有效频段")
    ax.set(xlabel="波数 (cm$^{-1}$)", ylabel="反射率 (%)", title="SiC 原始反射光谱与有效频段")
    ax.legend(frameon=False)
    save_figure(fig, "raw", "raw_q1_sic_spectra")

    fig, ax = plt.subplots()
    for r, color in zip(records, COLORS):
        z, _ = preprocess(r["x"], r["y"])
        ax.plot(r["x"], z, color=color, lw=0.65, alpha=0.8, label=f"{r['material']} {r['angle']:.0f}°")
    ax.set(xlabel="波数 (cm$^{-1}$)", ylabel="标准化去趋势残差", title="去趋势后的干涉条纹")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, "raw", "raw_q2_detrended_fringes")

    fig, ax = plt.subplots()
    labels = [f"附件{r['index']}" for r in records]
    outlier_rates = [100 * np.mean((r["y_all"] < 0) | (r["y_all"] > 1)) for r in records]
    ax.scatter(labels, outlier_rates, s=48, c=COLORS)
    ax.set(ylabel="超出 [0,100%] 的观测比例 (%)", title="仪器反射率越界诊断")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "raw", "raw_q3_measurement_diagnostics")

    # Process evidence.
    fig, ax = plt.subplots()
    for r, color in zip(records, COLORS):
        q = optical_coordinate(r["x"], r["angle"], r["material"])
        ax.plot(r["x"], q / 1e4, color=color, lw=1.0, label=f"{r['material']} {r['angle']:.0f}°")
    ax.set(xlabel="波数 (cm$^{-1}$)", ylabel=r"$q/10^4$ ($\mu$m$^{-1}$)", title="色散与角度修正后的光学坐标")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, "process", "process_q1_optical_coordinate")

    fig, ax = plt.subplots()
    positions = np.arange(len(records))
    for pos, r, color in zip(positions, records, COLORS):
        vals = 100 * (boot[r["index"]] / summary.iloc[pos]["thickness_um"] - 1)
        parts = ax.violinplot(vals, positions=[pos], widths=0.7, showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor(color); body.set_alpha(0.45)
    ax.set_xticks(positions, [f"{r['material']}\n{r['angle']:.0f}°" for r in records])
    ax.set(ylabel="相对基准厚度变化 (%)", title=f"局部条纹区块 Bootstrap（每谱 {BOOTSTRAP_REPEATS} 次）")
    save_figure(fig, "process", "process_q2_block_bootstrap")

    fig, ax = plt.subplots()
    cv = summary.set_index("record")
    x = np.arange(len(cv))
    ax.scatter(x - 0.08, cv["cv_nrmse_h1"], label="单谐波", marker="o", color=COLORS[0])
    ax.scatter(x + 0.08, cv["cv_nrmse_h3"], label="三谐波", marker="s", color=COLORS[1])
    ax.set_xticks(x, cv.index)
    ax.set(ylabel="5 折分块验证 NRMSE", title="单光束与多谐波模型的外推误差")
    ax.legend(frameon=False)
    save_figure(fig, "process", "process_q3_blocked_cv")

    # Final evidence.
    fig, ax = plt.subplots()
    for pos, (_, row) in enumerate(summary.iterrows()):
        ax.errorbar(pos, row["thickness_um"], yerr=[[row["thickness_um"] - row["bootstrap_lo"]], [row["bootstrap_hi"] - row["thickness_um"]]], fmt="o", color=COLORS[pos], capsize=4)
    ax.set_xticks(np.arange(len(summary)), summary["record"])
    ax.set(ylabel=r"厚度 ($\mu$m)", title="各附件厚度与 95% 区块 Bootstrap 区间")
    save_figure(fig, "result", "result_q1_thickness_intervals")

    fig, ax = plt.subplots()
    factor_names = {"angle_deg": "入射角", "refractive_index_scale": "折射率", "band_shift_fraction": "拟合频段", "reflectance_clip": "反射率截断"}
    sensitivity_df = sensitivity_df.assign(factor_zh=sensitivity_df["factor"].map(factor_names))
    for (record_name, group), color in zip(sensitivity_df.groupby("record"), COLORS):
        delta = 100 * (group["thickness_um"] / group["base_um"] - 1)
        ax.scatter(group["factor_zh"], delta, label=record_name, color=color, alpha=0.8)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.tick_params(axis="x", rotation=25)
    ax.set(ylabel="相对基准变化 (%)", title="角度、折射率、频段与截断敏感性")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, "result", "result_q2_sensitivity_envelope")

    fig, ax = plt.subplots()
    mat = summary.groupby("material").agg(thickness=("thickness_um", "mean"), angle_diff=("thickness_um", lambda s: float(np.ptp(s))), cv_gain=("cv_gain_pct", "mean"))
    for pos, (name, row) in enumerate(mat.iterrows()):
        ax.scatter(pos, row["thickness"], s=70, color=COLORS[pos], label=f"{name}: 角度差 {row['angle_diff']:.4f} μm，CV改善 {row['cv_gain']:.1f}%")
    ax.set_xticks(range(len(mat)), mat.index)
    ax.set(ylabel=r"推荐厚度 ($\mu$m)", title="最终厚度与跨角度/模型一致性")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "result", "result_q3_final_recommendation")


def main() -> None:
    specs = [(1, 10.0, "SiC", (1500.0, 4000.0)), (2, 15.0, "SiC", (1500.0, 4000.0)), (3, 10.0, "Si", (1000.0, 4000.0)), (4, 15.0, "Si", (1000.0, 4000.0))]
    records = [load_record(*spec) for spec in specs]
    boot, summary_rows, sensitivity_rows = {}, [], []
    for r in records:
        base = float(peak_thickness(r["x"], r["y"], r["angle"], r["material"])["thickness_um"])
        values = bootstrap_record(r)
        boot[r["index"]] = values
        lo, hi = np.quantile(values, [0.025, 0.975])
        cv1 = fixed_design_error(r, 1, base)
        cv3 = fixed_design_error(r, 3, base)
        label = f"附件{r['index']}({r['angle']:.0f}°)"
        summary_rows.append({"record": label, "material": r["material"], "angle_deg": r["angle"], "thickness_um": base, "bootstrap_lo": lo, "bootstrap_hi": hi, "bootstrap_n": len(values), "cv_nrmse_h1": cv1, "cv_nrmse_h3": cv3, "cv_gain_pct": 100 * (cv1 - cv3) / cv1})
        for row in sensitivity(r, base):
            sensitivity_rows.append({"record": label, "base_um": base, **row})
    summary = pd.DataFrame(summary_rows)
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT_ROOT / "robustness_summary.csv", index=False, encoding="utf-8-sig")
    sensitivity_df.to_csv(RESULT_ROOT / "sensitivity_details.csv", index=False, encoding="utf-8-sig")
    final_rows = []
    for material, group in summary.groupby("material", sort=False):
        indices = [r["index"] for r in records if r["material"] == material]
        joint = 0.5 * (boot[indices[0]] + boot[indices[1]])
        estimate = float(group["thickness_um"].mean())
        local_lo, local_hi = np.quantile(joint, [0.025, 0.975])
        sens = sensitivity_df[sensitivity_df["record"].isin(group["record"])]
        systematic = float(np.max(np.abs(sens["thickness_um"] - sens["base_um"])))
        final_rows.append({"material": material, "recommended_thickness_um": estimate, "bootstrap95_lo_um": local_lo, "bootstrap95_hi_um": local_hi, "sensitivity_max_abs_um": systematic, "angle_difference_um": float(np.ptp(group["thickness_um"])), "three_harmonic_cv_gain_pct": float(group["cv_gain_pct"].mean()), "multibeam_decision": "不采用多谐波厚度修正；仅保留谐波迹象"})
    pd.DataFrame(final_rows).to_csv(RESULT_ROOT / "final_recommendations.csv", index=False, encoding="utf-8-sig")
    make_figures(records, boot, summary, sensitivity_df)
    manifest = {
        "seed": SEED,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "block_length_points": BLOCK_LENGTH,
        "reproduce_command": f'"{sys.executable}" scripts/run_b.py',
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": plt.matplotlib.__version__},
        "inputs": {f"附件{i}.xlsx": sha256(ROOT / "B题" / "附件" / f"附件{i}.xlsx") for i in range(1, 5)},
        "outputs": ["results/b_results.json", "results/b_summary.csv", "results/robustness_summary.csv", "results/sensitivity_details.csv", "results/final_recommendations.csv"],
    }
    (RESULT_ROOT / "复现清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
