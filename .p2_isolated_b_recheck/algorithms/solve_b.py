from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks, savgol_filter


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "B题" / "附件"
OUT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures" / "baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def n_silicon(wavenumber_cm: np.ndarray) -> np.ndarray:
    """Edwards & Ochoa (1980), 26 C; wavelength is in micrometres."""
    lam = 1.0e4 / np.asarray(wavenumber_cm, dtype=float)
    L = 1.0 / (lam**2 - 0.028)
    return (
        3.41983
        + 1.59906e-1 * L
        - 1.23109e-1 * L**2
        + 1.26878e-6 * lam**2
        - 1.95104e-9 * lam**4
    )


def n_sic(wavenumber_cm: np.ndarray) -> np.ndarray:
    """Low-doped 4H-SiC Lorentz oscillator model away from the phonon band."""
    w = np.asarray(wavenumber_cm, dtype=float)
    eps_inf, w_l, w_t, gamma = 6.56, 970.1, 793.9, 5.501
    eps = eps_inf * (1.0 + (w_l**2 - w_t**2) / (w_t**2 - w**2 - 1j * gamma * w))
    return np.sqrt(np.maximum(np.real(eps), 1.0e-12))


def optical_coordinate(x: np.ndarray, angle_deg: float, material: str) -> np.ndarray:
    n = n_sic(x) if material == "SiC" else n_silicon(x)
    return 2.0 * x * np.sqrt(n**2 - np.sin(np.deg2rad(angle_deg)) ** 2)


def preprocess(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = len(y)
    window = min(1001, points - (1 - points % 2))
    window = max(window, 51)
    if window % 2 == 0:
        window -= 1
    baseline = savgol_filter(y, window, 3)
    residual = y - baseline
    scale = savgol_filter(np.abs(residual), min(window, 501) | 1, 2)
    scale = np.maximum(scale, np.quantile(np.abs(residual), 0.2) + 1e-9)
    return residual / scale, baseline


def fft_frequency(x: np.ndarray, signal: np.ndarray) -> float:
    dx = float(np.median(np.diff(x)))
    z = (signal - np.mean(signal)) * np.hanning(len(signal))
    f = np.fft.rfftfreq(len(z), dx)
    p = np.abs(np.fft.rfft(z)) ** 2
    valid = (f > 0.002) & (f < 0.25)
    return float(f[valid][np.argmax(p[valid])])


def quadratic_extrema_positions(x: np.ndarray, signal: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """在原始波数网格的三点邻域内做抛物线顶点插值。"""
    positions: list[float] = []
    for index in indices:
        if index <= 0 or index >= len(signal) - 1:
            positions.append(float(x[index]))
            continue
        left, center, right = signal[index - 1:index + 2]
        denominator = left - 2.0 * center + right
        if not np.isfinite(denominator) or abs(denominator) < 1.0e-12:
            offset = 0.0
        else:
            offset = float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))
        local_step = 0.5 * float(x[index + 1] - x[index - 1])
        positions.append(float(x[index] + offset * local_step))
    return np.asarray(positions)


def peak_thickness(
    x: np.ndarray, y: np.ndarray, angle: float, material: str
) -> dict[str, float | int]:
    z, _ = preprocess(x, y)
    f0 = fft_frequency(x, z)
    dx = float(np.median(np.diff(x)))
    period_points = max(8, int(round(1.0 / (f0 * dx))))
    q = optical_coordinate(x, angle, material)
    d_fft_um = f0 / float(np.median(np.gradient(q, x))) * 1.0e4
    estimates: list[float] = []
    counts: list[int] = []
    for sign in (1.0, -1.0):
        peaks, _ = find_peaks(
            sign * z,
            distance=max(5, int(0.55 * period_points)),
            prominence=max(0.18, 0.12 * float(np.std(z))),
        )
        if len(peaks) >= 4:
            extrema_x = quadratic_extrema_positions(x, sign * z, peaks)
            extrema_q = optical_coordinate(extrema_x, angle, material)
            # Consecutive extrema of the same type differ by one order. A median
            # local-spacing estimate is robust to a rare missed or spurious peak.
            local = 1.0e4 / np.diff(extrema_q)
            local = local[(local > 0.60 * d_fft_um) & (local < 1.40 * d_fft_um)]
            estimates.append(float(np.median(local)))
            counts.append(int(len(peaks)))
    return {
        "fft_frequency_cycles_per_cm-1": f0,
        "thickness_um": float(np.mean(estimates)),
        "peak_trough_disagreement_um": float(np.ptp(estimates)) if len(estimates) == 2 else np.nan,
        "extrema_count": int(sum(counts)),
    }


def harmonic_design(phi: np.ndarray, harmonics: int) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, len(phi))
    cols = [np.ones_like(t), t, t**2, t**3]
    for k in range(1, harmonics + 1):
        c, s = np.cos(k * phi), np.sin(k * phi)
        # A slowly varying complex amplitude absorbs the spectral envelope without
        # changing the physically constrained phase/frequency.
        cols.extend([c, s, t * c, t * s, t**2 * c, t**2 * s])
    return np.column_stack(cols)


def spectrum_rss(x: np.ndarray, y: np.ndarray, q: np.ndarray, d_um: float, harmonics: int) -> tuple[float, np.ndarray]:
    phi = 2.0 * np.pi * (d_um / 1.0e4) * q
    X = harmonic_design(phi, harmonics)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ coef
    return float(residual @ residual), coef


def joint_fourier_fit(records: list[dict], initial_um: float, harmonics: int) -> dict:
    def objective(d_um: float) -> float:
        total = 0.0
        for r in records:
            rss, _ = spectrum_rss(r["x"], r["y"], r["q"], d_um, harmonics)
            total += rss
        return total

    result = minimize_scalar(
        objective,
        bounds=(0.75 * initial_um, 1.25 * initial_um),
        method="bounded",
        options={"xatol": 1e-8},
    )
    d = float(result.x)
    rss = 0.0
    n_total = 0
    k_total = 1
    harmonic_ratios = []
    fits = []
    for r in records:
        local_rss, coef = spectrum_rss(r["x"], r["y"], r["q"], d, harmonics)
        rss += local_rss
        n_total += len(r["x"])
        k_total += len(coef)
        offset = 4
        fundamental = np.hypot(coef[offset], coef[offset + 1])
        if harmonics >= 2:
            second = offset + 6
            harmonic_ratios.append(float(np.hypot(coef[second], coef[second + 1]) / max(fundamental, 1e-12)))
        phi = 2.0 * np.pi * (d / 1.0e4) * r["q"]
        fits.append(harmonic_design(phi, harmonics) @ coef)
    bic = n_total * np.log(rss / n_total) + k_total * np.log(n_total)
    return {
        "thickness_um": d,
        "rss": rss,
        "bic": float(bic),
        "harmonic2_to_fundamental": harmonic_ratios,
        "fits": fits,
    }


def load_record(index: int, angle: float, material: str, band: tuple[float, float]) -> dict:
    d = pd.read_excel(DATA_DIR / f"附件{index}.xlsx")
    if d.shape != (7469, 2):
        raise ValueError(f"附件{index}形状异常: {d.shape}")
    x_all = d.iloc[:, 0].to_numpy(float)
    y_all = d.iloc[:, 1].to_numpy(float) / 100.0
    if not np.all(np.diff(x_all) > 0):
        raise ValueError(f"附件{index}波数必须严格递增")
    mask = (x_all >= band[0]) & (x_all <= band[1]) & np.isfinite(y_all) & (y_all > 0)
    x, y = x_all[mask], y_all[mask]
    return {
        "index": index,
        "angle": angle,
        "material": material,
        "x": x,
        "y": y,
        "q": optical_coordinate(x, angle, material),
        "x_all": x_all,
        "y_all": y_all,
    }


def analyse_material(material: str, indices: tuple[int, int], band: tuple[float, float]) -> dict:
    records = [load_record(indices[0], 10.0, material, band), load_record(indices[1], 15.0, material, band)]
    peak_results = [peak_thickness(r["x"], r["y"], r["angle"], material) for r in records]
    initial = float(np.mean([v["thickness_um"] for v in peak_results]))
    one = joint_fourier_fit(records, initial, 1)
    three = joint_fourier_fit(records, one["thickness_um"], 3)

    bands = []
    width = band[1] - band[0]
    for shift in (-0.12, 0.0, 0.12):
        lo = band[0] + shift * width
        hi = band[1] + shift * width
        rr = [load_record(indices[0], 10.0, material, (lo, hi)), load_record(indices[1], 15.0, material, (lo, hi))]
        local_peaks = [peak_thickness(r["x"], r["y"], r["angle"], material) for r in rr]
        local_initial = float(np.mean([v["thickness_um"] for v in local_peaks]))
        bands.append(local_initial)

    result = {
        "material": material,
        "band_cm-1": band,
        "per_angle_peak_regression": peak_results,
        "single_beam_joint": {k: v for k, v in one.items() if k != "fits"},
        "three_harmonic_joint": {k: v for k, v in three.items() if k != "fits"},
        "delta_BIC_three_minus_one": float(three["bic"] - one["bic"]),
        "band_sensitivity_um": bands,
        # Extrema positions determine the fundamental optical period and remain
        # identifiable when envelope flexibility makes full-waveform fits weakly
        # identified. The harmonic fits are retained as a multi-beam diagnostic.
        "reported_thickness_um": float(initial),
        "sensitivity_half_range_um": float(np.ptp(bands) / 2.0),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for row, (r, fit1, fit3) in enumerate(zip(records, one["fits"], three["fits"])):
        ax = axes[row, 0]
        ax.plot(r["x_all"], 100.0 * r["y_all"], lw=0.7, color="#335c81")
        ax.axvspan(*band, color="#e09f3e", alpha=0.18)
        ax.set(title=f"{material}，入射角 {r['angle']:.0f}°：原始反射谱", xlabel="波数 / cm$^{-1}$", ylabel="反射率 / %")
        ax = axes[row, 1]
        stride = max(1, len(r["x"]) // 1400)
        ax.plot(r["x"][::stride], 100.0 * r["y"][::stride], lw=0.8, color="#444444", label="实测")
        ax.plot(r["x"][::stride], 100.0 * fit1[::stride], lw=1.1, color="#2a9d8f", label="单谐波")
        ax.plot(r["x"][::stride], 100.0 * fit3[::stride], lw=1.0, color="#d1495b", label="三谐波")
        ax.set(title=f"拟合频段与模型对比（{r['angle']:.0f}°）", xlabel="波数 / cm$^{-1}$", ylabel="反射率 / %")
        ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.savefig(FIGURE_DIR / f"{material}_spectral_fit.png", dpi=200)
    plt.close(fig)
    return result


def main() -> None:
    sic = analyse_material("SiC", (1, 2), (1500.0, 4000.0))
    silicon = analyse_material("Si", (3, 4), (1000.0, 4000.0))
    results = {"SiC": sic, "Si": silicon}
    (OUT_DIR / "b_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for name, r in results.items():
        rows.append({
            "材料": name,
            "拟合波段下限(cm-1)": r["band_cm-1"][0],
            "拟合波段上限(cm-1)": r["band_cm-1"][1],
            "报告厚度(um)": r["reported_thickness_um"],
            "频段敏感性半极差(um)": r["sensitivity_half_range_um"],
            "三谐波-单谐波BIC": r["delta_BIC_three_minus_one"],
            "二次谐波/基频_10deg": r["three_harmonic_joint"]["harmonic2_to_fundamental"][0],
            "二次谐波/基频_15deg": r["three_harmonic_joint"]["harmonic2_to_fundamental"][1],
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "b_summary.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
