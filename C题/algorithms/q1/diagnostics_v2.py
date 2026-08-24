"""问题1补充实验：混合效应模型诊断与患者级Bootstrap。

实验指标：标准化残差异常比例、残差与拟合值/孕周相关、随机截距分布，以及孕周、
BMI和交互项系数的患者级Bootstrap 95%分位区间。Bootstrap每次抽取孕妇并保留其
全部记录；重复抽中的孕妇赋予新的簇ID，避免伪独立。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from solve_q1 import load_and_prepare


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "results" / "q1" / "robustness"
FORMULA = "Y_logit ~ week_c + week_c2 + bmi_c + week_c:bmi_c"
KEY = ("week_c", "week_c2", "bmi_c", "week_c:bmi_c")
SEED = 20260825


def fit(frame: pd.DataFrame, group: str):
    return smf.mixedlm(FORMULA, frame, groups=frame[group]).fit(reml=False, method="lbfgs", maxiter=1000, disp=False)


def bootstrap(data: pd.DataFrame, repeats: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    patients = data["孕妇代码"].drop_duplicates().to_numpy()
    rows = []
    for repeat in range(repeats):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        pieces = []
        for draw, patient in enumerate(sampled):
            block = data[data["孕妇代码"] == patient].copy()
            block["bootstrap_cluster"] = f"{draw}:{patient}"
            pieces.append(block)
        frame = pd.concat(pieces, ignore_index=True)
        try:
            model = fit(frame, "bootstrap_cluster")
            row = {"repeat": repeat + 1, "converged": bool(model.converged)}
            row.update({name: float(model.params[name]) for name in KEY})
            row["随机截距方差"] = float(model.cov_re.iloc[0, 0])
            row["随机截距退化"] = bool(model.cov_re.iloc[0, 0] < 1e-8)
        except Exception as exc:  # 失败必须保留，不静默丢弃。
            row = {"repeat": repeat + 1, "converged": False, "error": type(exc).__name__}
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    repeats = 10 if args.smoke else args.repeats
    data, _ = load_and_prepare()
    model = fit(data, "孕妇代码")
    fitted = np.asarray(model.fittedvalues, float)
    residual = np.asarray(model.resid, float)
    standardized = (residual - residual.mean()) / residual.std(ddof=1)
    diagnostics = pd.DataFrame({
        "序号": data["序号"].to_numpy(),
        "孕妇代码": data["孕妇代码"].to_numpy(),
        "孕周数": data["孕周数"].to_numpy(float),
        "拟合logit_Y": fitted,
        "残差": residual,
        "标准化残差": standardized,
        "绝对标准化残差大于3": np.abs(standardized) > 3,
    })
    random_intercepts = pd.DataFrame([
        {"孕妇代码": patient, "随机截距": float(np.asarray(value).ravel()[0])}
        for patient, value in model.random_effects.items()
    ])
    boot = bootstrap(data, repeats)
    valid = boot[boot["converged"]].dropna(subset=list(KEY))
    intervals = []
    for name in KEY:
        q = valid[name].quantile([0.025, 0.5, 0.975])
        intervals.append({"变量": name, "原模型系数": float(model.params[name]), "Bootstrap下限": float(q.loc[0.025]), "Bootstrap中位数": float(q.loc[0.5]), "Bootstrap上限": float(q.loc[0.975])})
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else f"b{repeats}"
    diagnostics.to_csv(OUT / f"残差诊断_{suffix}.csv", index=False, encoding="utf-8-sig")
    random_intercepts.to_csv(OUT / f"随机截距分布_{suffix}.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(OUT / f"患者Bootstrap原始结果_{suffix}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(intervals).to_csv(OUT / f"关键系数Bootstrap区间_{suffix}.csv", index=False, encoding="utf-8-sig")
    summary = {
        "mode": suffix,
        "records": len(data),
        "patients": int(data["孕妇代码"].nunique()),
        "model_converged": bool(model.converged),
        "model_formula": FORMULA,
        "random_intercept_variance": float(model.cov_re.iloc[0, 0]),
        "random_intercept_degenerate": bool(model.cov_re.iloc[0, 0] < 1e-8),
        "random_intercept_mean": float(random_intercepts["随机截距"].mean()),
        "random_intercept_sd": float(random_intercepts["随机截距"].std(ddof=1)),
        "standardized_residual_abs_gt_3_rate": float((np.abs(standardized) > 3).mean()),
        "residual_fitted_correlation": float(np.corrcoef(residual, fitted)[0, 1]),
        "residual_week_correlation": float(np.corrcoef(residual, data["孕周数"])[0, 1]),
        "bootstrap_repeats": repeats,
        "bootstrap_valid": len(valid),
        "bootstrap_valid_rate": float(len(valid) / repeats),
        "key_intervals": intervals,
    }
    (OUT / f"实验结论_{suffix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
