"""问题1：Y染色体浓度与孕周、BMI等因素的关系及显著性检验。

权威模型：以孕妇为随机截距的线性混合模型；响应变量为Y浓度的logit。
基础模型检验孕周、孕周二次项、BMI及交互；扩展模型加入年龄、身高、
IVF、唯一比对读段数、比对比例和GC含量，并用似然比检验整体增量。
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.formula.api as smf
from scipy.special import logit
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "raw" / "附件.xlsx"
OUTPUT = ROOT / "output" / "results" / "q1"
EXPECTED_ROWS = 1082
EXPECTED_WOMEN = 267
EXPECTED_INPUT_SHA256 = "14827156218bd4f7e4f16db4aa6d9f757c6648379e038ae6c6b58383648614af"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_week(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.fullmatch(r"\s*(\d+)[wW](?:\+(\d+))?\s*", str(value))
    if not match:
        raise ValueError(f"无法解析孕周：{value!r}")
    return float(match.group(1)) + float(match.group(2) or 0) / 7.0


def standardize(series: pd.Series) -> tuple[pd.Series, float, float]:
    mean = float(series.mean())
    sd = float(series.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"变量 {series.name} 无法标准化")
    return (series - mean) / sd, mean, sd


def load_and_prepare() -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    actual_hash = sha256(INPUT)
    if actual_hash != EXPECTED_INPUT_SHA256:
        raise ValueError(f"附件SHA-256异常：{actual_hash}")
    data = pd.read_excel(INPUT, sheet_name="男胎检测数据", header=0)
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"男胎记录数应为{EXPECTED_ROWS}，实际为{len(data)}")
    if data["孕妇代码"].nunique() != EXPECTED_WOMEN:
        raise ValueError("男胎孕妇数不等于267")
    if data.iloc[0]["序号"] != 1 or data.iloc[-1]["序号"] != 1082:
        raise ValueError("附件首末序号异常")

    data = data.copy()
    data["孕周数"] = data["检测孕周"].map(parse_week)
    if data["Y染色体浓度"].isna().any():
        raise ValueError("男胎Y染色体浓度存在缺失")
    data["Y_logit"] = logit(np.clip(data["Y染色体浓度"].astype(float), 1e-6, 1 - 1e-6))

    scales: dict[str, dict[str, float]] = {}
    sources = {
        "week_c": data["孕周数"].astype(float),
        "bmi_c": data["孕妇BMI"].astype(float),
        "age_c": data["年龄"].astype(float),
        "height_c": data["身高"].astype(float),
        "log_unique_c": np.log(data["唯一比对的读段数  "].astype(float)),
        "map_c": data["在参考基因组上比对的比例"].astype(float),
        "gc_c": data["GC含量"].astype(float),
    }
    for name, source in sources.items():
        data[name], mean, sd = standardize(source.rename(name))
        scales[name] = {"mean": mean, "sd": sd}
    data["week_c2"] = data["week_c"] ** 2
    data["ivf"] = (data["IVF妊娠"] != "自然受孕").astype(int)
    return data, scales


def fit_models(data: pd.DataFrame):
    base_formula = "Y_logit ~ week_c + week_c2 + bmi_c + week_c:bmi_c"
    full_formula = base_formula + " + age_c + height_c + ivf + log_unique_c + map_c + gc_c"
    base = smf.mixedlm(base_formula, data, groups=data["孕妇代码"]).fit(
        reml=False, method="lbfgs", maxiter=1000
    )
    full = smf.mixedlm(full_formula, data, groups=data["孕妇代码"]).fit(
        reml=False, method="lbfgs", maxiter=1000
    )
    if not base.converged or not full.converged:
        raise RuntimeError("混合模型未收敛")
    return base, full


def coefficient_table(model, model_name: str) -> pd.DataFrame:
    conf = model.conf_int()
    rows = []
    for term in model.fe_params.index:
        rows.append(
            {
                "模型": model_name,
                "变量": term,
                "系数": float(model.fe_params[term]),
                "标准误": float(model.bse_fe[term]),
                "p值": float(model.pvalues[term]),
                "95%CI下限": float(conf.loc[term, 0]),
                "95%CI上限": float(conf.loc[term, 1]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data, scales = load_and_prepare()
    base, full = fit_models(data)

    lr = max(0.0, 2.0 * (full.llf - base.llf))
    df = int(len(full.fe_params) - len(base.fe_params))
    comparison = {
        "基础模型AIC": float(base.aic),
        "扩展模型AIC": float(full.aic),
        "似然比统计量": lr,
        "自由度": df,
        "似然比检验p值": float(chi2.sf(lr, df)),
        "基础模型残差标准差": float(np.sqrt(base.scale)),
        "孕妇随机截距标准差": float(np.sqrt(base.cov_re.iloc[0, 0])),
    }

    coefficients = pd.concat(
        [coefficient_table(base, "基础模型"), coefficient_table(full, "扩展模型")],
        ignore_index=True,
    )
    coefficients.to_csv(OUTPUT / "q1_model_coefficients.csv", index=False, encoding="utf-8-sig")

    records = data[["序号", "孕妇代码", "孕周数", "孕妇BMI", "Y染色体浓度"]].copy()
    records["基础模型拟合_logit"] = np.asarray(base.fittedvalues)
    records["基础模型残差_logit"] = np.asarray(base.resid)
    records.to_csv(OUTPUT / "q1_record_fit.csv", index=False, encoding="utf-8-sig")

    descriptive = {
        "记录数": int(len(data)),
        "孕妇数": int(data["孕妇代码"].nunique()),
        "孕周范围": [float(data["孕周数"].min()), float(data["孕周数"].max())],
        "BMI范围": [float(data["孕妇BMI"].min()), float(data["孕妇BMI"].max())],
        "Y浓度均值": float(data["Y染色体浓度"].mean()),
        "Y浓度中位数": float(data["Y染色体浓度"].median()),
    }
    summary = {
        "task": "C题问题1",
        "input": {"path": str(INPUT.relative_to(ROOT)), "sha256": sha256(INPUT)},
        "sample": descriptive,
        "standardization": scales,
        "formulas": {
            "base": "logit(Y)=beta0+beta1*week_c+beta2*week_c^2+beta3*bmi_c+beta4*week_c*bmi_c+u_patient+epsilon",
            "full": "base+age_c+height_c+ivf+log_unique_c+map_c+gc_c",
        },
        "model_comparison": comparison,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
    }
    (OUTPUT / "q1_model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), **descriptive, **comparison}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
