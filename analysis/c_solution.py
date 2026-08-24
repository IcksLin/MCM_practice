from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import chi2, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "C题" / "data" / "raw" / "附件.xlsx"
OUT_DIR = ROOT / "C题" / "output" / "archive" / "legacy_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_week(value) -> float:
    if pd.isna(value):
        return np.nan
    m = re.fullmatch(r"\s*(\d+)[wW](?:\+(\d+))?\s*", str(value))
    if not m:
        raise ValueError(f"无法解析孕周：{value!r}")
    return float(m.group(1)) + float(m.group(2) or 0) / 7.0


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    male = pd.read_excel(SOURCE, sheet_name="男胎检测数据")
    female = pd.read_excel(SOURCE, sheet_name="女胎检测数据")
    for frame in (male, female):
        frame["孕周数"] = frame["检测孕周"].map(parse_week)
    return male, female


def prepare_male(male: pd.DataFrame) -> pd.DataFrame:
    d = male.copy()
    eps = 1e-6
    d["Y_logit"] = logit(np.clip(d["Y染色体浓度"], eps, 1 - eps))
    d["week_c"] = (d["孕周数"] - d["孕周数"].mean()) / d["孕周数"].std()
    d["week_c2"] = d["week_c"] ** 2
    d["bmi_c"] = (d["孕妇BMI"] - d["孕妇BMI"].mean()) / d["孕妇BMI"].std()
    d["age_c"] = (d["年龄"] - d["年龄"].mean()) / d["年龄"].std()
    d["height_c"] = (d["身高"] - d["身高"].mean()) / d["身高"].std()
    d["log_unique_c"] = np.log(d["唯一比对的读段数  "]).pipe(lambda s: (s - s.mean()) / s.std())
    d["map_c"] = (d["在参考基因组上比对的比例"] - d["在参考基因组上比对的比例"].mean()) / d["在参考基因组上比对的比例"].std()
    d["gc_c"] = (d["GC含量"] - d["GC含量"].mean()) / d["GC含量"].std()
    d["ivf"] = (d["IVF妊娠"] != "自然受孕").astype(int)
    return d


def fit_mixed_models(d: pd.DataFrame):
    base_formula = "Y_logit ~ week_c + week_c2 + bmi_c + week_c:bmi_c"
    full_formula = base_formula + " + age_c + height_c + ivf + log_unique_c + map_c + gc_c"
    base = smf.mixedlm(base_formula, d, groups=d["孕妇代码"]).fit(reml=False, method="lbfgs", maxiter=1000)
    full = smf.mixedlm(full_formula, d, groups=d["孕妇代码"]).fit(reml=False, method="lbfgs", maxiter=1000)
    lr = max(0.0, 2.0 * (full.llf - base.llf))
    df = len(full.fe_params) - len(base.fe_params)
    comparison = {
        "base_aic": float(base.aic),
        "full_aic": float(full.aic),
        "likelihood_ratio": lr,
        "df": int(df),
        "p_value": float(chi2.sf(lr, df)),
        "base_residual_sd": float(np.sqrt(base.scale)),
        "base_random_intercept_sd": float(np.sqrt(base.cov_re.iloc[0, 0])),
    }
    return base, full, comparison


def interval_table(male: pd.DataFrame, threshold: float = 0.04) -> pd.DataFrame:
    rows = []
    for code, g in male.sort_values("孕周数").groupby("孕妇代码"):
        g = g.sort_values("孕周数")
        hit = g["Y染色体浓度"].to_numpy() >= threshold
        weeks = g["孕周数"].to_numpy(float)
        if hit.any():
            first = int(np.argmax(hit))
            upper = weeks[first]
            previous_below = weeks[:first][~hit[:first]]
            lower = float(previous_below.max()) if len(previous_below) else 0.0
            censoring = "interval" if lower > 0 else "left"
        else:
            lower, upper, censoring = float(weeks.max()), np.inf, "right"
        first_row = g.iloc[0]
        rows.append({
            "孕妇代码": code,
            "BMI": float(first_row["孕妇BMI"]),
            "年龄": float(first_row["年龄"]),
            "身高": float(first_row["身高"]),
            "IVF": float(first_row["IVF妊娠"] != "自然受孕"),
            "lower": lower,
            "upper": upper,
            "censoring": censoring,
        })
    return pd.DataFrame(rows)


def fit_lognormal_aft(intervals: pd.DataFrame, X: np.ndarray) -> dict:
    lower = intervals["lower"].to_numpy(float)
    upper = intervals["upper"].to_numpy(float)
    X = np.asarray(X, float)

    def nll(theta: np.ndarray) -> float:
        beta, sigma = theta[:-1], np.exp(theta[-1])
        mu = X @ beta
        prob = np.empty(len(lower), dtype=float)
        left = (lower <= 0) & np.isfinite(upper)
        right = np.isinf(upper)
        inter = ~(left | right)
        prob[left] = norm.cdf((np.log(upper[left]) - mu[left]) / sigma)
        prob[right] = norm.sf((np.log(lower[right]) - mu[right]) / sigma)
        a = (np.log(lower[inter]) - mu[inter]) / sigma
        b = (np.log(upper[inter]) - mu[inter]) / sigma
        prob[inter] = norm.cdf(b) - norm.cdf(a)
        return float(-np.log(np.maximum(prob, 1e-14)).sum())

    init = np.r_[np.log(13.0), np.zeros(X.shape[1] - 1), np.log(0.25)]
    res = minimize(nll, init, method="BFGS", options={"maxiter": 3000, "gtol": 1e-7})
    return {
        "beta": res.x[:-1],
        "sigma": float(np.exp(res.x[-1])),
        "loglik": float(-res.fun),
        "success": bool(res.success or np.linalg.norm(res.jac) < 1e-3),
        "message": str(res.message),
    }


def choose_bmi_groups(intervals: pd.DataFrame) -> dict:
    bmi = intervals["BMI"].to_numpy()
    candidates = sorted(set(float(np.quantile(bmi, q)) for q in np.arange(0.15, 0.86, 0.10)))
    best = None
    all_models = []
    for groups in (2, 3, 4):
        for cuts in itertools.combinations(candidates, groups - 1):
            labels = np.digitize(bmi, cuts)
            counts = np.bincount(labels, minlength=groups)
            if counts.min() < 35:
                continue
            X = np.column_stack([np.ones(len(bmi))] + [(labels == j).astype(float) for j in range(1, groups)])
            fit = fit_lognormal_aft(intervals, X)
            k = X.shape[1] + 1
            bic = -2.0 * fit["loglik"] + k * np.log(len(intervals))
            item = {"groups": groups, "cuts": cuts, "labels": labels, "counts": counts, "fit": fit, "bic": float(bic)}
            all_models.append(item)
            if best is None or bic < best["bic"]:
                best = item
    if best is None:
        raise RuntimeError("没有满足最小组样本量约束的 BMI 分组")
    return {"best": best, "candidate_bic": sorted((m["groups"], m["bic"]) for m in all_models)}


def group_time_summary(intervals: pd.DataFrame, grouping: dict) -> pd.DataFrame:
    best = grouping["best"]
    labels, fit = best["labels"], best["fit"]
    beta, sigma = fit["beta"], fit["sigma"]
    rows = []
    cuts = [-np.inf, *best["cuts"], np.inf]
    for j in range(best["groups"]):
        x = np.r_[1.0, [(1.0 if j == k else 0.0) for k in range(1, best["groups"])]]
        mu = float(x @ beta)
        row = {
            "组": j + 1,
            "BMI下界": cuts[j],
            "BMI上界": cuts[j + 1],
            "人数": int((labels == j).sum()),
            "中位达标孕周": float(np.exp(mu)),
            "90%达标孕周": float(np.exp(mu + sigma * norm.ppf(0.90))),
            "95%达标孕周": float(np.exp(mu + sigma * norm.ppf(0.95))),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def measurement_error_sensitivity(male: pd.DataFrame, cuts: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for threshold in (0.035, 0.040, 0.045):
        intervals = interval_table(male, threshold)
        labels = np.digitize(intervals["BMI"].to_numpy(), cuts)
        groups = len(cuts) + 1
        X = np.column_stack([np.ones(len(labels))] + [(labels == j).astype(float) for j in range(1, groups)])
        fit = fit_lognormal_aft(intervals, X)
        for j in range(groups):
            x = np.r_[1.0, [(1.0 if j == k else 0.0) for k in range(1, groups)]]
            mu = float(x @ fit["beta"])
            rows.append({"阈值": threshold, "组": j + 1, "90%达标孕周": float(np.exp(mu + fit["sigma"] * norm.ppf(0.90)))})
    return pd.DataFrame(rows)


def multifactor_group_aft(intervals: pd.DataFrame, grouping: dict) -> tuple[dict, pd.DataFrame]:
    labels = grouping["best"]["labels"]
    groups = grouping["best"]["groups"]
    age_z = (intervals["年龄"].to_numpy() - intervals["年龄"].mean()) / intervals["年龄"].std()
    height_z = (intervals["身高"].to_numpy() - intervals["身高"].mean()) / intervals["身高"].std()
    ivf_z = (intervals["IVF"].to_numpy() - intervals["IVF"].mean()) / max(intervals["IVF"].std(), 1e-12)
    X = np.column_stack(
        [np.ones(len(labels))]
        + [(labels == j).astype(float) for j in range(1, groups)]
        + [age_z, height_z, ivf_z]
    )
    fit = fit_lognormal_aft(intervals, X)
    k = X.shape[1] + 1
    fit["bic"] = float(-2 * fit["loglik"] + k * np.log(len(intervals)))
    rows = []
    cuts = [-np.inf, *grouping["best"]["cuts"], np.inf]
    for j in range(groups):
        # Age, height and IVF are held at their sample means (standardized value 0).
        x = np.r_[1.0, [(1.0 if j == g else 0.0) for g in range(1, groups)], 0.0, 0.0, 0.0]
        mu = float(x @ fit["beta"])
        t90 = float(np.exp(mu + fit["sigma"] * norm.ppf(0.90)))
        rows.append({
            "组": j + 1,
            "BMI下界": cuts[j],
            "BMI上界": cuts[j + 1],
            "多因素90%达标孕周": t90,
            "建议检测时点(向上取整到天)": float(np.ceil(t90 * 7) / 7),
        })
    return fit, pd.DataFrame(rows)


def firth_logistic(X: np.ndarray, y: np.ndarray) -> dict:
    X1 = np.column_stack([np.ones(len(X)), X])

    def objective(beta: np.ndarray) -> float:
        eta = X1 @ beta
        p = expit(eta)
        ll = np.sum(y * np.log(np.maximum(p, 1e-15)) + (1 - y) * np.log(np.maximum(1 - p, 1e-15)))
        W = np.maximum(p * (1 - p), 1e-10)
        info = X1.T @ (W[:, None] * X1)
        sign, logdet = np.linalg.slogdet(info + 1e-9 * np.eye(info.shape[0]))
        penalty = 0.5 * logdet if sign > 0 else -1e6
        return -(ll + penalty)

    res = minimize(objective, np.zeros(X1.shape[1]), method="BFGS")
    return {"coef": res.x, "success": bool(res.success), "message": str(res.message)}


def female_classification(female: pd.DataFrame) -> tuple[dict, dict]:
    labels_text = female["染色体的非整倍体"].fillna("").astype(str)
    specs = {
        "T13": ("13号染色体的Z值", "13号染色体的GC含量"),
        "T18": ("18号染色体的Z值", "18号染色体的GC含量"),
        "T21": ("21号染色体的Z值", "21号染色体的GC含量"),
    }
    metrics, curves = {}, {}
    for target, (zcol, gccol) in specs.items():
        y = labels_text.str.contains(target).astype(int).to_numpy()
        qc = (
            np.abs(female[gccol].to_numpy(float) - 0.40)
            + female["被过滤掉读段数的比例"].to_numpy(float)
            + female["重复读段的比例"].to_numpy(float)
        )
        X = np.column_stack([female[zcol].to_numpy(float), female["X染色体的Z值"].to_numpy(float), qc])
        groups = female["孕妇代码"].to_numpy()
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2025)
        prob = np.zeros(len(y))
        for train, test in splitter.split(X, y, groups):
            scaler = StandardScaler().fit(X[train])
            model = LogisticRegression(C=0.25, class_weight="balanced", max_iter=3000)
            model.fit(scaler.transform(X[train]), y[train])
            prob[test] = model.predict_proba(scaler.transform(X[test]))[:, 1]
        pred = prob >= 0.5
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        scaler_all = StandardScaler().fit(X)
        firth = firth_logistic(scaler_all.transform(X), y)
        metrics[target] = {
            "positive_rows": int(y.sum()),
            "positive_women": int(pd.Series(groups[y == 1]).nunique()),
            "roc_auc_group_cv": float(roc_auc_score(y, prob)),
            "pr_auc_group_cv": float(average_precision_score(y, prob)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
            "firth_standardized_coef_intercept_z_xz_qc": firth["coef"].tolist(),
        }
        fpr, tpr, _ = roc_curve(y, prob)
        curves[target] = (fpr, tpr, metrics[target]["roc_auc_group_cv"])
    return metrics, curves


def make_plots(male: pd.DataFrame, intervals: pd.DataFrame, grouping: dict, times: pd.DataFrame, curves: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    sample_codes = male["孕妇代码"].drop_duplicates().iloc[::8]
    for _, g in male[male["孕妇代码"].isin(sample_codes)].groupby("孕妇代码"):
        axes[0].plot(g["孕周数"], g["Y染色体浓度"] * 100, color="#8ecae6", alpha=0.28, lw=0.7)
    bins = pd.cut(male["孕周数"], np.arange(10, 26.1, 2))
    emp = male.groupby(bins, observed=True).agg(孕周=("孕周数", "mean"), Y=("Y染色体浓度", "mean"))
    axes[0].plot(emp["孕周"], emp["Y"] * 100, "o-", color="#9b2226", lw=2, label="两周分箱均值")
    axes[0].axhline(4, ls="--", color="#ee9b00", label="4% 阈值")
    axes[0].set(xlabel="孕周", ylabel="Y 染色体浓度 / %", title="男胎重复测量轨迹")
    axes[0].legend(frameon=False)

    best = grouping["best"]
    beta, sigma = best["fit"]["beta"], best["fit"]["sigma"]
    grid = np.linspace(10, 25, 300)
    for j in range(best["groups"]):
        x = np.r_[1.0, [(1.0 if j == k else 0.0) for k in range(1, best["groups"])]]
        mu = x @ beta
        cdf = norm.cdf((np.log(grid) - mu) / sigma)
        lower = times.loc[j, "BMI下界"]
        upper = times.loc[j, "BMI上界"]
        if np.isneginf(lower):
            label = f"组{j + 1}: BMI < {upper:.1f}"
        elif np.isposinf(upper):
            label = f"组{j + 1}: BMI ≥ {lower:.1f}"
        else:
            label = f"组{j + 1}: {lower:.1f} ≤ BMI < {upper:.1f}"
        axes[1].plot(grid, cdf, lw=2, label=label)
        axes[1].axvline(times.loc[j, "90%达标孕周"], lw=0.8, alpha=0.5)
    axes[1].axhline(0.9, color="#555555", ls="--", lw=1)
    axes[1].set(xlabel="孕周", ylabel="估计达标比例", ylim=(0, 1.02), title="区间删失 AFT 达标曲线")
    axes[1].legend(frameon=False, fontsize=9)
    fig.savefig(OUT_DIR / "male_longitudinal_and_timing.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    for target, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=2, label=f"{target}（AUC={auc:.3f}）")
    ax.plot([0, 1], [0, 1], "--", color="#777777")
    ax.set(xlabel="假阳性率", ylabel="真阳性率", title="女胎 AB 标签：按孕妇分组交叉验证", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False)
    fig.savefig(OUT_DIR / "female_group_cv_roc.png", dpi=200)
    plt.close(fig)


def main() -> None:
    male_raw, female = load_data()
    male = prepare_male(male_raw)
    base, full, mixed_comparison = fit_mixed_models(male)
    intervals = interval_table(male, 0.04)
    continuous_X = np.column_stack([np.ones(len(intervals)), (intervals["BMI"] - intervals["BMI"].mean()) / intervals["BMI"].std()])
    aft_continuous = fit_lognormal_aft(intervals, continuous_X)
    grouping = choose_bmi_groups(intervals)
    times = group_time_summary(intervals, grouping)
    multifactor_fit, multifactor_times = multifactor_group_aft(intervals, grouping)
    sensitivity = measurement_error_sensitivity(male, grouping["best"]["cuts"])
    female_metrics, curves = female_classification(female)
    make_plots(male, intervals, grouping, times, curves)

    coef_table = pd.DataFrame({
        "系数": base.fe_params,
        "标准误": base.bse_fe,
        "p值": base.pvalues[base.fe_params.index],
    })
    coef_table.to_csv(OUT_DIR / "mixed_model_coefficients.csv", encoding="utf-8-sig")
    times.to_csv(OUT_DIR / "bmi_groups_and_timing.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(OUT_DIR / "measurement_error_sensitivity.csv", index=False, encoding="utf-8-sig")
    multifactor_times.to_csv(OUT_DIR / "multifactor_timing.csv", index=False, encoding="utf-8-sig")

    best = grouping["best"]
    output = {
        "sample": {
            "male_rows": len(male), "male_women": int(male["孕妇代码"].nunique()),
            "female_rows": len(female), "female_women": int(female["孕妇代码"].nunique()),
            "censoring_counts": intervals["censoring"].value_counts().to_dict(),
        },
        "mixed_model_comparison": mixed_comparison,
        "base_mixed_fixed_effects": {k: {"coef": float(base.fe_params[k]), "p": float(base.pvalues[k])} for k in base.fe_params.index},
        "continuous_bmi_aft": {
            "beta_intercept_bmi_standardized": aft_continuous["beta"].tolist(),
            "sigma": aft_continuous["sigma"], "loglik": aft_continuous["loglik"],
        },
        "bmi_grouping": {
            "groups": best["groups"], "cuts": list(best["cuts"]), "counts": best["counts"].tolist(), "bic": best["bic"],
            "aft_beta": best["fit"]["beta"].tolist(), "aft_sigma": best["fit"]["sigma"],
            "timing": times.replace({np.inf: None, -np.inf: None}).to_dict("records"),
        },
        "measurement_error_sensitivity": sensitivity.to_dict("records"),
        "multifactor_aft": {
            "beta_group_age_height_ivf": multifactor_fit["beta"].tolist(),
            "sigma": multifactor_fit["sigma"],
            "bic": multifactor_fit["bic"],
            "group_only_bic": best["bic"],
            "timing": multifactor_times.replace({np.inf: None, -np.inf: None}).to_dict("records"),
        },
        "female_classification": female_metrics,
        "interpretation_guardrail": "女胎 AE 列全为健康；AB 只能作为检测输出复核标签，不能据此验证真实胎儿异常诊断。",
    }
    (OUT_DIR / "c_results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
