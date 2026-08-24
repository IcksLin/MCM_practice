from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.special import expit
from scipy.stats import gumbel_l, logistic, norm
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "C题" / "data" / "raw" / "附件.xlsx"
OUT = ROOT / "C题" / "output" / "archive" / "legacy_analysis" / "传统机器学习重构"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 2025
TEST_WINDOW = (10.0, 25.0)
TIME_GRID = np.arange(TEST_WINDOW[0], TEST_WINDOW[1] + 1 / 14, 1 / 14)

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfcfe",
        "axes.edgecolor": "#667085",
        "axes.titleweight": "bold",
        "grid.alpha": 0.25,
    }
)


def parse_week(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.fullmatch(r"\s*(\d+)[wW](?:\+(\d+))?\s*", str(value))
    if not match:
        raise ValueError(f"无法解析孕周：{value!r}")
    return float(match.group(1)) + float(match.group(2) or 0) / 7.0


def numeric_count(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"\d+", str(value))
    return float(match.group()) if match else np.nan


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    male = pd.read_excel(SOURCE, sheet_name="男胎检测数据")
    female = pd.read_excel(SOURCE, sheet_name="女胎检测数据")
    for data in (male, female):
        data["孕周"] = data["检测孕周"].map(parse_week)
        data["IVF"] = (data["IVF妊娠"] != "自然受孕").astype(float)
        data["孕次"] = data["怀孕次数"].map(numeric_count)
        data["产次"] = data["生产次数"].map(numeric_count)
        data["log原始读段"] = np.log1p(data["原始读段数"])
        unique = "唯一比对的读段数  " if "唯一比对的读段数  " in data else "唯一比对的读段数"
        data["log唯一读段"] = np.log1p(data[unique])
    return male, female


STATIC_FEATURES = ["孕妇BMI", "年龄", "身高", "体重", "IVF", "孕次", "产次"]
FEMALE_FEATURES = [
    "孕周",
    *STATIC_FEATURES,
    "13号染色体的Z值",
    "18号染色体的Z值",
    "21号染色体的Z值",
    "X染色体的Z值",
    "log原始读段",
    "在参考基因组上比对的比例",
    "重复读段的比例",
    "log唯一读段",
    "GC含量",
    "13号染色体的GC含量",
    "18号染色体的GC含量",
    "21号染色体的GC含量",
    "被过滤掉读段数的比例",
    "X染色体浓度",
]


def interval_table(male: pd.DataFrame, threshold: float = 0.04) -> pd.DataFrame:
    """Build one interval-censored earliest-attainment label per woman.

    Technical replicates at the same gestational week are aggregated by their
    median Y concentration, preventing row order from defining the event time.
    """
    rows: list[dict[str, object]] = []
    for code, woman in male.groupby("孕妇代码", sort=False):
        weekly = (
            woman.groupby("孕周", as_index=False)
            .agg(Y=("Y染色体浓度", "median"))
            .sort_values("孕周")
        )
        weeks = weekly["孕周"].to_numpy(float)
        hit = weekly["Y"].to_numpy(float) >= threshold
        if hit.any():
            first = int(np.argmax(hit))
            upper = float(weeks[first])
            previous_below = weeks[:first][~hit[:first]]
            lower = float(previous_below.max()) if len(previous_below) else 0.0
            censoring = "interval" if lower > 0 else "left"
        else:
            lower, upper, censoring = float(weeks.max()), np.inf, "right"

        row: dict[str, object] = {
            "孕妇代码": code,
            "lower": lower,
            "upper": upper,
            "censoring": censoring,
            "检测次数": int(len(woman)),
            "不同孕周数": int(len(weekly)),
        }
        for feature in STATIC_FEATURES:
            row[feature] = float(woman[feature].median())
        rows.append(row)
    return pd.DataFrame(rows)


def set_aft_bounds(matrix: xgb.DMatrix, lower: np.ndarray, upper: np.ndarray) -> None:
    matrix.set_float_info("label_lower_bound", np.asarray(lower, np.float32))
    matrix.set_float_info("label_upper_bound", np.asarray(upper, np.float32))


def sample_aft_params(trial: optuna.Trial) -> dict[str, object]:
    return {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 40.0, log=True),
        "eta": trial.suggest_float("eta", 0.008, 0.16, log=True),
        "subsample": trial.suggest_float("subsample", 0.55, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-3, 100.0, log=True),
        "aft_loss_distribution": trial.suggest_categorical(
            "aft_loss_distribution", ["normal", "logistic", "extreme"]
        ),
        "aft_loss_distribution_scale": trial.suggest_float(
            "aft_loss_distribution_scale", 0.3, 3.0, log=True
        ),
        "seed": SEED,
        "nthread": 4,
        "verbosity": 0,
    }


def train_aft_fold(
    X_train: pd.DataFrame,
    lower_train: np.ndarray,
    upper_train: np.ndarray,
    X_valid: pd.DataFrame,
    lower_valid: np.ndarray,
    upper_valid: np.ndarray,
    params: dict[str, object],
) -> tuple[xgb.Booster, float, int]:
    dtrain = xgb.DMatrix(X_train, feature_names=list(X_train.columns))
    dvalid = xgb.DMatrix(X_valid, feature_names=list(X_valid.columns))
    set_aft_bounds(dtrain, lower_train, upper_train)
    set_aft_bounds(dvalid, lower_valid, upper_valid)
    history: dict[str, dict[str, list[float]]] = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2500,
        evals=[(dvalid, "valid")],
        evals_result=history,
        early_stopping_rounds=80,
        verbose_eval=False,
    )
    score = float(history["valid"]["aft-nloglik"][model.best_iteration])
    return model, score, int(model.best_iteration + 1)


def tune_aft(
    X: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    trials: int,
    seed: int,
    folds: int = 3,
) -> tuple[dict[str, object], int, float]:
    censoring = np.where(lower <= 0, "left", np.where(np.isinf(upper), "right", "interval"))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = sample_aft_params(trial)
        scores, rounds = [], []
        for train, valid in splitter.split(X, censoring):
            _, score, best_round = train_aft_fold(
                X.iloc[train], lower[train], upper[train],
                X.iloc[valid], lower[valid], upper[valid], params,
            )
            scores.append(score)
            rounds.append(best_round)
        trial.set_user_attr("best_round", int(np.median(rounds)))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "tree_method": "hist",
        "seed": SEED,
        "nthread": 4,
        "verbosity": 0,
        **study.best_params,
    }
    return params, int(study.best_trial.user_attrs["best_round"]), float(study.best_value)


def interval_concordance(lower: np.ndarray, upper: np.ndarray, prediction: np.ndarray) -> float:
    concordant = comparable = 0.0
    for i, j in itertools.combinations(range(len(lower)), 2):
        if upper[i] < lower[j]:
            comparable += 1
            concordant += float(prediction[i] < prediction[j]) + 0.5 * float(prediction[i] == prediction[j])
        elif upper[j] < lower[i]:
            comparable += 1
            concordant += float(prediction[j] < prediction[i]) + 0.5 * float(prediction[i] == prediction[j])
    return float(concordant / comparable) if comparable else np.nan


def interval_distance(lower: np.ndarray, upper: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    below = np.maximum(lower - prediction, 0.0)
    above = np.where(np.isfinite(upper), np.maximum(prediction - upper, 0.0), 0.0)
    return below + above


def fit_nested_aft(
    intervals: pd.DataFrame,
    trials: int,
) -> tuple[dict[str, object], xgb.Booster, np.ndarray, np.ndarray]:
    X = intervals[STATIC_FEATURES].copy()
    lower = intervals["lower"].to_numpy(float)
    upper = intervals["upper"].to_numpy(float)
    outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    prediction = np.full(len(intervals), np.nan)
    probability = np.full((len(intervals), len(TIME_GRID)), np.nan)
    fold_scores: list[float] = []
    fold_params: list[dict[str, object]] = []

    censoring = intervals["censoring"].to_numpy()
    for fold, (train, valid) in enumerate(outer.split(X, censoring), start=1):
        print(f"  AFT outer fold {fold}/4: {trials} TPE trials", flush=True)
        params, _, _ = tune_aft(
            X.iloc[train].reset_index(drop=True), lower[train], upper[train],
            trials=trials, seed=SEED + fold,
        )
        model, score, _ = train_aft_fold(
            X.iloc[train], lower[train], upper[train],
            X.iloc[valid], lower[valid], upper[valid], params,
        )
        prediction[valid] = model.predict(xgb.DMatrix(X.iloc[valid], feature_names=STATIC_FEATURES))
        probability[valid] = np.column_stack(
            [aft_cdf(prediction[valid], week, params) for week in TIME_GRID]
        )
        fold_scores.append(score)
        fold_params.append(params)

    print(f"  AFT final search: {max(trials * 2, 20)} TPE trials", flush=True)
    final_params, final_rounds, final_cv = tune_aft(
        X, lower, upper, trials=max(trials * 2, 20), seed=SEED + 99, folds=4
    )
    dall = xgb.DMatrix(X, feature_names=STATIC_FEATURES)
    set_aft_bounds(dall, lower, upper)
    final_model = xgb.train(final_params, dall, num_boost_round=final_rounds, verbose_eval=False)

    distance = interval_distance(lower, upper, prediction)
    metrics = {
        "outer_aft_nloglik_mean": float(np.mean(fold_scores)),
        "outer_aft_nloglik_sd": float(np.std(fold_scores)),
        "interval_concordance": interval_concordance(lower, upper, prediction),
        "prediction_inside_observed_interval_rate": float(np.mean((prediction >= lower) & (prediction <= upper))),
        "median_distance_to_observed_interval_weeks": float(np.median(distance)),
        "final_cv_aft_nloglik": final_cv,
        "final_boost_rounds": final_rounds,
        "final_params": final_params,
        "outer_params": fold_params,
        "final_feature_importance_gain": {
            feature: float(final_model.get_score(importance_type="gain").get(feature, 0.0))
            for feature in STATIC_FEATURES
        },
    }
    return metrics, final_model, prediction, probability


def aft_cdf(predicted_time: np.ndarray, week: float, params: dict[str, object]) -> np.ndarray:
    scale = float(params["aft_loss_distribution_scale"])
    z = (np.log(week) - np.log(np.clip(predicted_time, 1e-6, None))) / scale
    distribution = str(params["aft_loss_distribution"])
    if distribution == "normal":
        return norm.cdf(z)
    if distribution == "logistic":
        return logistic.cdf(z)
    return gumbel_l.cdf(z)


def individual_t90(probability: np.ndarray) -> np.ndarray:
    values = np.full(len(probability), TEST_WINDOW[1])
    for i, probs in enumerate(probability):
        hit = np.flatnonzero(probs >= 0.90)
        if len(hit):
            values[i] = TIME_GRID[hit[0]]
    return values


def choose_bmi_groups(
    intervals: pd.DataFrame,
    oof_probability: np.ndarray,
    min_group: int = 30,
    export_search: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    bmi = intervals["孕妇BMI"].to_numpy(float)
    t90 = individual_t90(oof_probability)
    candidates = sorted(set(float(np.quantile(bmi, q)) for q in np.arange(0.15, 0.86, 0.05)))
    search_rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for groups in (2, 3, 4):
        for cuts in itertools.combinations(candidates, groups - 1):
            labels = np.digitize(bmi, cuts)
            counts = np.bincount(labels, minlength=groups)
            if counts.min() < min_group:
                continue
            sse = sum(float(np.sum((t90[labels == group] - np.mean(t90[labels == group])) ** 2)) for group in range(groups))
            bic = len(bmi) * math.log(max(sse / len(bmi), 1e-8)) + groups * math.log(len(bmi))
            item = {"groups": groups, "cuts": cuts, "counts": counts, "bic": bic, "labels": labels}
            search_rows.append({"组数": groups, "切点": "|".join(f"{v:.3f}" for v in cuts), "BIC": bic})
            if best is None or bic < float(best["bic"]):
                best = item
    if best is None:
        raise RuntimeError("没有找到满足最小组人数约束的 BMI 分组")

    labels = np.asarray(best["labels"])
    cuts = [-np.inf, *best["cuts"], np.inf]
    rows = []
    for group in range(int(best["groups"])):
        mask = labels == group
        mean_prob = np.mean(oof_probability[mask], axis=0)
        hit90 = np.flatnonzero(mean_prob >= 0.90)
        hit95 = np.flatnonzero(mean_prob >= 0.95)
        time90 = float(TIME_GRID[hit90[0]]) if len(hit90) else np.nan
        time95 = float(TIME_GRID[hit95[0]]) if len(hit95) else np.nan
        rows.append(
            {
                "组": group + 1,
                "BMI下界": cuts[group],
                "BMI上界": cuts[group + 1],
                "人数": int(mask.sum()),
                "平均预测90%达标孕周": time90,
                "建议检测时点_向上取整到天": float(np.ceil(time90 * 7) / 7) if np.isfinite(time90) else np.nan,
                "平均预测95%达标孕周": time95,
                "25周平均达标概率": float(mean_prob[-1]),
            }
        )
    details = {
        "groups": int(best["groups"]),
        "cuts": [float(value) for value in best["cuts"]],
        "counts": np.asarray(best["counts"]).astype(int).tolist(),
        "bic": float(best["bic"]),
        "candidate_count": len(search_rows),
    }
    if export_search:
        pd.DataFrame(search_rows).sort_values("BIC").to_csv(OUT / "BMI分组搜索.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows), details


def assess_group_stability(
    intervals: pd.DataFrame,
    params: dict[str, object],
    repeats: int = 8,
) -> tuple[pd.DataFrame, dict[str, object]]:
    X = intervals[STATIC_FEATURES]
    lower = intervals["lower"].to_numpy(float)
    upper = intervals["upper"].to_numpy(float)
    censoring = intervals["censoring"].to_numpy()
    rows = []
    for repeat in range(repeats):
        print(f"  BMI stability repeat {repeat + 1}/{repeats}", flush=True)
        splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 300 + repeat)
        probability = np.full((len(intervals), len(TIME_GRID)), np.nan)
        for train, valid in splitter.split(X, censoring):
            model, _, _ = train_aft_fold(
                X.iloc[train], lower[train], upper[train],
                X.iloc[valid], lower[valid], upper[valid], params,
            )
            predicted = model.predict(xgb.DMatrix(X.iloc[valid], feature_names=STATIC_FEATURES))
            probability[valid] = np.column_stack([aft_cdf(predicted, week, params) for week in TIME_GRID])
        timing, grouping = choose_bmi_groups(intervals, probability, export_search=False)
        rows.append(
            {
                "重复": repeat + 1,
                "组数": grouping["groups"],
                "切点": "|".join(f"{value:.3f}" for value in grouping["cuts"]),
                "各组人数": "|".join(str(value) for value in grouping["counts"]),
                "建议时点": "|".join(
                    "NA" if pd.isna(value) else f"{value:.3f}"
                    for value in timing["建议检测时点_向上取整到天"]
                ),
                "BIC": grouping["bic"],
            }
        )
    frame = pd.DataFrame(rows)
    group_frequency = {str(int(key)): int(value) for key, value in frame["组数"].value_counts().items()}
    cut_frequency = {str(key): int(value) for key, value in frame["切点"].value_counts().items()}
    summary = {
        "repeats": repeats,
        "group_count_frequency": group_frequency,
        "exact_cut_frequency": cut_frequency,
        "stable_three_group_rule": bool(
            group_frequency.get("3", 0) >= math.ceil(0.75 * repeats)
            and frame.loc[frame["组数"] == 3, "切点"].nunique() <= 3
        ),
    }
    return frame, summary


def aggregate_woman_metric(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"group": groups, "y": y, "p": p})
    aggregated = frame.groupby("group").agg(y=("y", "max"), p=("p", "max"))
    return aggregated["y"].to_numpy(int), aggregated["p"].to_numpy(float)


def cat_params(trial: optuna.Trial, positive_ratio: float) -> dict[str, object]:
    weight_power = trial.suggest_float("weight_power", 0.35, 1.0)
    return {
        "iterations": 1800,
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.18, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 100.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "rsm": trial.suggest_float("rsm", 0.6, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 192),
        "class_weights": [1.0, positive_ratio**weight_power],
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": SEED,
        "thread_count": 4,
        "verbose": False,
        "allow_writing_files": False,
    }


def tune_catboost(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    trials: int,
    seed: int,
) -> dict[str, object]:
    positive_ratio = float(np.sum(y == 0) / max(np.sum(y == 1), 1))
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = cat_params(trial, positive_ratio)
        scores = []
        for train, valid in splitter.split(X, y, groups):
            model = CatBoostClassifier(**params)
            model.fit(X.iloc[train], y[train], eval_set=(X.iloc[valid], y[valid]), early_stopping_rounds=80)
            pred = model.predict_proba(X.iloc[valid])[:, 1]
            y_woman, p_woman = aggregate_woman_metric(y[valid], pred, groups[valid])
            scores.append(average_precision_score(y_woman, p_woman))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    clean = dict(study.best_params)
    weight_power = float(clean.pop("weight_power"))
    return {
        "iterations": 1800,
        **clean,
        "class_weights": [1.0, positive_ratio**weight_power],
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": SEED,
        "thread_count": 4,
        "verbose": False,
        "allow_writing_files": False,
    }


def logistic_oof(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    prediction = np.full(len(y), np.nan)
    for train, valid in splits:
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)),
            ]
        )
        model.fit(X.iloc[train], y[train])
        prediction[valid] = model.predict_proba(X.iloc[valid])[:, 1]
    return prediction


def classification_metrics(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    y_woman, p_woman = aggregate_woman_metric(y, p, groups)
    rng = np.random.default_rng(SEED)
    auc_samples, pr_samples = [], []
    for _ in range(1000):
        index = rng.integers(0, len(y_woman), len(y_woman))
        if np.unique(y_woman[index]).size == 2:
            auc_samples.append(roc_auc_score(y_woman[index], p_woman[index]))
            pr_samples.append(average_precision_score(y_woman[index], p_woman[index]))

    candidates = np.unique(np.r_[0.0, p_woman, 1.0])
    operating = None
    for threshold in candidates:
        predicted = p_woman >= threshold
        sensitivity = float(np.sum(predicted & (y_woman == 1)) / max(np.sum(y_woman == 1), 1))
        specificity = float(np.sum(~predicted & (y_woman == 0)) / max(np.sum(y_woman == 0), 1))
        if sensitivity >= 0.90 and (operating is None or specificity > operating["specificity"]):
            operating = {"threshold": float(threshold), "sensitivity": sensitivity, "specificity": specificity}
    if operating is None:
        operating = {"threshold": 0.0, "sensitivity": 1.0, "specificity": 0.0}

    return {
        "woman_roc_auc": float(roc_auc_score(y_woman, p_woman)),
        "woman_roc_auc_bootstrap_95ci": [float(value) for value in np.quantile(auc_samples, [0.025, 0.975])],
        "woman_pr_auc": float(average_precision_score(y_woman, p_woman)),
        "woman_pr_auc_bootstrap_95ci": [float(value) for value in np.quantile(pr_samples, [0.025, 0.975])],
        "woman_brier": float(brier_score_loss(y_woman, p_woman)),
        "woman_log_loss": float(log_loss(y_woman, np.clip(p_woman, 1e-6, 1 - 1e-6))),
        "record_roc_auc": float(roc_auc_score(y, p)),
        "record_pr_auc": float(average_precision_score(y, p)),
        "screening_operating_point_at_least_90pct_sensitivity": operating,
    }


def fit_female_models(female: pd.DataFrame, trials: int) -> dict[str, object]:
    X = female[FEMALE_FEATURES].copy()
    groups = female["孕妇代码"].to_numpy()
    label_text = female["染色体的非整倍体"].fillna("").astype(str)
    all_results: dict[str, object] = {}
    prediction_rows = []
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    for index, target in enumerate(("T13", "T18", "T21")):
        y = label_text.str.contains(target).astype(int).to_numpy()
        splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + index)
        splits = list(splitter.split(X, y, groups))
        logit_pred = logistic_oof(X, y, groups, splits)

        cat_pred = np.full(len(y), np.nan)
        params_by_fold = []
        for fold, (train, valid) in enumerate(splits, start=1):
            print(f"  {target} CatBoost outer fold {fold}/4: {trials} TPE trials", flush=True)
            params = tune_catboost(
                X.iloc[train].reset_index(drop=True), y[train], groups[train],
                trials=trials, seed=SEED + index * 10 + fold,
            )
            model = CatBoostClassifier(**params)
            model.fit(X.iloc[train], y[train], eval_set=(X.iloc[valid], y[valid]), early_stopping_rounds=100)
            cat_pred[valid] = model.predict_proba(X.iloc[valid])[:, 1]
            model.save_model(OUT / f"{target}_catboost_fold{fold}.cbm")
            params_by_fold.append(params)

        logit_metrics = classification_metrics(y, logit_pred, groups)
        cat_metrics = classification_metrics(y, cat_pred, groups)
        selected = "CatBoost" if cat_metrics["woman_pr_auc"] > logit_metrics["woman_pr_auc"] else "RegularizedLogit"
        selected_pred = cat_pred if selected == "CatBoost" else logit_pred
        y_woman, p_woman = aggregate_woman_metric(y, selected_pred, groups)
        frac, mean = calibration_curve(y_woman, p_woman, n_bins=5, strategy="quantile")
        axes[index].plot(mean, frac, "o-", color="#2e90a5", label=selected)
        axes[index].plot([0, 1], [0, 1], "--", color="#667085")
        axes[index].set(
            title=f"{target} 患者级校准",
            xlabel="预测概率",
            ylabel="实际阳性比例",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axes[index].grid(True)
        axes[index].legend(frameon=False)
        all_results[target] = {
            "positive_records": int(y.sum()),
            "positive_women": int(pd.Series(groups[y == 1]).nunique()),
            "RegularizedLogit": logit_metrics,
            "CatBoost": cat_metrics,
            "selected_model": selected,
            "catboost_outer_params": params_by_fold,
        }
        sequence_column = "样本序号" if "样本序号" in female.columns else "序号"
        for row_index in range(len(female)):
            prediction_rows.append(
                {
                    "目标": target,
                    "样本序号": female.iloc[row_index][sequence_column],
                    "孕妇代码": groups[row_index],
                    "标签": int(y[row_index]),
                    "Logit_OOF概率": float(logit_pred[row_index]),
                    "CatBoost_OOF概率": float(cat_pred[row_index]),
                    "入选模型": selected,
                }
            )

        if selected == "RegularizedLogit":
            final_logit = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)),
                ]
            )
            final_logit.fit(X, y)
            joblib.dump(final_logit, OUT / f"{target}_regularized_logit.joblib")

        (OUT / "female_checkpoint.json").write_text(
            json.dumps(serializable(all_results), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pd.DataFrame(prediction_rows).to_csv(OUT / "女胎OOF预测.csv", index=False, encoding="utf-8-sig")

    figure.suptitle("女胎 AB 输出：严格按孕妇分组的外层验证", fontsize=16)
    figure.savefig(OUT / "女胎患者级校准.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(prediction_rows).to_csv(OUT / "女胎OOF预测.csv", index=False, encoding="utf-8-sig")
    return all_results


def plot_aft(intervals: pd.DataFrame, predicted: np.ndarray, timing: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    finite_upper = np.where(np.isfinite(intervals["upper"]), intervals["upper"], TEST_WINDOW[1])
    axes[0].scatter(finite_upper, predicted, c=intervals["孕妇BMI"], cmap="viridis", s=25, alpha=0.75)
    axes[0].plot([8, 26], [8, 26], "--", color="#667085")
    axes[0].set(
        title="AFT 外层验证：预测时间与观测上界",
        xlabel="观测区间上界（右删失截为25周）",
        ylabel="OOF预测达标时间",
        xlim=(8, 26),
        ylim=(8, 26),
    )
    axes[0].grid(True)

    labels = []
    values = []
    for row in timing.itertuples(index=False):
        if np.isneginf(row.BMI下界):
            label = f"BMI < {row.BMI上界:.1f}"
        elif np.isposinf(row.BMI上界):
            label = f"BMI ≥ {row.BMI下界:.1f}"
        else:
            label = f"{row.BMI下界:.1f}–{row.BMI上界:.1f}"
        labels.append(label)
        values.append(row.建议检测时点_向上取整到天)
    bars = axes[1].bar(labels, values, color="#2e90a5")
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].axhline(12, ls="--", color="#f79009", label="早期/中期边界")
    axes[1].set(title="BMI分组与建议检测时点", ylabel="孕周", ylim=(9, 25.5))
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(True, axis="y")
    axes[1].legend(frameon=False)
    fig.savefig(OUT / "AFT验证与BMI时点.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def sensitivity_thresholds(
    male: pd.DataFrame,
    cuts: list[float],
    params: dict[str, object],
    rounds: int,
) -> pd.DataFrame:
    rows = []
    for threshold in (0.035, 0.040, 0.045):
        intervals = interval_table(male, threshold)
        X = intervals[STATIC_FEATURES]
        lower = intervals["lower"].to_numpy(float)
        upper = intervals["upper"].to_numpy(float)
        matrix = xgb.DMatrix(X, feature_names=STATIC_FEATURES)
        set_aft_bounds(matrix, lower, upper)
        model = xgb.train(params, matrix, num_boost_round=rounds, verbose_eval=False)
        prediction = model.predict(matrix)
        labels = np.digitize(intervals["孕妇BMI"].to_numpy(float), cuts)
        for group in range(len(cuts) + 1):
            mask = labels == group
            probs = np.asarray([np.mean(aft_cdf(prediction[mask], week, params)) for week in TIME_GRID])
            hit = np.flatnonzero(probs >= 0.90)
            week = float(TIME_GRID[hit[0]]) if len(hit) else np.nan
            rows.append(
                {
                    "Y浓度阈值": threshold,
                    "组": group + 1,
                    "人数": int(mask.sum()),
                    "90%达标孕周": week,
                    "建议时点_向上取整到天": float(np.ceil(week * 7) / 7) if np.isfinite(week) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def serializable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="C题传统机器学习重构：区间删失 AFT + 患者级 CatBoost")
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--aft-trials", type=int)
    parser.add_argument("--class-trials", type=int)
    parser.add_argument("--skip-female", action="store_true")
    parser.add_argument("--stability-only", action="store_true")
    args = parser.parse_args()
    aft_trials = args.aft_trials or (8 if args.mode == "pilot" else 40)
    class_trials = args.class_trials or (5 if args.mode == "pilot" else 25)

    print("[1/5] 读取数据并构造患者级区间删失标签", flush=True)
    male, female = load_data()
    intervals = interval_table(male, 0.04)
    intervals.to_csv(OUT / "患者级删失区间.csv", index=False, encoding="utf-8-sig")

    if args.stability_only:
        result_path = OUT / "tree_results.json"
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        stability, stability_summary = assess_group_stability(
            intervals, existing["aft"]["final_params"]
        )
        stability.to_csv(OUT / "BMI分组稳定性.csv", index=False, encoding="utf-8-sig")
        existing["bmi_grouping_stability"] = stability_summary
        result_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(stability_summary, ensure_ascii=False, indent=2), flush=True)
        return

    print("[2/5] 嵌套交叉验证 XGBoost AFT", flush=True)
    aft_metrics, aft_model, oof_prediction, oof_probability = fit_nested_aft(intervals, aft_trials)
    aft_model.save_model(OUT / "xgboost_aft_model.json")
    full_prediction = aft_model.predict(xgb.DMatrix(intervals[STATIC_FEATURES], feature_names=STATIC_FEATURES))

    print("[3/5] 搜索 BMI 分组并计算建议时点", flush=True)
    timing, grouping = choose_bmi_groups(
        intervals,
        oof_probability,
    )
    timing.to_csv(OUT / "BMI分组与时点.csv", index=False, encoding="utf-8-sig")
    sensitivity = sensitivity_thresholds(
        male,
        grouping["cuts"],
        aft_metrics["final_params"],
        int(aft_metrics["final_boost_rounds"]),
    )
    sensitivity.to_csv(OUT / "检测误差敏感性.csv", index=False, encoding="utf-8-sig")
    plot_aft(intervals, oof_prediction, timing)
    (OUT / "aft_checkpoint.json").write_text(
        json.dumps(
            serializable(
                {
                    "sample": {
                        "male_records": len(male),
                        "male_women": int(male["孕妇代码"].nunique()),
                        "censoring": intervals["censoring"].value_counts().to_dict(),
                    },
                    "aft": aft_metrics,
                    "bmi_grouping": grouping,
                    "timing": timing.replace({np.inf: None, -np.inf: None, np.nan: None}).to_dict("records"),
                    "threshold_sensitivity": sensitivity.replace({np.nan: None}).to_dict("records"),
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[4/5] 女胎 AB 标签患者级分类", flush=True)
    female_results = {} if args.skip_female else fit_female_models(female, class_trials)

    print("[5/5] 导出机器可读结果", flush=True)
    output = {
        "run_mode": args.mode,
        "search_budget": {"aft_trials_per_outer_fold": aft_trials, "catboost_trials_per_outer_fold": class_trials},
        "sample": {
            "male_records": len(male),
            "male_women": int(male["孕妇代码"].nunique()),
            "female_records": len(female),
            "female_women": int(female["孕妇代码"].nunique()),
            "censoring": intervals["censoring"].value_counts().to_dict(),
        },
        "aft": aft_metrics,
        "bmi_grouping": grouping,
        "timing": timing.replace({np.inf: None, -np.inf: None, np.nan: None}).to_dict("records"),
        "threshold_sensitivity": sensitivity.replace({np.nan: None}).to_dict("records"),
        "female": female_results,
        "guardrails": [
            "所有外层和内层划分按孕妇隔离；同一孕妇的重复记录不会跨折。",
            "问题2/3时点模型只使用检测前可获得的人口学与孕情变量，不使用同次测序QC。",
            "女胎模型复核AB检测输出，不把附件AE列包装为真实临床诊断终点。",
            "最终建议严格限制在题目给出的10至25周检测窗口。",
        ],
    }
    (OUT / "tree_results.json").write_text(
        json.dumps(serializable(output), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(serializable(output), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
