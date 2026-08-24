from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


SEED = 2025
ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK_DIR = Path(__file__).resolve().parent
SOURCE_XLSX = ROOT / "C题" / "C题" / "附件.xlsx"
SOURCE_PDF = ROOT / "C题" / "C题" / "C题.pdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_week(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.fullmatch(r"\s*(\d+)[wW](?:\+(\d+))?\s*", str(value))
    if not match:
        raise ValueError(f"无法解析孕周：{value!r}")
    return float(match.group(1)) + float(match.group(2) or 0) / 7.0


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    male = pd.read_excel(SOURCE_XLSX, sheet_name="男胎检测数据")
    female = pd.read_excel(SOURCE_XLSX, sheet_name="女胎检测数据")
    male["孕周"] = male["检测孕周"].map(parse_week)
    female["孕周"] = female["检测孕周"].map(parse_week)
    return male, female


def build_male_intervals(male: pd.DataFrame, threshold: float = 0.04) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code, woman in male.groupby("孕妇代码", sort=False):
        weekly = (
            woman.groupby("孕周", as_index=False)
            .agg(Y=("Y染色体浓度", "median"))
            .sort_values("孕周", kind="mergesort")
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
        rows.append(
            {
                "孕妇代码": code,
                "lower": lower,
                "upper": upper,
                "censoring": censoring,
                "原始记录数": int(len(woman)),
                "不同孕周数": int(len(weekly)),
            }
        )
    return pd.DataFrame(rows).sort_values("孕妇代码", kind="mergesort").reset_index(drop=True)


def freeze_male_folds(intervals: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    result = intervals.copy()
    result["outer_fold"] = -1
    for outer in range(4):
        result[f"inner_fold_when_outer_{outer}"] = -1

    labels = result["censoring"].to_numpy()
    outer_splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    fold_composition: dict[str, object] = {}
    for outer, (train, test) in enumerate(outer_splitter.split(result, labels)):
        result.loc[test, "outer_fold"] = outer
        inner_splitter = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=SEED + outer + 1
        )
        train_labels = labels[train]
        for inner, (_, valid_local) in enumerate(inner_splitter.split(train, train_labels)):
            valid_global = train[valid_local]
            result.loc[valid_global, f"inner_fold_when_outer_{outer}"] = inner
        test_counts = result.loc[test, "censoring"].value_counts().sort_index().to_dict()
        fold_composition[str(outer)] = {
            "test_women": int(len(test)),
            "train_women": int(len(train)),
            "test_censoring": {str(key): int(value) for key, value in test_counts.items()},
            "inner_seed": SEED + outer + 1,
        }

    assert (result["outer_fold"] >= 0).all()
    assert result["孕妇代码"].is_unique
    for outer in range(4):
        column = f"inner_fold_when_outer_{outer}"
        test_mask = result["outer_fold"] == outer
        assert (result.loc[test_mask, column] == -1).all()
        assert set(result.loc[~test_mask, column].unique()) == {0, 1, 2}
    return result, fold_composition


def freeze_female_folds(female: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    sequence_column = "样本序号" if "样本序号" in female.columns else "序号"
    label_text = female["染色体的非整倍体"].fillna("").astype(str)
    groups = female["孕妇代码"].to_numpy()
    all_frames = []
    fold_composition: dict[str, object] = {}

    for target_index, target in enumerate(("T13", "T18", "T21")):
        y = label_text.str.contains(target).astype(int).to_numpy()
        result = pd.DataFrame(
            {
                "目标": target,
                "样本序号": female[sequence_column].to_numpy(),
                "孕妇代码": groups,
                "标签": y,
                "outer_fold": -1,
            }
        )
        for outer in range(4):
            result[f"inner_fold_when_outer_{outer}"] = -1

        outer_seed = SEED + target_index
        outer_splitter = StratifiedGroupKFold(
            n_splits=4, shuffle=True, random_state=outer_seed
        )
        target_folds: dict[str, object] = {}
        for outer, (train, test) in enumerate(outer_splitter.split(female, y, groups)):
            result.loc[test, "outer_fold"] = outer
            inner_seed = SEED + target_index * 10 + outer + 1
            inner_splitter = StratifiedGroupKFold(
                n_splits=3, shuffle=True, random_state=inner_seed
            )
            train_y, train_groups = y[train], groups[train]
            for inner, (_, valid_local) in enumerate(
                inner_splitter.split(female.iloc[train], train_y, train_groups)
            ):
                valid_global = train[valid_local]
                result.loc[valid_global, f"inner_fold_when_outer_{outer}"] = inner

            train_women = int(pd.Series(groups[train]).nunique())
            test_women = int(pd.Series(groups[test]).nunique())
            target_folds[str(outer)] = {
                "test_records": int(len(test)),
                "test_women": test_women,
                "test_positive_women": int(pd.Series(groups[test][y[test] == 1]).nunique()),
                "train_records": int(len(train)),
                "train_women": train_women,
                "inner_seed": inner_seed,
            }

        assert (result["outer_fold"] >= 0).all()
        for outer in range(4):
            column = f"inner_fold_when_outer_{outer}"
            test_mask = result["outer_fold"] == outer
            assert (result.loc[test_mask, column] == -1).all()
            assert set(result.loc[~test_mask, column].unique()) == {0, 1, 2}

            outer_woman_counts = result.groupby("孕妇代码")["outer_fold"].nunique()
            assert int(outer_woman_counts.max()) == 1
            train_result = result.loc[~test_mask]
            inner_woman_counts = train_result.groupby("孕妇代码")[column].nunique()
            assert int(inner_woman_counts.max()) == 1

        fold_composition[target] = {
            "outer_seed": outer_seed,
            "positive_records": int(y.sum()),
            "positive_women": int(pd.Series(groups[y == 1]).nunique()),
            "folds": target_folds,
        }
        all_frames.append(result)

    return pd.concat(all_frames, ignore_index=True), fold_composition


def dataset_summary(data: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(data)),
        "women": int(data["孕妇代码"].nunique()),
        "columns": [str(column) for column in data.columns],
        "missing_count": {
            str(key): int(value) for key, value in data.isna().sum().items() if int(value) > 0
        },
    }


def main() -> None:
    FRAMEWORK_DIR.mkdir(parents=True, exist_ok=True)
    male, female = load_data()
    intervals = build_male_intervals(male)
    male_folds, male_composition = freeze_male_folds(intervals)
    female_folds, female_composition = freeze_female_folds(female)

    male_path = FRAMEWORK_DIR / "男胎患者级折号.csv"
    female_path = FRAMEWORK_DIR / "女胎患者级折号.csv"
    male_folds.to_csv(male_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    female_folds.to_csv(female_path, index=False, encoding="utf-8-sig", lineterminator="\n")

    manifest = {
        "schema_version": 1,
        "freeze_date": "2026-08-24",
        "base_seed": SEED,
        "source": {
            "xlsx": {
                "path": str(SOURCE_XLSX.relative_to(ROOT)).replace("\\", "/"),
                "bytes": SOURCE_XLSX.stat().st_size,
                "sha256": sha256(SOURCE_XLSX),
            },
            "pdf": {
                "path": str(SOURCE_PDF.relative_to(ROOT)).replace("\\", "/"),
                "bytes": SOURCE_PDF.stat().st_size,
                "sha256": sha256(SOURCE_PDF),
            },
        },
        "datasets": {
            "male": dataset_summary(male),
            "female": dataset_summary(female),
            "male_interval_labels": {
                "women": int(len(intervals)),
                "threshold": 0.04,
                "same_woman_same_week_aggregation": "median Y concentration",
                "censoring": {
                    str(key): int(value)
                    for key, value in intervals["censoring"].value_counts().items()
                },
            },
        },
        "validation": {
            "male": {
                "unit": "woman",
                "outer": "4-fold StratifiedKFold by censoring",
                "inner": "3-fold StratifiedKFold inside each outer training fold",
                "fold_composition": male_composition,
            },
            "female": {
                "training_unit": "record",
                "split_unit": "woman",
                "outer": "4-fold StratifiedGroupKFold per target",
                "inner": "3-fold StratifiedGroupKFold inside each outer training fold",
                "targets": female_composition,
            },
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scikit-learn": importlib.metadata.version("scikit-learn"),
        },
    }
    manifest_path = FRAMEWORK_DIR / "数据清单.json"
    write_json(manifest_path, manifest)

    generated = [
        Path(__file__),
        FRAMEWORK_DIR / "验证协议.md",
        manifest_path,
        male_path,
        female_path,
    ]
    checks = {
        "schema_version": 1,
        "checks_passed": {
            "source_files_exist": SOURCE_XLSX.exists() and SOURCE_PDF.exists(),
            "male_one_row_per_woman": bool(male_folds["孕妇代码"].is_unique),
            "male_outer_complete": bool((male_folds["outer_fold"] >= 0).all()),
            "female_outer_complete": bool((female_folds["outer_fold"] >= 0).all()),
            "female_rows_equal_three_targets": len(female_folds) == len(female) * 3,
        },
        "file_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in generated
            if path.exists()
        },
    }
    assert all(checks["checks_passed"].values())
    write_json(FRAMEWORK_DIR / "固化校验.json", checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
