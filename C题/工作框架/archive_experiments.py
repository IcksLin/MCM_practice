from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ACTIVE = ROOT / "outputs" / "C题" / "传统机器学习重构"
ARCHIVE = ROOT / "outputs" / "C题" / "_archive" / "20260824_决策优化前"

FILES = [
    ("aft_checkpoint.json", "checkpoints", "AFT运行中断恢复文件"),
    ("female_checkpoint.json", "checkpoints", "女胎运行中断恢复文件"),
    *[(f"T13_catboost_fold{i}.cbm", "未入选模型/T13_CatBoost", "T13最终回退正则逻辑回归") for i in range(1, 5)],
    *[(f"T21_catboost_fold{i}.cbm", "未入选模型/T21", "T21无稳定判别信号") for i in range(1, 5)],
    ("T21_regularized_logit.joblib", "未入选模型/T21", "T21仅保留结果指标，不作为正式判定模型"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def main() -> None:
    manifest = ARCHIVE / "归档清单.csv"
    all_archived = all(
        not (ACTIVE / filename).exists() and (ARCHIVE / category / filename).exists()
        for filename, category, _ in FILES
    )
    if manifest.exists() and all_archived:
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            existing = list(csv.DictReader(stream))
        if len(existing) != len(FILES):
            raise RuntimeError("已有归档清单条目数与配置不一致")
        for row in existing:
            target = ROOT / row["归档路径"]
            if not target.exists() or sha256(target) != row["SHA256"]:
                raise RuntimeError(f"已有归档校验失败：{row['归档路径']}")
        print(f"archive_already_verified={len(existing)} manifest={relative(manifest)}")
        return

    rows = []
    for filename, category, reason in FILES:
        source = ACTIVE / filename
        target = ARCHIVE / category / filename
        target.parent.mkdir(parents=True, exist_ok=True)

        if source.exists() and target.exists():
            if sha256(source) != sha256(target):
                raise RuntimeError(f"源文件和归档文件冲突：{filename}")
            status = "already_archived_duplicate_source_retained"
        elif source.exists():
            before_hash = sha256(source)
            size = source.stat().st_size
            shutil.move(str(source), str(target))
            if sha256(target) != before_hash:
                raise RuntimeError(f"移动后哈希变化：{filename}")
            status = "moved"
            rows.append(
                {
                    "原路径": relative(source),
                    "归档路径": relative(target),
                    "类别": category,
                    "原因": reason,
                    "字节数": size,
                    "SHA256": before_hash,
                    "状态": status,
                }
            )
            continue
        elif target.exists():
            status = "already_archived"
        else:
            raise FileNotFoundError(f"待归档文件不存在：{filename}")

        rows.append(
            {
                "原路径": relative(source),
                "归档路径": relative(target),
                "类别": category,
                "原因": reason,
                "字节数": target.stat().st_size if target.exists() else source.stat().st_size,
                "SHA256": sha256(target if target.exists() else source),
                "状态": status,
            }
        )

    with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["原路径", "归档路径", "类别", "原因", "字节数", "SHA256", "状态"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"archived={len(rows)} manifest={relative(manifest)}")


if __name__ == "__main__":
    main()
