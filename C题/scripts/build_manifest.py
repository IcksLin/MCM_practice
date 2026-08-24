"""生成精简工程的可追溯文件清单。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "manifests" / "project_manifest.json"
INCLUDE = ("data", "docs", "algorithms", "scripts", "output/results", "output/reports", "output/figures")
IGNORE_SUFFIXES = {".pyc", ".aux", ".log", ".out", ".toc", ".xdv", ".fls", ".fdb_latexmk"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    files = []
    for folder in INCLUDE:
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix.lower() not in IGNORE_SUFFIXES and "__pycache__" not in path.parts:
                files.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": ROOT.name,
        "reproduction": {
            "light": "python scripts/run_all.py",
            "full": "python scripts/run_all.py --full",
        },
        "file_count": len(files),
        "files": files,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {OUT} ({len(files)} files)")


if __name__ == "__main__":
    main()
