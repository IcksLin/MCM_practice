"""从权威思路表生成、审计并发布 PDF。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "C题总思路表.md"
BUILD = ROOT / "output" / "latex" / "thoughts"
PDF = ROOT / "output" / "reports" / "C题总思路表.pdf"
LATEXMK = shutil.which("latexmk") or r"D:\develop_env\texlive\2026\bin\windows\latexmk.exe"
BLOCKERS = ("Overfull", "Underfull", "Missing character", "LaTeX Warning", "Package Warning")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "render_thoughts_tex.py")], check=True)
    subprocess.run([LATEXMK, "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=BUILD, check=True)
    log = (BUILD / "main.log").read_text(encoding="utf-8", errors="replace")
    found = [token for token in BLOCKERS if token in log]
    if found:
        raise RuntimeError(f"LaTeX日志仍有阻断项: {found}")
    PDF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUILD / "main.pdf", PDF)
    manifest = {
        "status": "passed",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "xelatex via latexmk",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "tex_sha256": sha256(BUILD / "main.tex"),
        "pdf_sha256": sha256(PDF),
        "warning_blockers": found,
    }
    PDF.with_suffix(".pdf.build.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run([LATEXMK, "-c", "main.tex"], cwd=BUILD, check=True)
    for generated in (BUILD / "main.pdf", BUILD / "main.xdv"):
        generated.unlink(missing_ok=True)
    print(f"published: {PDF}")


if __name__ == "__main__":
    main()
