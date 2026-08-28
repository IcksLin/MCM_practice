from __future__ import annotations

import sys
from pathlib import Path

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "docx_tools"))
import paper_format as pf


if __name__ == "__main__":
    path = ROOT / "docs" / "B题完整论文.docx"
    doc = Document(path)
    issues = pf.validate_paper_structure(
        doc,
        contest="cumcm",
        min_content_units=4500,
        min_equations=10,
        min_figures=9,
        min_tables=4,
        rendered_pages=15,
        target_pages=15,
        official_max_pages=30,
    )
    if issues:
        print("\n".join(issues))
        raise SystemExit(2)
    print("PASS: 精简论文结构、图表引用、文献对应、篇幅和页数均满足项目覆盖阈值。")
