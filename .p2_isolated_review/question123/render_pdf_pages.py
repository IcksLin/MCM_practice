#!/usr/bin/env python3
"""用工作区运行时的 PDFium 把Word导出的PDF渲染为逐页PNG。"""

from pathlib import Path
import pypdfium2 as pdfium

root = Path(__file__).resolve().parents[1]
qa = root / "doc" / "_qa_q123_render"
pdf = qa / "C题问题1-3统合报告_v1.pdf"
doc = pdfium.PdfDocument(str(pdf))
for i in range(len(doc)):
    page = doc[i]
    bitmap = page.render(scale=2.0)
    image = bitmap.to_pil()
    image.save(qa / f"page-{i + 1}.png")
print(f"pages={len(doc)} output={qa}")
