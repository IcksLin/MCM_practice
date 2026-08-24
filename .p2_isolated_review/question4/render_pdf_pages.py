"""将Word导出的PDF逐页渲染为PNG，供报告视觉质检。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pypdfium2 as pdfium


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(args.pdf))
    scale = args.dpi / 72.0
    for i in range(len(pdf)):
        bitmap = pdf[i].render(scale=scale)
        bitmap.to_pil().save(args.out / f"page-{i+1:02d}.png", dpi=(args.dpi, args.dpi))
    print(f"pages={len(pdf)}")


if __name__ == "__main__":
    main()
