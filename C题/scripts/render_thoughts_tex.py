"""将C题总思路表Markdown转换为独立ctexart项目。

仅实现本文档实际使用的Markdown子集：标题、段落、列表、管道表格、引用、
行间公式、行内代码和粗体。转换结果用于XeLaTeX编译，不修改权威Markdown。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "docs" / "C题总思路表.md"
DEFAULT_TARGET = ROOT / "output" / "latex" / "thoughts" / "main.tex"


def split_table_row(line: str) -> list[str]:
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    cells, current, in_code = [], [], False
    for char in text:
        if char == "`":
            in_code = not in_code
            current.append(char)
        elif char == "|" and not in_code:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    cells.append("".join(current).strip())
    return cells


def is_separator(line: str) -> bool:
    cells = split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def escape_plain(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "≤": r"$\leq$", "≥": r"$\geq$",
        "∈": r"$\in$", "α": r"$\alpha$", "β": r"$\beta$",
        "σ": r"$\sigma$", "ε": r"$\varepsilon$",
    }
    return "".join(replacements.get(char, char) for char in text)


def inline(text: str) -> str:
    tokens: list[str] = []

    def protect(pattern: str, repl):
        nonlocal text
        def sub(match):
            tokens.append(repl(match))
            return f"@@TOKEN{len(tokens)-1}@@"
        text = re.sub(pattern, sub, text)

    protect(r"`([^`]+)`", lambda m: r"\textsf{\seqsplit{" + escape_plain(m.group(1)) + "}}")
    protect(r"\$([^$]+)\$", lambda m: "$" + m.group(1) + "$")
    protect(r"\*\*([^*]+)\*\*", lambda m: r"\textbf{" + escape_plain(m.group(1)) + "}")
    text = escape_plain(text)
    for index, token in enumerate(tokens):
        text = text.replace(f"@@TOKEN{index}@@", token)
    return text


def table_to_tex(rows: list[list[str]]) -> list[str]:
    cols = max(len(row) for row in rows)
    normalized = [row + [""] * (cols - len(row)) for row in rows]
    width = max(0.068, 0.84 / cols)
    spec = "|" + "|".join(f">{{\\raggedright\\arraybackslash}}p{{{width:.3f}\\textwidth}}" for _ in range(cols)) + "|"
    font_size = r"\footnotesize" if cols <= 4 else (r"\scriptsize" if cols <= 6 else r"\tiny")
    out = [r"\begingroup", font_size, r"\setlength{\tabcolsep}{1.5pt}", rf"\begin{{longtable}}{{{spec}}}", r"\hline"]
    for index, row in enumerate(normalized):
        out.append(" & ".join(inline(cell) for cell in row) + r" \\ \hline")
        if index == 0:
            out.append(r"\endfirsthead")
            out.append(r"\hline")
            out.append(" & ".join(inline(cell) for cell in normalized[0]) + r" \\ \hline")
            out.append(r"\endhead")
    out.extend([r"\end{longtable}", r"\endgroup"])
    return out


def convert(markdown: str) -> str:
    lines = markdown.replace("\r\n", "\n").split("\n")
    body: list[str] = []
    paragraph: list[str] = []
    list_mode: str | None = None
    in_math = False
    in_fence = False
    fence_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            body.append(inline(" ".join(part.strip() for part in paragraph)))
            body.append("")
            paragraph = []

    def close_list() -> None:
        nonlocal list_mode
        if list_mode:
            body.append(rf"\end{{{list_mode}}}")
            body.append("")
            list_mode = None

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph(); close_list()
            if in_fence:
                body.extend([r"\begin{quote}\small\sffamily"])
                body.extend(escape_plain(value) + r"\\" for value in fence_lines)
                body.extend([r"\end{quote}", ""])
                fence_lines = []
                in_fence = False
            else:
                in_fence = True
            index += 1; continue
        if in_fence:
            fence_lines.append(line)
            index += 1; continue
        if stripped == r"\[":
            flush_paragraph(); close_list(); in_math = True; body.append(r"\["); index += 1; continue
        if in_math:
            body.append(line)
            if stripped == r"\]":
                in_math = False
            index += 1; continue
        if stripped.startswith("|") and index + 1 < len(lines) and is_separator(lines[index + 1]):
            flush_paragraph(); close_list()
            rows = [split_table_row(line)]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(split_table_row(lines[index])); index += 1
            body.extend(table_to_tex(rows)); body.append(""); continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph(); close_list()
            level = len(heading.group(1))
            command = {1: "section", 2: "section", 3: "subsection", 4: "subsubsection", 5: "paragraph", 6: "subparagraph"}[level]
            title = inline(heading.group(2))
            if level == 1:
                body.append(r"\begin{center}\LARGE\bfseries " + title + r"\end{center}")
            else:
                body.append(rf"\{command}{{{title}}}")
            index += 1; continue
        item = re.match(r"^[-*]\s+(.+)$", stripped)
        numbered = re.match(r"^\d+\.\s+(.+)$", stripped)
        if item or numbered:
            flush_paragraph()
            wanted = "itemize" if item else "enumerate"
            if list_mode != wanted:
                close_list(); body.append(rf"\begin{{{wanted}}}"); list_mode = wanted
            body.append(r"\item " + inline((item or numbered).group(1)))
            index += 1; continue
        if stripped.startswith(">"):
            flush_paragraph(); close_list(); body.append(r"\begin{quote}" + inline(stripped[1:].strip()) + r"\end{quote}"); index += 1; continue
        if not stripped:
            flush_paragraph(); close_list(); index += 1; continue
        paragraph.append(stripped); index += 1
    flush_paragraph(); close_list()
    return "\n".join(body)


PREAMBLE = r"""\documentclass[UTF8,12pt,a4paper]{ctexart}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,array,xcolor,hyperref,seqsplit}
\usepackage{enumitem}
\setlist{nosep,leftmargin=2em}
\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!60!black}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.35em}
\setlength{\emergencystretch}{3em}
\hbadness=10000
\sloppy
\setcounter{tocdepth}{3}
\title{C题建模总思路}
\author{}
\date{}
\begin{document}
\maketitle
\tableofcontents
\clearpage
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    source = args.source.resolve()
    target = args.target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8")
    target.write_text(PREAMBLE + convert(text) + "\n\\end{document}\n", encoding="utf-8")
    print(f"generated: {target}")


if __name__ == "__main__":
    main()
