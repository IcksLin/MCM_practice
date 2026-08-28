from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "B题总思路表.md"
BUILD = ROOT / "output" / "thought_table_pdf"
TEX = BUILD / "B题总思路表.tex"
PDF = ROOT / "docs" / "B题总思路表.pdf"


def escape_plain(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "⁻": r"\textsuperscript{-}", "⁴": r"\textsuperscript{4}",
        "₀": r"\textsubscript{0}", "₁": r"\textsubscript{1}",
        "₂": r"\textsubscript{2}", "₅": r"\textsubscript{5}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def inline(text: str) -> str:
    tokens = re.split(r"(`[^`]*`|\*\*[^*]+\*\*)", text)
    rendered: list[str] = []
    for token in tokens:
        if token.startswith("`") and token.endswith("`"):
            rendered.append(r"\texttt{" + escape_plain(token[1:-1]) + "}")
        elif token.startswith("**") and token.endswith("**"):
            rendered.append(r"\textbf{" + escape_plain(token[2:-2]) + "}")
        else:
            rendered.append(escape_plain(token))
    return "".join(rendered)


def table_rows(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        # A pipe inside inline code is literal Markdown content, not a column
        # separator (for example ``|d10-d15|``).
        cells, cell, in_code = [], [], False
        for char in line.strip().strip("|"):
            if char == "`":
                in_code = not in_code
                cell.append(char)
            elif char == "|" and not in_code:
                cells.append("".join(cell).strip())
                cell = []
            else:
                cell.append(char)
        cells.append("".join(cell).strip())
        if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    return rows


def render_table(lines: list[str]) -> str:
    rows = table_rows(lines)
    cols = len(rows[0])
    landscape = cols >= 5
    usable = 0.92 if not landscape else 0.94
    width = usable / cols
    # Subtract each column's tab padding so rules and padding do not push the
    # longtable beyond the available line width.
    spec = "|" + "|".join([
        rf">{{\raggedright\arraybackslash}}p{{\dimexpr {width:.4f}\linewidth-2\tabcolsep\relax}}"
    ] * cols) + "|"
    out = []
    if landscape:
        out.append(r"\begin{landscape}")
    out.extend([r"\begin{center}", r"\scriptsize", rf"\begin{{longtable}}{{{spec}}}", r"\hline"])
    for index, row in enumerate(rows):
        cells = [inline(cell) for cell in row]
        if index == 0:
            cells = [r"\textbf{" + cell + "}" for cell in cells]
        out.append(" & ".join(cells) + r" \\ \hline")
        if index == 0:
            out.append(r"\endfirsthead")
            out.append(r"\hline " + " & ".join(cells) + r" \\ \hline")
            out.append(r"\endhead")
    out.extend([r"\end{longtable}", r"\normalsize", r"\end{center}"])
    if landscape:
        out.append(r"\end{landscape}")
    return "\n".join(out)


def convert(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    in_math = False
    in_code = False
    list_kind: str | None = None

    def flush_paragraph() -> None:
        if paragraph:
            out.append(inline(" ".join(part.strip() for part in paragraph)))
            out.append("\n")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            out.append(rf"\end{{{list_kind}}}")
            list_kind = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped == "\\[":
            flush_paragraph(); close_list(); in_math = True; out.append(r"\[")
        elif stripped == "\\]":
            in_math = False; out.append(r"\]")
        elif in_math:
            out.append(line)
        elif stripped.startswith("```"):
            flush_paragraph(); close_list()
            if not in_code:
                out.append(r"\begin{lstlisting}"); in_code = True
            else:
                out.append(r"\end{lstlisting}"); in_code = False
        elif in_code:
            # The selected monospaced CJK font lacks Unicode super/subscript
            # glyphs; keep code-flow diagrams legible with ASCII equivalents.
            out.append(line.translate(str.maketrans({"⁻": "-", "¹": "1", "₂": "2"})))
        elif stripped.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            flush_paragraph(); close_list()
            block = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            i -= 1
            out.append(render_table(block))
        elif match := re.match(r"^(#{1,4})\s+(.*)$", stripped):
            flush_paragraph(); close_list()
            level = len(match.group(1)); title = inline(match.group(2))
            command = {1: "section", 2: "section", 3: "subsection", 4: "subsubsection"}[level]
            if level == 1:
                out.append(rf"\begin{{center}}\LARGE\bfseries {title}\end{{center}}")
            else:
                out.append(rf"\{command}*{{{title}}}")
        elif stripped.startswith(">"):
            flush_paragraph(); close_list()
            out.append(r"\begin{quote}\itshape " + inline(stripped[1:].strip()) + r"\end{quote}")
        elif match := re.match(r"^[-*]\s+(.*)$", stripped):
            flush_paragraph()
            if list_kind != "itemize":
                close_list(); out.append(r"\begin{itemize}"); list_kind = "itemize"
            out.append(r"\item " + inline(match.group(1)))
        elif match := re.match(r"^\d+\.\s+(.*)$", stripped):
            flush_paragraph()
            if list_kind != "enumerate":
                close_list(); out.append(r"\begin{enumerate}"); list_kind = "enumerate"
            out.append(r"\item " + inline(match.group(1)))
        elif not stripped:
            flush_paragraph(); close_list()
        else:
            paragraph.append(line)
        i += 1
    flush_paragraph(); close_list()
    return "\n".join(out)


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    body = convert(SOURCE.read_text(encoding="utf-8"))
    tex = r"""\documentclass[11pt,a4paper]{ctexart}
\usepackage[margin=1.65cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,array,pdflscape,xcolor,enumitem,listings,hyperref}
\setmainfont{Times New Roman}
\setsansfont{Microsoft YaHei}
\setmonofont{Microsoft YaHei}
\hypersetup{hidelinks,pdfauthor={数学建模项目},pdftitle={B题总思路表}}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.35em}
\setlength{\LTpre}{0.35em}
\setlength{\LTpost}{0.35em}
\renewcommand{\arraystretch}{1.28}
\setlist{nosep,leftmargin=2.2em}
\lstset{basicstyle=\ttfamily\small,breaklines=true,frame=single,columns=fullflexible}
\pagestyle{plain}
\begin{document}
""" + body + "\n" + r"\end{document}" + "\n"
    TEX.write_text(tex, encoding="utf-8")
    command = ["xelatex", "-interaction=nonstopmode", "-halt-on-error", TEX.name]
    for _ in range(2):
        completed = subprocess.run(command, cwd=BUILD, text=True, encoding="utf-8", errors="replace")
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
    built_pdf = BUILD / TEX.with_suffix(".pdf").name
    PDF.write_bytes(built_pdf.read_bytes())
    print(PDF)


if __name__ == "__main__":
    main()
