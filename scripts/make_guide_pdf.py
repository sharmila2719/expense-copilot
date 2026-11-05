"""Render PROJECT_GUIDE.md into a clean, multi-page PDF (PROJECT_GUIDE.pdf).

A lightweight Markdown-to-PDF renderer using reportlab (pure Python, no system
deps). It handles the subset of Markdown used in the guide: headings (#..####),
bullet lists, fenced code blocks, tables (| a | b |), bold (**x**), inline code
(`x`), horizontal rules (---), and blockquotes (>).

Usage:
    python -m scripts.make_guide_pdf
"""
from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "PROJECT_GUIDE.md"
OUT = ROOT / "PROJECT_GUIDE.pdf"

NAVY = colors.HexColor("#1b2a4a")
ACCENT = colors.HexColor("#2a6f97")
CODE_BG = colors.HexColor("#f4f6f8")
GRID = colors.HexColor("#c9d3df")


def make_styles():
    base = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("h1", parent=base["Title"], fontSize=19, textColor=NAVY,
                             spaceAfter=6, leading=23),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=13.5, textColor=NAVY,
                             spaceBefore=12, spaceAfter=4, leading=16),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=11, textColor=ACCENT,
                             spaceBefore=7, spaceAfter=2, leading=14),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9.2, leading=13,
                               spaceAfter=4),
        "bullet": ParagraphStyle("bullet", parent=base["Normal"], fontSize=9.2, leading=12.5),
        "quote": ParagraphStyle("quote", parent=base["Normal"], fontSize=9.2, leading=13,
                                leftIndent=12, textColor=colors.HexColor("#444"),
                                borderColor=ACCENT, spaceAfter=4, fontName="Helvetica-Oblique"),
        "code": ParagraphStyle("code", parent=base["Code"], fontSize=7.6, leading=9.6,
                               textColor=colors.HexColor("#1a1a1a")),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=8.3, leading=11),
        "cellh": ParagraphStyle("cellh", parent=base["Normal"], fontSize=8.3, leading=11,
                                textColor=colors.white, fontName="Helvetica-Bold"),
    }


def inline(text: str) -> str:
    """Convert a subset of inline markdown to reportlab markup (escaped)."""
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier" size="8">\1</font>', text)
    return text


def main() -> None:
    s = make_styles()
    lines = SRC.read_text(encoding="utf-8").splitlines()
    flow = []
    i = 0
    n = len(lines)

    def flush_bullets(items):
        if items:
            flow.append(ListFlowable(
                [ListItem(Paragraph(inline(t), s["bullet"]), leftIndent=10) for t in items],
                bulletType="bullet", start="•", leftIndent=14, spaceAfter=5,
            ))

    while i < n:
        line = lines[i]

        # Fenced code block
        if line.strip().startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            flow.append(Preformatted("\n".join(buf), s["code"]))
            flow.append(Spacer(1, 4))
            continue

        # Table block (consecutive lines containing |)
        if "|" in line and line.strip().startswith("|"):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            # Drop the separator row (---)
            parsed = []
            for r in rows:
                cells = [c.strip() for c in r.strip().strip("|").split("|")]
                if all(set(c) <= set("-: ") for c in cells):
                    continue
                parsed.append(cells)
            if parsed:
                header, *bodyrows = parsed
                data = [[Paragraph(inline(c), s["cellh"]) for c in header]]
                data += [[Paragraph(inline(c), s["cell"]) for c in row] for row in bodyrows]
                t = Table(data, hAlign="LEFT")
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                    ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CODE_BG]),
                ]))
                flow.append(t)
                flow.append(Spacer(1, 6))
            continue

        # Headings
        if line.startswith("#### "):
            flow.append(Paragraph(inline(line[5:]), s["h3"]))
        elif line.startswith("### "):
            flow.append(Paragraph(inline(line[4:]), s["h3"]))
        elif line.startswith("## "):
            flow.append(Paragraph(inline(line[3:]), s["h2"]))
        elif line.startswith("# "):
            flow.append(Paragraph(inline(line[2:]), s["h1"]))
        # Horizontal rule
        elif line.strip() in ("---", "***", "___"):
            flow.append(Spacer(1, 2))
            flow.append(HRFlowable(width="100%", thickness=0.6, color=GRID))
            flow.append(Spacer(1, 2))
        # Blockquote
        elif line.startswith("> "):
            flow.append(Paragraph(inline(line[2:]), s["quote"]))
        # Bullet list
        elif re.match(r"^\s*[-*] ", line):
            items = []
            while i < n and re.match(r"^\s*[-*] ", lines[i]):
                items.append(re.sub(r"^\s*[-*] ", "", lines[i]))
                i += 1
            flush_bullets(items)
            continue
        # Numbered list -> render as plain paragraphs with the number
        elif re.match(r"^\s*\d+\. ", line):
            flow.append(Paragraph(inline(line.strip()), s["body"]))
        # Blank line
        elif not line.strip():
            flow.append(Spacer(1, 3))
        # Normal paragraph
        else:
            flow.append(Paragraph(inline(line), s["body"]))

        i += 1

    doc = SimpleDocTemplate(
        str(OUT), pagesize=letter,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.6 * inch, bottomMargin=0.55 * inch,
        title="Expense Copilot — Complete Project Guide",
    )
    doc.build(flow)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
