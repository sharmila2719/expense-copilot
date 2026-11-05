"""Render the 1-page write-up to a clean PDF (WRITEUP.pdf).

Uses reportlab (pure-Python, no system dependencies) so it runs anywhere.

Usage:
    python -m scripts.make_writeup_pdf
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "WRITEUP.pdf"

NAVY = "#1b2a4a"
ACCENT = "#2a6f97"


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontSize=20, textColor=NAVY,
            spaceAfter=2, leading=24,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontSize=10, textColor=ACCENT,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontSize=12.5, textColor=NAVY,
            spaceBefore=9, spaceAfter=3, leading=15,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontSize=9.3, leading=13,
            alignment=TA_LEFT, spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["Normal"], fontSize=9.3, leading=12.5,
        ),
    }


def main() -> None:
    s = styles()
    doc = SimpleDocTemplate(
        str(OUT), pagesize=letter,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.6 * inch, bottomMargin=0.55 * inch,
        title="Expense Copilot — Write-up",
    )
    flow = []

    def para(text, style="body"):
        flow.append(Paragraph(text, s[style]))

    def bullets(items):
        flow.append(ListFlowable(
            [ListItem(Paragraph(t, s["bullet"]), leftIndent=10) for t in items],
            bulletType="bullet", start="•", leftIndent=12, spaceAfter=4,
        ))

    para("Expense Copilot", "title")
    para("AI receipt extraction &middot; RAG policy checks &middot; chat with your spend", "subtitle")

    para("The problem", "h2")
    para(
        "Corporate expense review is manual, slow, and inconsistent &mdash; someone reads "
        "each receipt, recalls the policy, and makes a judgment call. It is high-volume, "
        "rules-heavy, and language-heavy, which is a sweet spot for AI. I chose it because "
        "it sits squarely in the corporate-spend domain and lets me demonstrate the three "
        "techniques the challenge highlights (LLMs, RAG, and an agent) on one coherent, "
        "demoable workflow rather than a sprawling platform."
    )

    para("What I built", "h2")
    para(
        "<b>Expense Copilot</b> runs a receipt through a pipeline: "
        "<b>(1) Extraction</b> &mdash; structured fields (vendor, date, total, tax, category, "
        "line items); <b>(2) Policy decision via RAG</b> &mdash; retrieves the relevant policy "
        "clauses from a vector store and returns auto-approve / flag / reject <b>with citations</b>; "
        "<b>(3) Chat agent</b> &mdash; answers natural-language questions about spend and policy by "
        "calling analytics and retrieval tools and grounding its answer in real data."
    )
    para(
        "Stack: Python, FastAPI (API), Streamlit (UI), SQLite (storage), ChromaDB (vectors), "
        "sentence-transformers (embeddings), and Amazon Bedrock (Claude + Titan) as the cloud option."
    )

    para("Key decisions and trade-offs", "h2")
    bullets([
        "<b>Pluggable LLM provider.</b> The whole app depends on one interface with two "
        "implementations: a free offline <i>local</i> provider and an <i>Amazon Bedrock</i> provider. "
        "A reviewer can run and demo the app with zero credentials or cost, while the same code "
        "upgrades to real cloud vision and generation by flipping one env var.",
        "<b>RAG explains, rules decide.</b> Semantic retrieval surfaces the relevant clauses as "
        "human-readable citations, but the actual approve/flag/reject call runs on a structured "
        "ruleset (per-category caps, global approval threshold). In finance you want deterministic, "
        "auditable outcomes, not a decision that drifts with whichever chunk ranked highest.",
        "<b>Deterministic local generation.</b> The free provider gives genuine embeddings (so RAG "
        "is real) plus rule-based, reproducible answers. Free-form conversational prose needs Bedrock "
        "&mdash; an explicit, documented limitation rather than a silent degradation.",
        "<b>Scope discipline.</b> One clean workflow end-to-end; I skipped local image OCR (delegated "
        "to Bedrock vision), multi-user auth, and accounting integrations, all noted as next steps.",
    ])

    para("How I used AI tools", "h2")
    para(
        "I built this with Kiro (an agentic AI IDE) as the pair-programmer: I set the architecture "
        "and decisions and drove the agent to scaffold modules and write the extraction, RAG, agent, "
        "API, and UI layers. The most valuable moment was verification &mdash; after wiring everything, "
        "running the seed pipeline surfaced two real bugs (the extractor grabbed the &ldquo;Subtotal&rdquo; "
        "line as the total, and the first-draft policy logic rejected everything because a broad "
        "&ldquo;prohibited&rdquo; keyword leaked across clauses). That drove the redesign to the "
        "structured-ruleset approach. AI accelerated the plumbing; the engineering judgment "
        "(provider abstraction, rules-vs-RAG split, scope) was the human part."
    )

    para("What I'd do next", "h2")
    bullets([
        "Bedrock Claude tool-calling for the agent (true function calling instead of keyword routing) "
        "and LLM-assisted extraction with JSON-schema validation.",
        "Confidence-based human-in-the-loop review queue and an approvals workflow.",
        "Multi-currency normalization, duplicate-receipt detection, per-employee running totals against "
        "time-boxed limits (e.g. daily meal caps).",
        "Real image OCR in local mode and integrations with accounting systems; tests and evals on a "
        "labeled receipt set.",
    ])

    flow.append(Spacer(1, 6))
    doc.build(flow)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
