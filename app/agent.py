"""Chat agent: answer questions about expenses and policy.

Architecture — a lightweight tool-using agent:

    user question
        -> router classifies intent + extracts arguments (category, period...)
        -> runs one or more TOOLS over real data:
             * expenses_by_category   (SQL analytics)
             * spend_total            (SQL analytics)
             * decision_breakdown     (SQL analytics)
             * policy_lookup          (RAG retrieval over the policy)
             * list_flagged           (SQL filter)
        -> composes an answer:
             * Bedrock: LLM writes a natural answer grounded in the tool output
             * local:   deterministic templating of the tool output

This mirrors how a production agent works (tools + grounding) while staying
fully functional offline. Answers are always grounded in retrieved data, not
hallucinated.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from . import store
from .llm.base import LLMProvider
from .models import CATEGORIES, ChatResponse
from .policy import PolicyEngine

_CATEGORY_LOOKUP = {c.lower(): c for c in CATEGORIES}
_CATEGORY_ALIASES = {
    "travel": "Travel",
    "flight": "Travel",
    "flights": "Travel",
    "meal": "Meals & Entertainment",
    "meals": "Meals & Entertainment",
    "food": "Meals & Entertainment",
    "dining": "Meals & Entertainment",
    "hotel": "Lodging",
    "hotels": "Lodging",
    "lodging": "Lodging",
    "software": "Software & Subscriptions",
    "subscription": "Software & Subscriptions",
    "subscriptions": "Software & Subscriptions",
    "transport": "Transportation",
    "transportation": "Transportation",
    "office": "Office Supplies",
    "supplies": "Office Supplies",
}


class ExpenseAgent:
    def __init__(self, provider: LLMProvider, policy: PolicyEngine) -> None:
        self._provider = provider
        self._policy = policy

    # -- public -----------------------------------------------------------
    def answer(self, question: str) -> ChatResponse:
        q = question.lower().strip()
        tools_used: List[str] = []
        facts: List[str] = []

        category = self._detect_category(q)

        # --- tool selection (intent routing) ---
        if any(w in q for w in ["policy", "allowed", "limit", "cap", "rule", "reimburse", "can i"]):
            tools_used.append("policy_lookup")
            citations = self._policy.retrieve(question, k=3)
            if citations:
                facts.append("Relevant policy clauses:")
                for c in citations:
                    facts.append(f"- {c.clause}")
            else:
                facts.append("No policy clauses matched that question.")

        if any(w in q for w in ["flag", "flagged", "violat", "reject", "problem", "issue"]):
            tools_used.append("list_flagged")
            facts.append(self._list_flagged())

        if any(w in q for w in ["how much", "total", "spend", "spent", "sum", "cost"]):
            if category:
                tools_used.append("expenses_by_category")
                facts.append(self._spend_for_category(category))
            else:
                tools_used.append("spend_total")
                facts.append(self._spend_total())

        if any(w in q for w in ["breakdown", "by category", "categories", "summary", "overview"]):
            tools_used.append("expenses_by_category")
            facts.append(self._category_breakdown())

        if any(w in q for w in ["decision", "approved", "auto", "status"]):
            tools_used.append("decision_breakdown")
            facts.append(self._decision_breakdown())

        # Fallback: give a useful overview when nothing matched.
        if not tools_used:
            tools_used.extend(["spend_total", "decision_breakdown"])
            facts.append(self._spend_total())
            facts.append(self._decision_breakdown())

        answer = self._compose(question, facts)
        return ChatResponse(answer=answer, used_tools=tools_used)

    # -- tools ------------------------------------------------------------
    def _spend_total(self) -> str:
        return f"Total recorded spend across all expenses: {store.grand_total():,.2f}."

    def _spend_for_category(self, category: str) -> str:
        totals = store.totals_by_category()
        amount = totals.get(category, 0.0)
        return f"Total spend in {category}: {amount:,.2f}."

    def _category_breakdown(self) -> str:
        totals = store.totals_by_category()
        if not totals:
            return "No expenses recorded yet."
        lines = ["Spend by category:"]
        for cat, amt in sorted(totals.items(), key=lambda x: -x[1]):
            lines.append(f"- {cat}: {amt:,.2f}")
        return "\n".join(lines)

    def _decision_breakdown(self) -> str:
        counts = store.totals_by_decision()
        if not counts:
            return "No expenses recorded yet."
        pretty = {"auto_approve": "Auto-approved", "flag": "Flagged", "reject": "Rejected"}
        parts = [f"{pretty.get(k, k)}: {v}" for k, v in counts.items()]
        return "Decision breakdown — " + ", ".join(parts) + "."

    def _list_flagged(self) -> str:
        records = [r for r in store.list_expenses() if r.decision in ("flag", "reject")]
        if not records:
            return "No expenses are currently flagged or rejected."
        lines = [f"{len(records)} expense(s) need attention:"]
        for r in records:
            lines.append(
                f"- #{r.id} {r.vendor} ({r.category}) {r.currency} {r.total:,.2f} "
                f"-> {r.decision.upper()}: {r.decision_reason}"
            )
        return "\n".join(lines)

    # -- answer composition ----------------------------------------------
    def _compose(self, question: str, facts: List[str]) -> str:
        fact_block = "\n".join(facts).strip()

        if self._provider.name == "bedrock":
            system = (
                "You are Expense Copilot, a concise finance assistant. Answer the "
                "user's question using ONLY the facts provided. Do not invent "
                "numbers. If the facts don't cover it, say so."
            )
            prompt = f"Question: {question}\n\nFacts:\n{fact_block}\n\nAnswer:"
            try:
                return self._provider.generate(system, prompt, max_tokens=500).strip()
            except Exception as exc:  # pragma: no cover - network path
                return f"{fact_block}\n\n(Note: LLM composition unavailable: {exc})"

        # Local mode: deterministic, grounded answer.
        return fact_block or "I don't have enough data to answer that yet."

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _detect_category(q: str) -> str | None:
        for word in re.findall(r"[a-z&]+", q):
            if word in _CATEGORY_LOOKUP:
                return _CATEGORY_LOOKUP[word]
            if word in _CATEGORY_ALIASES:
                return _CATEGORY_ALIASES[word]
        return None
