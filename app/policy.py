"""RAG-based expense policy engine.

Pipeline:
1. Load the company expense policy (markdown) and split it into clause-sized
   chunks (one per heading/bullet group).
2. Embed each chunk and store it in a persistent ChromaDB collection.
3. For a given expense, build a query, retrieve the most relevant clauses,
   and decide auto_approve / flag / reject — citing the clauses used.

The embeddings come from whichever LLM provider is active (Titan on Bedrock,
sentence-transformers locally), so retrieval quality follows the provider.

The decision itself is rule-based on top of the retrieved clauses: we parse
monetary limits and "requires receipt/approval" signals out of the matched
policy text. This keeps decisions explainable and auditable — important in a
finance/compliance context — rather than a black-box yes/no.
"""
from __future__ import annotations

import re
from typing import List, Optional

import chromadb

from .config import get_settings
from .llm.base import LLMProvider
from .models import ExpenseExtraction, PolicyCitation, PolicyDecision

COLLECTION = "expense_policy"


class EmbeddingFn:
    """Adapts an LLMProvider to Chroma's embedding-function protocol."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def name(self) -> str:  # Chroma >=0.5 expects a name()
        return f"expense-copilot-{self._provider.name}"

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        return self._provider.embed(input)


class PolicyEngine:
    def __init__(self, provider: LLMProvider) -> None:
        self._settings = get_settings()
        self._provider = provider
        self._client = chromadb.PersistentClient(
            path=str(self._settings.chroma_dir),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._embed_fn = EmbeddingFn(provider)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION, embedding_function=self._embed_fn
        )

    # -- indexing ---------------------------------------------------------
    def index_policy(self, policy_text: Optional[str] = None, force: bool = False) -> int:
        """Chunk + embed the policy. Returns number of chunks indexed."""
        if policy_text is None:
            policy_text = self._load_policy_file()

        if self._collection.count() > 0 and not force:
            return self._collection.count()

        if force and self._collection.count() > 0:
            # Recreate the collection to clear it.
            self._client.delete_collection(COLLECTION)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION, embedding_function=self._embed_fn
            )

        chunks = self._chunk_policy(policy_text)
        ids = [f"clause-{i}" for i in range(len(chunks))]
        self._collection.add(ids=ids, documents=chunks)
        return len(chunks)

    def _load_policy_file(self) -> str:
        path = self._settings.policy_path
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    @staticmethod
    def _chunk_policy(text: str) -> List[str]:
        """Split policy into clause chunks on markdown headings."""
        if not text.strip():
            return []
        # Split on ## headings; keep the heading with its body.
        parts = re.split(r"\n(?=#{1,6}\s)", text)
        chunks: List[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Further split long sections by blank-line paragraphs.
            if len(part) > 600:
                for para in re.split(r"\n\s*\n", part):
                    para = para.strip()
                    if para:
                        chunks.append(para)
            else:
                chunks.append(part)
        return chunks

    # -- retrieval --------------------------------------------------------
    def retrieve(self, query: str, k: int = 4) -> List[PolicyCitation]:
        if self._collection.count() == 0:
            self.index_policy()
        if self._collection.count() == 0:
            return []
        n = min(k, self._collection.count())
        res = self._collection.query(query_texts=[query], n_results=n)
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0] or [0.0] * len(docs)
        citations: List[PolicyCitation] = []
        for doc, dist in zip(docs, dists):
            # Convert distance to a rough similarity score in [0,1].
            score = max(0.0, 1.0 - float(dist))
            citations.append(PolicyCitation(clause=doc.strip(), score=round(score, 3)))
        return citations

    # -- decision ---------------------------------------------------------
    #
    # Design: the decision is driven by a STRUCTURED ruleset that mirrors the
    # policy document (per-category spend limits, prohibited categories, global
    # thresholds). RAG retrieval runs in parallel purely to attach the most
    # relevant policy clauses as human-readable *citations*. This split is
    # deliberate — retrieval is great for "show me the relevant rule", but a
    # reimbursement decision must be deterministic and auditable, not dependent
    # on which chunks happened to rank highest.

    # Per-category spend caps (USD). Mirrors data/expense_policy.md.
    CATEGORY_LIMITS = {
        "Meals & Entertainment": 150.0,  # client business meal cap
        "Travel": 650.0,
        "Lodging": 250.0,
        "Transportation": 100.0,
        "Software & Subscriptions": 500.0,
        "Office Supplies": 200.0,
    }
    PROHIBITED_CATEGORIES: set = set()  # prohibition is item-level in this policy
    RECEIPT_REQUIRED_OVER = 25.0
    DIRECTOR_APPROVAL_OVER = 1000.0

    def decide(self, expense: ExpenseExtraction) -> PolicyDecision:
        query = (
            f"Expense of {expense.total} {expense.currency} at {expense.vendor} "
            f"in category {expense.category}. What spend limit, approval, or "
            f"receipt requirement applies?"
        )
        citations = self.retrieve(query, k=3)
        return self._reason(expense, citations)

    def _reason(
        self, expense: ExpenseExtraction, citations: List[PolicyCitation]
    ) -> PolicyDecision:
        total = expense.total
        cur = expense.currency
        cat = expense.category
        confidence = (
            round(sum(c.score for c in citations) / len(citations), 3)
            if citations else 0.0
        )

        def result(decision: str, reason: str) -> PolicyDecision:
            return PolicyDecision(
                decision=decision, reason=reason,
                citations=citations, confidence=confidence,
            )

        # 1. Invalid / unreadable total -> needs human review.
        if total <= 0:
            return result("flag", "Could not read a valid total amount; manual review needed.")

        # 2. Explicitly prohibited categories -> reject.
        if cat in self.PROHIBITED_CATEGORIES:
            return result("reject", f"{cat} is a non-reimbursable category under policy.")

        # 3. Global director-approval threshold.
        if total > self.DIRECTOR_APPROVAL_OVER:
            return result(
                "flag",
                f"{cur} {total:,.2f} exceeds the {cur} "
                f"{self.DIRECTOR_APPROVAL_OVER:,.0f} director-approval threshold.",
            )

        # 4. Per-category spend cap.
        limit = self.CATEGORY_LIMITS.get(cat)
        if limit is not None and total > limit:
            if total > limit * 1.5:
                return result(
                    "reject",
                    f"{cur} {total:,.2f} is more than 50% over the {cat} limit of "
                    f"{cur} {limit:,.0f}; rejected pending justification.",
                )
            return result(
                "flag",
                f"{cur} {total:,.2f} exceeds the {cat} limit of {cur} "
                f"{limit:,.0f}; requires manager approval.",
            )

        # 5. Receipt-required threshold (receipt text present here, so this is
        #    informational; a production integration would verify attachments).
        if total >= self.RECEIPT_REQUIRED_OVER and not expense.raw_text.strip():
            return result(
                "flag",
                f"Expenses at/above {cur} {self.RECEIPT_REQUIRED_OVER:,.0f} require "
                "an itemized receipt; none detected.",
            )

        # 6. Within all limits.
        return result(
            "auto_approve",
            f"{cur} {total:,.2f} for {cat} is within the applicable policy limits.",
        )
