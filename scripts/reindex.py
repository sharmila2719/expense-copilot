"""Rebuild the policy vector index from data/expense_policy.md.

Useful after switching LLM providers (embeddings change) or editing the policy.

Usage:
    python -m scripts.reindex
"""
from __future__ import annotations

from app.llm.factory import get_provider
from app.policy import PolicyEngine


def main() -> None:
    engine = PolicyEngine(get_provider())
    count = engine.index_policy(force=True)
    print(f"Reindexed {count} policy clauses using provider '{get_provider().name}'.")


if __name__ == "__main__":
    main()
