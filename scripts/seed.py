"""Seed the database with the sample receipts.

Runs the full pipeline (extract -> policy decision -> store) on every file in
data/sample_receipts so the app has data to show immediately.

Usage:
    python -m scripts.seed
"""
from __future__ import annotations

from pathlib import Path

from app import store
from app.config import get_settings
from app.extraction import extract_fields
from app.llm.factory import get_provider
from app.models import ExpenseRecord
from app.policy import PolicyEngine


def main() -> None:
    settings = get_settings()
    provider = get_provider()
    policy = PolicyEngine(provider)

    store.init_db()
    print("Indexing policy...")
    chunks = policy.index_policy(force=True)
    print(f"  indexed {chunks} policy clauses")

    sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample_receipts"
    files = sorted(sample_dir.glob("*.txt"))
    if not files:
        print(f"No sample receipts found in {sample_dir}")
        return

    print(f"Clearing existing expenses and seeding {len(files)} receipts...")
    store.clear_all()

    for f in files:
        text = f.read_text(encoding="utf-8")
        extraction = extract_fields(text)
        decision = policy.decide(extraction)
        record = ExpenseRecord(
            vendor=extraction.vendor,
            date=extraction.date,
            currency=extraction.currency,
            total=extraction.total,
            tax=extraction.tax,
            category=extraction.category,
            line_items=extraction.line_items,
            decision=decision.decision,
            decision_reason=decision.reason,
            citations=decision.citations,
        )
        saved = store.add_expense(record)
        print(
            f"  #{saved.id} {saved.vendor[:28]:<28} {saved.currency} "
            f"{saved.total:>8,.2f}  [{saved.category}]  -> {saved.decision.upper()}"
        )

    print("\nDone. Start the backend and Streamlit UI to explore.")


if __name__ == "__main__":
    main()
