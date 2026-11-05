"""SQLite-backed storage for expenses and their policy decisions."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

from .config import get_settings
from .models import ExpenseRecord, LineItem, PolicyCitation

_SCHEMA = """
CREATE TABLE IF NOT EXISTS expenses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    vendor          TEXT NOT NULL,
    date            TEXT,
    currency        TEXT DEFAULT 'USD',
    total           REAL DEFAULT 0,
    tax             REAL DEFAULT 0,
    category        TEXT DEFAULT 'Other',
    line_items      TEXT DEFAULT '[]',
    decision        TEXT DEFAULT 'flag',
    decision_reason TEXT DEFAULT '',
    citations       TEXT DEFAULT '[]',
    created_at      TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    conn = sqlite3.connect(str(settings.db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(_SCHEMA)
        conn.commit()


def _row_to_record(row: sqlite3.Row) -> ExpenseRecord:
    return ExpenseRecord(
        id=row["id"],
        vendor=row["vendor"],
        date=row["date"],
        currency=row["currency"],
        total=row["total"],
        tax=row["tax"],
        category=row["category"],
        line_items=[LineItem(**li) for li in json.loads(row["line_items"] or "[]")],
        decision=row["decision"],
        decision_reason=row["decision_reason"],
        citations=[PolicyCitation(**c) for c in json.loads(row["citations"] or "[]")],
        created_at=row["created_at"],
    )


def add_expense(record: ExpenseRecord) -> ExpenseRecord:
    init_db()
    created_at = record.created_at or datetime.utcnow().isoformat(timespec="seconds")
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO expenses
                (vendor, date, currency, total, tax, category, line_items,
                 decision, decision_reason, citations, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record.vendor,
                record.date,
                record.currency,
                record.total,
                record.tax,
                record.category,
                json.dumps([li.model_dump() for li in record.line_items]),
                record.decision,
                record.decision_reason,
                json.dumps([c.model_dump() for c in record.citations]),
                created_at,
            ),
        )
        conn.commit()
        record.id = cur.lastrowid
        record.created_at = created_at
    return record


def list_expenses() -> List[ExpenseRecord]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM expenses ORDER BY id DESC"
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def get_expense(expense_id: int) -> Optional[ExpenseRecord]:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM expenses WHERE id = ?", (expense_id,)
        ).fetchone()
    return _row_to_record(row) if row else None


def delete_expense(expense_id: int) -> bool:
    init_db()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
        conn.commit()
    return cur.rowcount > 0


def clear_all() -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM expenses")
        # Reset the autoincrement counter so IDs start at 1 again.
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'expenses'")
        conn.commit()


# ---- analytics helpers (used by the chat agent) -------------------------
def totals_by_category() -> Dict[str, float]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT category, SUM(total) AS s FROM expenses GROUP BY category"
        ).fetchall()
    return {r["category"]: round(r["s"] or 0.0, 2) for r in rows}


def totals_by_decision() -> Dict[str, int]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT decision, COUNT(*) AS c FROM expenses GROUP BY decision"
        ).fetchall()
    return {r["decision"]: r["c"] for r in rows}


def grand_total() -> float:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT SUM(total) AS s FROM expenses").fetchone()
    return round(row["s"] or 0.0, 2)
