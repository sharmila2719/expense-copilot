"""FastAPI backend for Expense Copilot.

Endpoints:
    GET  /health                 -> service + provider status
    POST /receipts               -> upload a receipt (multipart) -> extract + decide + save
    POST /receipts/text          -> submit raw receipt text (json) -> extract + decide + save
    GET  /expenses               -> list all stored expenses
    GET  /expenses/{id}          -> one expense
    DELETE /expenses/{id}        -> delete an expense
    GET  /analytics              -> totals by category / decision + grand total
    POST /chat                   -> ask the expense agent
    POST /policy/reindex         -> rebuild the policy vector index
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import store
from .agent import ExpenseAgent
from .config import get_settings
from .extraction import extract, extract_fields
from .llm.factory import get_provider
from .models import ChatRequest, ChatResponse, ExpenseRecord
from .policy import PolicyEngine

app = FastAPI(title="Expense Copilot", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def _policy_engine() -> PolicyEngine:
    return PolicyEngine(get_provider())


@lru_cache
def _agent() -> ExpenseAgent:
    return ExpenseAgent(get_provider(), _policy_engine())


class TextReceipt(BaseModel):
    text: str


@app.on_event("startup")
def _startup() -> None:
    store.init_db()
    # Index the policy if present (safe no-op if already indexed).
    try:
        _policy_engine().index_policy()
    except Exception:
        # Don't block startup on indexing; endpoints will retry lazily.
        pass


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "provider": get_provider().name,
        "db": str(settings.db_path),
    }


def _process(extraction) -> ExpenseRecord:
    """Shared flow: run policy decision + persist."""
    decision = _policy_engine().decide(extraction)
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
    return store.add_expense(record)


@app.post("/receipts", response_model=ExpenseRecord)
async def upload_receipt(file: UploadFile = File(...)) -> ExpenseRecord:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    extraction = extract(
        get_provider(),
        content=content,
        filename=file.filename or "receipt",
        content_type=file.content_type,
    )
    return _process(extraction)


@app.post("/receipts/text", response_model=ExpenseRecord)
def upload_receipt_text(payload: TextReceipt) -> ExpenseRecord:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    extraction = extract_fields(payload.text)
    return _process(extraction)


@app.get("/expenses", response_model=list[ExpenseRecord])
def list_expenses() -> list[ExpenseRecord]:
    return store.list_expenses()


@app.get("/expenses/{expense_id}", response_model=ExpenseRecord)
def get_expense(expense_id: int) -> ExpenseRecord:
    record = store.get_expense(expense_id)
    if not record:
        raise HTTPException(status_code=404, detail="Expense not found")
    return record


@app.delete("/expenses/{expense_id}")
def delete_expense(expense_id: int) -> dict:
    ok = store.delete_expense(expense_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Expense not found")
    return {"deleted": expense_id}


@app.get("/analytics")
def analytics() -> dict:
    return {
        "grand_total": store.grand_total(),
        "by_category": store.totals_by_category(),
        "by_decision": store.totals_by_decision(),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    return _agent().answer(req.message)


@app.post("/policy/reindex")
def reindex() -> dict:
    count = _policy_engine().index_policy(force=True)
    return {"indexed_chunks": count}
