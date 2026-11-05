"""Pydantic data models shared across the app."""
from __future__ import annotations

from datetime import date as date_type
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Decision = Literal["auto_approve", "flag", "reject"]

CATEGORIES = [
    "Travel",
    "Meals & Entertainment",
    "Lodging",
    "Software & Subscriptions",
    "Office Supplies",
    "Transportation",
    "Other",
]


class LineItem(BaseModel):
    description: str = ""
    amount: float = 0.0


class ExpenseExtraction(BaseModel):
    """Structured data extracted from a receipt."""

    vendor: str = "Unknown"
    date: Optional[str] = None  # ISO yyyy-mm-dd
    currency: str = "USD"
    total: float = 0.0
    tax: float = 0.0
    category: str = "Other"
    line_items: List[LineItem] = Field(default_factory=list)
    raw_text: str = ""


class PolicyCitation(BaseModel):
    clause: str
    source: str = "expense_policy.md"
    score: float = 0.0


class PolicyDecision(BaseModel):
    decision: Decision
    reason: str
    citations: List[PolicyCitation] = Field(default_factory=list)
    confidence: float = 0.0


class ExpenseRecord(BaseModel):
    id: Optional[int] = None
    vendor: str
    date: Optional[str] = None
    currency: str = "USD"
    total: float = 0.0
    tax: float = 0.0
    category: str = "Other"
    line_items: List[LineItem] = Field(default_factory=list)
    decision: Decision = "flag"
    decision_reason: str = ""
    citations: List[PolicyCitation] = Field(default_factory=list)
    created_at: Optional[str] = None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    used_tools: List[str] = Field(default_factory=list)
