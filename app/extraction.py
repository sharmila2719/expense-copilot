"""Receipt extraction pipeline.

Turns a raw receipt (image, PDF, or pasted text) into a structured
ExpenseExtraction. Two stages:

1. Get text:
   - image  -> vision model (Bedrock) OR, in local mode, we rely on the user
     pasting/uploading text (no heavyweight OCR dependency required).
   - pdf    -> extracted with pypdf.
   - text   -> used directly.

2. Parse text -> structured fields with regex + heuristics. When the Bedrock
   provider is active we additionally ask the LLM to return structured JSON,
   which is more robust on noisy receipts; the regex parser is the deterministic
   baseline and the local-mode path.
"""
from __future__ import annotations

import io
import json
import re
from datetime import datetime
from typing import List, Optional

from .llm.base import LLMProvider
from .models import CATEGORIES, ExpenseExtraction, LineItem

# ---- currency detection -------------------------------------------------
CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "¥": "JPY"}
CURRENCY_CODES = {"USD", "EUR", "GBP", "INR", "JPY", "CAD", "AUD"}

# ---- category keyword map (used by both providers) ----------------------
CATEGORY_KEYWORDS = {
    "Travel": [
        "airline", "airlines", "flight", "airways", "airport", "boarding",
        "united", "delta", "ticket", "passenger", "fare",
    ],
    "Lodging": [
        "hotel", "hotels", "inn", "resort", "lodging", "marriott", "hilton",
        "motel", "room (", "night",
    ],
    "Meals & Entertainment": [
        "restaurant", "cafe", "café", "coffee", "bar", "grill", "diner",
        "bistro", "pizza", "kitchen", "starbucks", "food", "menu", "tasting",
        "wine", "dining", "eatery", "laundry",  # "French Laundry" is a restaurant
    ],
    "Transportation": [
        "uber", "lyft", "taxi", "cab", "rental", "hertz", "avis", "fuel",
        "gas", "parking", "mileage",
    ],
    "Software & Subscriptions": [
        "subscription", "saas", "software", "github", "aws", "adobe", "zoom",
        "slack", "notion", "openai", "license", "invoice", "annual",
    ],
    "Office Supplies": ["staples", "office", "supplies", "paper", "printer", "ink"],
}

AMOUNT_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})*(?:\.\d{2})|\d+\.\d{2})")
DATE_PATTERNS = [
    ("%Y-%m-%d", re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")),
    ("%m/%d/%Y", re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")),
    ("%m/%d/%y", re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2})\b")),
    ("%d %b %Y", re.compile(r"\b(\d{1,2} [A-Za-z]{3} \d{4})\b")),
    ("%b %d, %Y", re.compile(r"\b([A-Za-z]{3} \d{1,2}, \d{4})\b")),
]


def get_raw_text(
    provider: LLMProvider,
    *,
    content: bytes,
    filename: str,
    content_type: Optional[str],
) -> str:
    """Resolve raw receipt text from an uploaded file's bytes."""
    name = (filename or "").lower()
    ctype = (content_type or "").lower()

    # PDF
    if name.endswith(".pdf") or "pdf" in ctype:
        return _pdf_to_text(content)

    # Plain text
    if name.endswith((".txt", ".md")) or ctype.startswith("text/"):
        return content.decode("utf-8", errors="ignore")

    # Image -> try the provider's vision capability (Bedrock). If unavailable
    # (local mode), fall back to decoding as text so pasted/synthetic receipts
    # still work without an OCR engine.
    if name.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")) or ctype.startswith("image/"):
        media_type = ctype if ctype.startswith("image/") else _guess_media_type(name)
        try:
            return provider.extract_receipt_text(content, media_type)
        except NotImplementedError:
            return (
                "[Local mode: image OCR not available. Paste the receipt text "
                "or set LLM_PROVIDER=bedrock for vision extraction.]"
            )

    # Unknown: best-effort decode
    return content.decode("utf-8", errors="ignore")


def _pdf_to_text(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _guess_media_type(name: str) -> str:
    if name.endswith(".png"):
        return "image/png"
    if name.endswith(".webp"):
        return "image/webp"
    if name.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


def extract_fields(text: str) -> ExpenseExtraction:
    """Deterministic parse of receipt text into structured fields."""
    vendor = _guess_vendor(text)
    date = _guess_date(text)
    currency = _guess_currency(text)
    amounts = _find_amounts(text)
    total = _guess_total(text, amounts)
    tax = _guess_tax(text, amounts)
    category = _guess_category(text, vendor)
    line_items = _guess_line_items(text)

    return ExpenseExtraction(
        vendor=vendor,
        date=date,
        currency=currency,
        total=total,
        tax=tax,
        category=category,
        line_items=line_items,
        raw_text=text.strip(),
    )


def extract(
    provider: LLMProvider,
    *,
    content: bytes,
    filename: str,
    content_type: Optional[str],
) -> ExpenseExtraction:
    """Full pipeline: bytes -> raw text -> structured fields."""
    text = get_raw_text(
        provider, content=content, filename=filename, content_type=content_type
    )
    return extract_fields(text)


# ---- heuristics ---------------------------------------------------------
def _guess_vendor(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # Skip lines that are clearly amounts/dates/labels
        if AMOUNT_RE.search(s) and len(s) < 12:
            continue
        if re.match(r"^(receipt|invoice|date|order|tel|phone|www\.)", s, re.I):
            continue
        return s[:80]
    return "Unknown"


def _guess_date(text: str) -> Optional[str]:
    for fmt, pattern in DATE_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                return datetime.strptime(m.group(1), fmt).date().isoformat()
            except ValueError:
                continue
    return None


def _guess_currency(text: str) -> str:
    for sym, code in CURRENCY_SYMBOLS.items():
        if sym in text:
            return code
    for code in CURRENCY_CODES:
        if re.search(rf"\b{code}\b", text):
            return code
    return "USD"


def _find_amounts(text: str) -> List[float]:
    out: List[float] = []
    for m in AMOUNT_RE.finditer(text):
        try:
            out.append(float(m.group(1).replace(",", "")))
        except ValueError:
            continue
    return out


def _amount_near_label(
    text: str, labels: List[str], exclude: Optional[List[str]] = None
) -> Optional[float]:
    """Return the amount on the last line whose label matches (and is not excluded)."""
    exclude = exclude or []
    found: Optional[float] = None
    for line in text.splitlines():
        low = line.lower()
        if any(ex in low for ex in exclude):
            continue
        if any(lbl in low for lbl in labels):
            amts = _find_amounts(line)
            if amts:
                found = amts[-1]  # keep last match (totals appear near the end)
    return found


def _guess_total(text: str, amounts: List[float]) -> float:
    # Prefer an explicit grand-total style label, excluding "subtotal".
    labelled = _amount_near_label(
        text,
        ["total due", "grand total", "amount due", "balance due", "total"],
        exclude=["subtotal", "sub total", "sub-total"],
    )
    if labelled is not None:
        return labelled
    return max(amounts) if amounts else 0.0


def _guess_tax(text: str, amounts: List[float]) -> float:
    labelled = _amount_near_label(text, ["tax", "vat", "gst"])
    return labelled if labelled is not None else 0.0


def _guess_category(text: str, vendor: str) -> str:
    hay = f"{vendor} {text}".lower()
    best_cat, best_hits = "Other", 0
    for cat, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in hay)
        if hits > best_hits:
            best_cat, best_hits = cat, hits
    return best_cat if best_cat in CATEGORIES else "Other"


def _guess_line_items(text: str) -> List[LineItem]:
    items: List[LineItem] = []
    skip = ("total", "subtotal", "tax", "vat", "gst", "balance", "change", "cash", "card", "tip")
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if any(k in low for k in skip):
            continue
        amts = _find_amounts(s)
        if not amts:
            continue
        desc = AMOUNT_RE.sub("", s).strip(" -\t:$€£")
        if desc:
            items.append(LineItem(description=desc[:80], amount=amts[-1]))
    return items[:15]
