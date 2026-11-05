"""Expense Copilot — Streamlit UI.

Talks to the FastAPI backend over HTTP. Three tabs:
  1. Submit    — upload a receipt file OR paste receipt text; see extraction +
                 the policy decision with cited clauses.
  2. Expenses  — table of all expenses + analytics charts.
  3. Chat      — ask natural-language questions about spend and policy.
"""
from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Expense Copilot", page_icon="🧾", layout="wide")

DECISION_STYLE = {
    "auto_approve": ("✅ Auto-approved", "#1b7f3b"),
    "flag": ("⚠️ Flagged for review", "#b8860b"),
    "reject": ("⛔ Rejected", "#b22222"),
}

SAMPLE_RECEIPT = """BLUE BOTTLE COFFEE
123 Market Street, San Francisco, CA
Date: 2026-03-14
------------------------------
Cappuccino            5.50
Avocado Toast        12.00
Sparkling Water       3.50
------------------------------
Subtotal             21.00
Tax                   1.85
Total                22.85
Card **** 4242
"""


def _get(path: str):
    try:
        r = requests.get(f"{BACKEND}{path}", timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Backend error on GET {path}: {exc}")
        return None


def _post(path: str, **kwargs):
    try:
        r = requests.post(f"{BACKEND}{path}", timeout=120, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Backend error on POST {path}: {exc}")
        return None


def _render_decision(record: dict, nested: bool = False) -> None:
    """Render a decision card + its policy citations.

    When ``nested`` is True the caller is already inside a st.expander, so we
    render the citations inline (Streamlit does not allow nested expanders).
    """
    label, color = DECISION_STYLE.get(record["decision"], (record["decision"], "#555"))
    st.markdown(
        f"<div style='padding:10px;border-radius:8px;background:{color}22;"
        f"border:1px solid {color};'><b style='color:{color}'>{label}</b><br>"
        f"{record.get('decision_reason','')}</div>",
        unsafe_allow_html=True,
    )
    cites = record.get("citations", [])
    if not cites:
        return

    def _render_cites() -> None:
        for c in cites:
            st.markdown(f"**(score {c.get('score', 0):.2f})** {c['clause']}")

    if nested:
        st.caption(f"📑 Policy clauses cited ({len(cites)})")
        _render_cites()
    else:
        with st.expander(f"📑 Policy clauses cited ({len(cites)})"):
            _render_cites()


# ---- header -------------------------------------------------------------
health = _get("/health")
provider = health["provider"] if health else "unknown"

st.title("🧾 Expense Copilot")
st.caption(
    f"AI receipt extraction · RAG policy checks · chat with your spend  —  "
    f"LLM provider: **{provider}**"
)

tab_submit, tab_expenses, tab_chat = st.tabs(["➕ Submit", "📊 Expenses", "💬 Chat"])

# ---- Submit tab ---------------------------------------------------------
with tab_submit:
    st.subheader("Submit a receipt")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Option A — upload a file** (image / PDF / text)")
        uploaded = st.file_uploader(
            "Drop a receipt", type=["png", "jpg", "jpeg", "webp", "pdf", "txt", "md"]
        )
        if st.button("Process file", disabled=uploaded is None):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            with st.spinner("Extracting and applying policy..."):
                rec = _post("/receipts", files=files)
            if rec:
                st.success(f"Saved expense #{rec['id']}")
                st.json({k: rec[k] for k in ("vendor", "date", "currency", "total", "tax", "category")})
                _render_decision(rec)

    with col2:
        st.markdown("**Option B — paste receipt text**")
        text = st.text_area("Receipt text", value=SAMPLE_RECEIPT, height=240)
        if st.button("Process text"):
            with st.spinner("Extracting and applying policy..."):
                rec = _post("/receipts/text", json={"text": text})
            if rec:
                st.success(f"Saved expense #{rec['id']}")
                st.json({k: rec[k] for k in ("vendor", "date", "currency", "total", "tax", "category")})
                if rec.get("line_items"):
                    st.markdown("**Line items**")
                    st.table(rec["line_items"])
                _render_decision(rec)

# ---- Expenses tab -------------------------------------------------------
with tab_expenses:
    st.subheader("All expenses")
    analytics = _get("/analytics")
    if analytics:
        m1, m2, m3 = st.columns(3)
        m1.metric("Grand total", f"{analytics['grand_total']:,.2f}")
        by_dec = analytics["by_decision"]
        m2.metric("Flagged", by_dec.get("flag", 0) + by_dec.get("reject", 0))
        m3.metric("Auto-approved", by_dec.get("auto_approve", 0))

        if analytics["by_category"]:
            st.markdown("**Spend by category**")
            st.bar_chart(analytics["by_category"])

    expenses = _get("/expenses")
    if expenses:
        rows = [
            {
                "ID": e["id"],
                "Vendor": e["vendor"],
                "Date": e["date"],
                "Category": e["category"],
                "Total": f"{e['currency']} {e['total']:,.2f}",
                "Decision": e["decision"],
            }
            for e in expenses
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

        with st.expander("🔎 Inspect a decision"):
            ids = [e["id"] for e in expenses]
            if ids:
                sel = st.selectbox("Expense ID", ids)
                chosen = next((e for e in expenses if e["id"] == sel), None)
                if chosen:
                    _render_decision(chosen, nested=True)
    else:
        st.info("No expenses yet. Submit one in the Submit tab.")

# ---- Chat tab -----------------------------------------------------------
with tab_chat:
    st.subheader("Ask about your expenses and policy")
    st.caption(
        "Try: *How much did I spend on meals?* · *Which expenses are flagged?* · "
        "*What's the policy on hotels?* · *Give me a category breakdown.*"
    )

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)

    prompt = st.chat_input("Ask a question...")
    if prompt:
        st.session_state.chat.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = _post("/chat", json={"message": prompt})
            if resp:
                answer = resp["answer"]
                if resp.get("used_tools"):
                    answer += f"\n\n*tools: {', '.join(resp['used_tools'])}*"
                st.markdown(answer)
                st.session_state.chat.append(("assistant", answer))
