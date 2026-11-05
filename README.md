# 🧾 Expense Copilot

An AI-powered expense assistant for corporate spend. Drop in a receipt and
Expense Copilot:

1. **Extracts** structured data (vendor, date, total, tax, category, line items).
2. **Decides** auto-approve / flag / reject against a company expense policy,
   using **Retrieval-Augmented Generation (RAG)** to cite the exact clauses it relied on.
3. **Answers questions** about your spend and policy through a tool-using chat agent
   ("How much did I spend on meals?", "Which expenses are flagged?",
   "What's the policy on hotels?").

Built for the AI Engineer vibe-coding challenge. The problem lives in the
corporate-spend domain, and the app is deliberately scoped to one clean workflow:
**receipt → structured data → policy decision → conversational analytics.**

---

## Why this problem

Expense review is slow, manual, and inconsistent. A human has to read each
receipt, remember the policy, and make a judgment call. That is exactly the kind
of narrow, high-volume, rules-plus-language task where AI shines: structured
extraction, semantic retrieval over policy text, and natural-language Q&A.

---

## Architecture

```
                    ┌──────────────────────────────┐
  Receipt           │        Streamlit UI           │   Chat / Analytics
 (img/pdf/text) ──▶ │  Submit · Expenses · Chat     │ ◀── questions
                    └───────────────┬───────────────┘
                                    │ HTTP (REST)
                    ┌───────────────▼───────────────┐
                    │          FastAPI backend       │
                    │                                │
    extraction.py ──┤  1. extract structured fields  │
       policy.py  ──┤  2. RAG policy decision + cites │──▶ ChromaDB (vectors)
        agent.py  ──┤  3. tool-using chat agent      │──▶ SQLite (expenses)
                    └───────────────┬────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   LLM provider (pluggable)      │
                    │   • local  (free, offline)      │
                    │   • bedrock (Claude + Titan)    │
                    └────────────────────────────────┘
```

**Pluggable LLM provider** is the central design decision. The entire app
depends on one interface (`app/llm/base.py`) with two implementations:

| Provider  | Embeddings | Vision | Generation | Cost / keys |
|-----------|------------|--------|------------|-------------|
| `local` (default) | `sentence-transformers` (offline) | — (paste/upload text) | deterministic, rule-based | **free, no keys** |
| `bedrock` | Amazon Titan | Anthropic Claude (receipt OCR) | Claude | AWS account |

This means **anyone can run and demo the app with zero credentials**, while the
same codebase upgrades to real cloud AI by flipping one env var.

---

## Quick start (free local mode — no keys)

Requires Python 3.10+.

```powershell
# 1. Create a virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. (optional) copy the env template; local mode is the default
copy .env.example .env              # macOS/Linux: cp .env.example .env

# 3. Seed sample policy + receipts (downloads the embedding model once)
python -m scripts.seed

# 4. Start the backend  (terminal 1)
uvicorn app.main:app --reload --port 8000

# 5. Start the UI       (terminal 2)
streamlit run frontend/streamlit_app.py
```

Open the Streamlit URL (usually http://localhost:8501). The FastAPI docs are at
http://localhost:8000/docs.

> The first `python -m scripts.seed` downloads the `all-MiniLM-L6-v2` embedding
> model (~90 MB) once, then runs fully offline.

---

## Using Amazon Bedrock (real LLM + vision)

1. Enable model access in the Bedrock console for Anthropic Claude and Amazon
   Titan Embeddings in your region.
2. Configure AWS credentials the usual way (`aws configure`, env vars, or IAM role).
3. In `.env`:

   ```env
   LLM_PROVIDER=bedrock
   AWS_REGION=us-east-1
   BEDROCK_TEXT_MODEL=anthropic.claude-3-5-sonnet-20240620-v1:0
   BEDROCK_EMBED_MODEL=amazon.titan-embed-text-v2:0
   ```

4. Re-seed to rebuild the vector index with Titan embeddings:
   `python -m scripts.reindex` *(or `python -m scripts.seed`)*, then start the app.

In Bedrock mode you can upload **receipt images** and Claude transcribes them;
the chat agent composes natural-language answers grounded in the retrieved data.

---

## How the pieces work

- **Extraction** (`app/extraction.py`) — resolves raw text from image/PDF/text,
  then parses vendor, date, currency, total (subtotal-aware), tax, category, and
  line items with robust regex + heuristics. Vision OCR is delegated to the
  provider so Bedrock handles real receipt photos.
- **Policy RAG** (`app/policy.py`) — the policy markdown is chunked by clause,
  embedded, and stored in ChromaDB. For every expense we retrieve the most
  relevant clauses as **citations**. The decision itself runs on a **structured
  ruleset** (per-category caps, global approval threshold) so outcomes are
  deterministic and auditable — RAG explains, rules decide.
- **Chat agent** (`app/agent.py`) — routes a question to one or more tools
  (SQL analytics + policy retrieval), gathers real facts, and composes a grounded
  answer. In Bedrock mode the LLM writes the prose; locally it's deterministic.
- **Store** (`app/store.py`) — SQLite persistence + analytics helpers.
- **API** (`app/main.py`) — FastAPI endpoints (`/receipts`, `/receipts/text`,
  `/expenses`, `/analytics`, `/chat`, `/policy/reindex`, `/health`).

---

## Project layout

```
app/
  __init__.py          package init (disables chroma telemetry)
  config.py            settings & provider selection
  models.py            pydantic data models
  extraction.py        receipt -> structured fields
  policy.py            RAG index + structured policy decision
  store.py             SQLite persistence + analytics
  agent.py             tool-using chat agent
  main.py              FastAPI app
  llm/
    base.py            LLMProvider interface
    local_provider.py  free offline provider (sentence-transformers)
    bedrock_provider.py Amazon Bedrock (Claude + Titan)
    factory.py         provider selection
frontend/
  streamlit_app.py     3-tab UI (Submit / Expenses / Chat)
data/
  expense_policy.md    company policy (the RAG knowledge base)
  sample_receipts/     example receipts for seeding
scripts/
  seed.py              load sample data
  reindex.py           rebuild the policy vector index
requirements.txt
WRITEUP.md             1-page write-up
```

---

## Try these in the Chat tab

- *How much did I spend on meals?*
- *Which expenses are flagged?*
- *What's the policy on hotels?*
- *Give me a category breakdown.*
- *What's my total spend?*

---

## Notes & trade-offs

- **Local generation is deterministic, not generative.** The free provider gives
  real embeddings (so RAG retrieval is genuine) and rule-based, explainable
  answers for the app's narrow tasks. Switch to Bedrock for free-form LLM prose.
- **Decisions are rule-based on purpose.** In finance you want auditable,
  reproducible outcomes, not a black box. RAG supplies the citations; a
  structured ruleset makes the call.
- Local image OCR is intentionally omitted (no heavy OCR dependency); paste text
  or use Bedrock vision for photos.
```
