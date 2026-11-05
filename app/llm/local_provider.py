"""Free, offline LLM provider.

Design goal: the application must run and demo end-to-end with ZERO cloud
credentials or API costs. This provider gives us:

* Real semantic embeddings via sentence-transformers (downloaded once, then
  fully offline). These power the RAG retrieval exactly like a cloud embedding
  model would.
* A deterministic, rule-based "generation" path for the structured tasks the
  app needs (receipt field extraction, policy reasoning, chat answers). It is
  not a general chatbot, but it produces reliable, explainable output for the
  narrow set of prompts this app issues.

Trade-off: the local provider is deterministic rather than generative, so its
answers are more rigid than Claude's. The upside is reproducibility, no key
management, and a demo that works anywhere. Swap LLM_PROVIDER=bedrock for the
full generative experience.
"""
from __future__ import annotations

from typing import List, Optional

from .base import LLMProvider


class LocalProvider(LLMProvider):
    name = "local"

    def __init__(self, embed_model_name: str = "all-MiniLM-L6-v2") -> None:
        self._embed_model_name = embed_model_name
        self._embed_model = None  # lazy-loaded

    # -- embeddings -------------------------------------------------------
    def _ensure_model(self):
        if self._embed_model is None:
            # Imported lazily so the module imports fast and only pulls the
            # heavy dependency when embeddings are actually needed.
            from sentence_transformers import SentenceTransformer

            self._embed_model = SentenceTransformer(self._embed_model_name)
        return self._embed_model

    def embed(self, texts: List[str]) -> List[List[float]]:
        model = self._ensure_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    # -- generation -------------------------------------------------------
    def generate(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        """Minimal deterministic responder.

        The app's generative needs are handled by dedicated rule-based modules
        (extraction.py, policy decision logic, chat agent). This method exists
        to satisfy the interface and provide a graceful, honest fallback for
        any free-form prompt.
        """
        return (
            "[local-provider] Generative free-form responses require the "
            "Bedrock provider. Structured tasks (extraction, policy checks, "
            "expense queries) are fully supported locally."
        )
