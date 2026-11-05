"""Abstract LLM provider interface.

The rest of the app depends only on this interface, never on a concrete
provider. That lets us swap between Amazon Bedrock (real LLM) and a free,
offline local provider without changing any business logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class LLMProvider(ABC):
    """Interface every provider must implement."""

    name: str = "base"

    @abstractmethod
    def generate(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        """Return a text completion for the given system + user prompt."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return an embedding vector for each input text."""

    def extract_receipt_text(self, image_bytes: bytes, media_type: str) -> str:
        """Return text content from a receipt image.

        Providers with vision capability (Bedrock/Claude) override this.
        Providers without vision raise NotImplementedError so callers can
        fall back to OCR-free / text-based paths.
        """
        raise NotImplementedError
