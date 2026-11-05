"""Provider factory: returns a cached LLMProvider based on settings."""
from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from .base import LLMProvider


@lru_cache
def get_provider() -> LLMProvider:
    settings = get_settings()
    if settings.use_bedrock:
        from .bedrock_provider import BedrockProvider

        return BedrockProvider(
            region=settings.aws_region,
            text_model=settings.bedrock_text_model,
            embed_model=settings.bedrock_embed_model,
        )

    from .local_provider import LocalProvider

    return LocalProvider()
