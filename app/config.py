"""Central configuration, loaded from environment / .env file."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Silence ChromaDB's anonymized telemetry (it emits noisy, harmless stderr
# warnings on some versions). Set before chromadb is imported anywhere.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    """Runtime settings resolved from environment variables."""

    def __init__(self) -> None:
        self.llm_provider: str = os.getenv("LLM_PROVIDER", "local").lower().strip()

        # Bedrock
        self.aws_region: str = os.getenv("AWS_REGION", "us-east-1")
        self.bedrock_text_model: str = os.getenv(
            "BEDROCK_TEXT_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0"
        )
        self.bedrock_embed_model: str = os.getenv(
            "BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0"
        )

        # Paths
        data_dir = os.getenv("DATA_DIR", str(BASE_DIR / "data"))
        self.data_dir: Path = Path(data_dir)
        self.db_path: Path = self.data_dir / "expenses.db"
        self.chroma_dir: Path = self.data_dir / "chroma"
        self.uploads_dir: Path = self.data_dir / "uploads"
        self.policy_path: Path = self.data_dir / "expense_policy.md"

        self.backend_url: str = os.getenv("BACKEND_URL", "http://localhost:8000")

        # Ensure directories exist
        for d in (self.data_dir, self.chroma_dir, self.uploads_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def use_bedrock(self) -> bool:
        return self.llm_provider == "bedrock"


@lru_cache
def get_settings() -> Settings:
    return Settings()
