"""Expense Copilot application package."""
import os

# Disable ChromaDB anonymized telemetry before chromadb is imported anywhere.
# Avoids noisy, harmless stderr warnings on some chromadb versions.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")

__version__ = "0.1.0"
