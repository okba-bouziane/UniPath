"""Central configuration, read from environment variables (.env)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT_DIR / "chroma_db"))
COLLECTION_NAME = "unipath"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 4
# Chunks with a relevance score below this are treated as "not found".
MIN_RELEVANCE = 0.3

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
}


def llm_model() -> str:
    return os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(LLM_PROVIDER, "")
