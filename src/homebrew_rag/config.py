"""Settings, resolved once from the environment (and a local .env if present).

Every tunable in the pipeline lives here so that chunking, embedding, retrieval
and generation can be re-pointed without editing code — which matters because
retrieval quality work is mostly a loop of "change one knob, re-run the eval".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

# bge models are trained asymmetrically: documents are embedded plain, queries get
# an instruction prefix. Omitting it does not error — it just quietly degrades
# retrieval. If you swap embedding models, check the model card and update this.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- Vector store ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection: str = "documents"

    # --- Embeddings ---
    embed_model: str = "BAAI/bge-base-en-v1.5"
    embed_dim: int = 768
    embed_batch_size: int = 64
    query_prefix: str = BGE_QUERY_PREFIX

    # --- Chunking ---
    chunk_size: int = 500
    chunk_overlap: int = 75

    # --- Retrieval ---
    top_k: int = 5
    score_threshold: float | None = None

    # --- Generation ---
    anthropic_api_key: str | None = None
    claude_model: str = "claude-opus-5"
    max_tokens: int = 16000
    enable_refusal_fallback: bool = True

    # --- API ---
    api_key: str | None = None
    log_level: str = "INFO"
    log_answers: bool = False

    extra: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        threshold = os.environ.get("RAG_SCORE_THRESHOLD")
        return cls(
            qdrant_url=_env_str("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
            collection=_env_str("RAG_COLLECTION", "documents"),
            embed_model=_env_str("RAG_EMBED_MODEL", "BAAI/bge-base-en-v1.5"),
            embed_dim=_env_int("RAG_EMBED_DIM", 768),
            embed_batch_size=_env_int("RAG_EMBED_BATCH_SIZE", 64),
            query_prefix=_env_str("RAG_QUERY_PREFIX", BGE_QUERY_PREFIX),
            chunk_size=_env_int("RAG_CHUNK_SIZE", 500),
            chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 75),
            top_k=_env_int("RAG_TOP_K", 5),
            score_threshold=float(threshold) if threshold else None,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
            claude_model=_env_str("RAG_CLAUDE_MODEL", "claude-opus-5"),
            max_tokens=_env_int("RAG_MAX_TOKENS", 16000),
            enable_refusal_fallback=_env_bool("RAG_ENABLE_REFUSAL_FALLBACK", True),
            api_key=os.environ.get("RAG_API_KEY") or None,
            log_level=_env_str("RAG_LOG_LEVEL", "INFO").upper(),
            log_answers=_env_bool("RAG_LOG_ANSWERS", False),
        )

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}) — otherwise chunking never advances."
            )
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Cached; call `get_settings.cache_clear()` in tests."""
    return Settings.from_env()
