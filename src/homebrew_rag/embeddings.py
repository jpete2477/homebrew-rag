"""Embedding backends.

Everything the rest of the pipeline knows about embeddings is the `Embedder`
protocol below. That is deliberate: swapping the local model for an API-based
one (Voyage, for instance, which is Anthropic's recommended embeddings partner)
means writing one new class here and changing nothing else.

The default is local — `sentence-transformers` running on CPU — so document
text never leaves the machine during ingestion or retrieval. The only network
call the system makes is the final generation request to Claude, which carries
just the top-k retrieved snippets.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Minimal surface the pipeline needs from an embedding backend."""

    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder:
    """`sentence-transformers` backend, loaded lazily on first use.

    Loading the model takes several seconds and a few hundred MB of RAM, so it
    happens once per process — never inside a loop — and not at import time, so
    that tests and `--help` stay fast.
    """

    def __init__(
        self,
        model_name: str,
        dim: int,
        query_prefix: str = "",
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on install extras
                raise ImportError(
                    "The local embedding backend needs sentence-transformers. "
                    'Install it with: pip install -e ".[local-embeddings]"'
                ) from exc
            logger.info("Loading embedding model %s (first call downloads it)", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            # Renamed in sentence-transformers 6.0; support both spellings.
            get_dim = (
                getattr(self._model, "get_embedding_dimension", None)
                or self._model.get_sentence_embedding_dimension
            )
            actual = get_dim()
            if actual != self.dim:
                raise ValueError(
                    f"{self.model_name} produces {actual}-dim vectors but RAG_EMBED_DIM "
                    f"is {self.dim}. Fix the setting, and re-create the Qdrant collection "
                    f"if it was built at the old dimension."
                )
        return self._model

    def warm(self) -> None:
        """Force the model load now rather than on the first user request."""
        self._load()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        vector = model.encode(
            self.query_prefix + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()


def build_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    return LocalEmbedder(
        model_name=settings.embed_model,
        dim=settings.embed_dim,
        query_prefix=settings.query_prefix,
        batch_size=settings.embed_batch_size,
    )
