from __future__ import annotations

import pytest

from homebrew_rag.config import Settings, get_settings
from homebrew_rag.generation import Answer
from homebrew_rag.store import SearchHit


@pytest.fixture(autouse=True)
def clean_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    return Settings(embed_dim=4, top_k=3, claude_model="claude-opus-5")


class FakeEmbedder:
    """Deterministic stand-in: no model download, no torch, no network."""

    dim = 4

    def __init__(self) -> None:
        self.documents_seen: list[str] = []
        self.queries_seen: list[str] = []

    def _vector(self, text: str) -> list[float]:
        return [float(len(text) % 7), float(text.count("a")), float(text.count("e")), 1.0]

    def embed_documents(self, texts):
        self.documents_seen.extend(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        self.queries_seen.append(text)
        return self._vector(text)


class FakeStore:
    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self.hits = hits or []
        self.searches: list[dict] = []
        self.upserted: list[tuple] = []
        self.deleted_sources: list[tuple[str, str]] = []
        self.deleted_tags: list[str] = []
        self.collection = "test"
        self.ensured = False

    def ensure_collection(self):
        self.ensured = True

    def delete_source(self, source, source_tag=""):
        self.deleted_sources.append((source, source_tag))

    def delete_tag(self, source_tag):
        self.deleted_tags.append(source_tag)

    def upsert_chunks(self, chunks, vectors, source_tag=""):
        self.upserted.append((list(chunks), list(vectors), source_tag))
        return len(chunks)

    def search(self, vector, top_k=5, source_tag=None, score_threshold=None):
        self.searches.append({"vector": vector, "top_k": top_k, "source_tag": source_tag})
        return self.hits[:top_k]

    def count(self, source_tag=None):
        return len(self.hits)

    def source_tags(self):
        return sorted({h.source_tag for h in self.hits if h.source_tag})

    def healthy(self):
        return True


class FakeGenerator:
    def __init__(self, text: str = "A grounded answer. [source: a.md]") -> None:
        self.text = text
        self.calls: list[tuple[str, list[SearchHit]]] = []

    def generate(self, question, hits):
        self.calls.append((question, hits))
        return Answer(
            text=self.text,
            model="claude-opus-5",
            stop_reason="end_turn",
            input_tokens=100,
            output_tokens=20,
        )


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def fake_generator() -> FakeGenerator:
    return FakeGenerator()


@pytest.fixture
def hit_factory():
    def _make(
        source: str, score: float = 0.9, text: str = "chunk text", tag: str = ""
    ) -> SearchHit:
        return SearchHit(
            text=text, source=source, section="## Heading", score=score, source_tag=tag
        )

    return _make
