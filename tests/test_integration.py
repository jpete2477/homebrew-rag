"""End-to-end tests against a real Qdrant.

    docker compose up -d
    pytest -m integration

Generation is exercised only when ANTHROPIC_API_KEY is set, so the retrieval
half can be verified without spending anything.
"""

from __future__ import annotations

import importlib.util
import os
import uuid

import pytest

from homebrew_rag.chunking import chunk_text
from homebrew_rag.config import Settings
from homebrew_rag.ingest import ingest_directory
from homebrew_rag.pipeline import RagPipeline
from homebrew_rag.store import QdrantStore

pytestmark = pytest.mark.integration


@pytest.fixture
def settings() -> Settings:
    # A throwaway collection per run, so a failed test never poisons a real index.
    return Settings(
        collection=f"test_{uuid.uuid4().hex[:8]}",
        embed_dim=4,
        chunk_size=40,
        chunk_overlap=8,
    )


@pytest.fixture
def store(settings):
    store = QdrantStore.from_settings(settings)
    if not store.healthy():
        pytest.skip(f"Qdrant not reachable at {settings.qdrant_url}")
    store.ensure_collection()
    yield store
    store.client.delete_collection(settings.collection)


def test_upsert_search_and_filter_round_trip(store, fake_embedder, settings):
    chunks = chunk_text(
        "# Pricing\nThe AI Opportunity Audit is a fixed-fee engagement.",
        source="pricing.md",
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    vectors = fake_embedder.embed_documents([c.text for c in chunks])
    assert store.upsert_chunks(chunks, vectors, source_tag="services") == len(chunks)

    hits = store.search(vectors[0], top_k=3)
    assert hits and hits[0].source == "pricing.md"
    assert hits[0].source_tag == "services"

    assert store.search(vectors[0], top_k=3, source_tag="services")
    assert store.search(vectors[0], top_k=3, source_tag="nonexistent") == []
    assert store.count(source_tag="services") == len(chunks)
    assert store.source_tags() == ["services"]


def test_reingest_is_idempotent(tmp_path, store, fake_embedder, settings):
    (tmp_path / "doc.md").write_text("# A\n" + " ".join(f"w{i}" for i in range(200)))

    first = ingest_directory(
        tmp_path, source_tag="corpus", embedder=fake_embedder, store=store, settings=settings
    )
    assert store.count() == first.points

    second = ingest_directory(
        tmp_path, source_tag="corpus", embedder=fake_embedder, store=store, settings=settings
    )
    assert second.points == first.points
    assert store.count() == first.points  # overwritten, not duplicated


def test_shortened_document_drops_stale_chunks(tmp_path, store, fake_embedder, settings):
    path = tmp_path / "doc.md"
    path.write_text("# A\n" + " ".join(f"w{i}" for i in range(300)))
    ingest_directory(
        tmp_path, source_tag="corpus", embedder=fake_embedder, store=store, settings=settings
    )
    before = store.count()

    path.write_text("# A\nshort now")
    ingest_directory(
        tmp_path, source_tag="corpus", embedder=fake_embedder, store=store, settings=settings
    )
    assert store.count() == 1 < before


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping the paid generation leg",
)
@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason='needs the real embedder: pip install -e ".[local-embeddings]"',
)
def test_full_answer_against_claude(tmp_path, store, settings):
    from homebrew_rag.embeddings import build_embedder
    from homebrew_rag.generation import build_generator

    real_settings = Settings(collection=settings.collection)
    (tmp_path / "pricing.md").write_text(
        "# Audit pricing\nThe AI Opportunity Audit is billed at a fixed fee of $7,500 "
        "and completes within three weeks of kickoff."
    )
    embedder = build_embedder(real_settings)
    store.dim = real_settings.embed_dim
    store.client.delete_collection(settings.collection)
    store.ensure_collection()

    ingest_directory(
        tmp_path, source_tag="services", embedder=embedder, store=store, settings=real_settings
    )
    pipeline = RagPipeline(
        embedder=embedder,
        store=store,
        generator=build_generator(real_settings),
        settings=real_settings,
    )
    result = pipeline.answer("What does the AI Opportunity Audit cost?")

    assert "7,500" in result.answer or "7500" in result.answer
    assert result.sources[0]["source"] == "pricing.md"
