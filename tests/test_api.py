from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from homebrew_rag import api
from homebrew_rag.config import Settings, get_settings
from homebrew_rag.pipeline import RagPipeline


@pytest.fixture
def client(fake_embedder, fake_store, fake_generator, hit_factory):
    fake_store.hits = [hit_factory("audit.md", score=0.91, tag="services")]
    api.set_pipeline(
        RagPipeline(
            embedder=fake_embedder,
            store=fake_store,
            generator=fake_generator,
            settings=Settings(),
        )
    )
    with TestClient(api.app) as test_client:
        yield test_client
    api.set_pipeline(None)


def test_health_reports_store_state(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["qdrant"] == "up"
    assert body["auth_required"] is False


def test_query_returns_answer_and_sources(client, fake_generator):
    response = client.post("/query", json={"question": "What is the audit?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == fake_generator.text
    assert body["sources"][0]["source"] == "audit.md"
    assert body["usage"]["input_tokens"] == 100
    assert "retrieval" in body["timings_ms"]


def test_query_honours_top_k_and_source_tag(client, fake_store):
    client.post("/query", json={"question": "q", "top_k": 2, "source_tag": "services"})
    assert fake_store.searches[-1]["top_k"] == 2
    assert fake_store.searches[-1]["source_tag"] == "services"


def test_blank_question_is_rejected(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_stats_lists_indexed_tags(client):
    body = client.get("/stats").json()
    assert body["points"] == 1
    assert body["source_tags"] == {"services": 1}


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Homebrew RAG" in response.text


class TestApiKeyAuth:
    @pytest.fixture(autouse=True)
    def _require_key(self, monkeypatch):
        monkeypatch.setenv("RAG_API_KEY", "s3cret")
        get_settings.cache_clear()
        yield
        get_settings.cache_clear()

    def test_query_without_key_is_rejected(self, client):
        assert client.post("/query", json={"question": "q"}).status_code == 401

    def test_query_with_wrong_key_is_rejected(self, client):
        response = client.post("/query", json={"question": "q"}, headers={"X-API-Key": "wrong"})
        assert response.status_code == 401

    def test_query_with_correct_key_succeeds(self, client):
        response = client.post("/query", json={"question": "q"}, headers={"X-API-Key": "s3cret"})
        assert response.status_code == 200

    def test_health_stays_open_for_probes(self, client):
        body = client.get("/health").json()
        assert body["auth_required"] is True


def test_generation_failure_returns_502(client, fake_generator, monkeypatch):
    from homebrew_rag.generation import GenerationError

    def boom(question, hits):
        raise GenerationError("Claude API returned 401. Check ANTHROPIC_API_KEY.")

    monkeypatch.setattr(fake_generator, "generate", boom)
    response = client.post("/query", json={"question": "q"})

    assert response.status_code == 502
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]
