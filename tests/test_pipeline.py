from __future__ import annotations

import pytest

from homebrew_rag.config import Settings
from homebrew_rag.generation import build_prompt
from homebrew_rag.pipeline import NO_CONTEXT_ANSWER, RagPipeline


@pytest.fixture
def pipeline(fake_embedder, fake_store, fake_generator, settings):
    return RagPipeline(
        embedder=fake_embedder, store=fake_store, generator=fake_generator, settings=settings
    )


def test_retrieve_applies_query_prefix_free_text(pipeline, fake_embedder, fake_store, hit_factory):
    fake_store.hits = [hit_factory("a.md"), hit_factory("b.md")]
    hits = pipeline.retrieve("what is the audit?")
    assert [h.source for h in hits] == ["a.md", "b.md"]
    assert fake_embedder.queries_seen == ["what is the audit?"]
    assert fake_store.searches[0]["top_k"] == 3  # settings.top_k


def test_retrieve_passes_top_k_and_tag_through(pipeline, fake_store, hit_factory):
    fake_store.hits = [hit_factory("a.md")]
    pipeline.retrieve("q", top_k=1, source_tag="agreements")
    assert fake_store.searches[-1] == {
        "vector": fake_store.searches[-1]["vector"],
        "top_k": 1,
        "source_tag": "agreements",
    }


def test_answer_returns_grounded_result(pipeline, fake_store, fake_generator, hit_factory):
    fake_store.hits = [hit_factory("a.md", score=0.83)]
    result = pipeline.answer("what does the audit cost?")

    assert result.answer == fake_generator.text
    assert result.model == "claude-opus-5"
    assert [s["source"] for s in result.sources] == ["a.md"]
    assert result.sources[0]["score"] == pytest.approx(0.83)
    assert result.input_tokens == 100 and result.output_tokens == 20
    assert fake_generator.calls[0][0] == "what does the audit cost?"


def test_answer_short_circuits_without_hits(pipeline, fake_generator):
    result = pipeline.answer("nothing matches this")
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []
    assert fake_generator.calls == []  # no API call is made when retrieval is empty


def test_retrieval_only_pipeline_refuses_to_generate(fake_embedder, fake_store, hit_factory):
    fake_store.hits = [hit_factory("a.md")]
    pipeline = RagPipeline(
        embedder=fake_embedder, store=fake_store, generator=None, settings=Settings()
    )
    assert pipeline.retrieve("q")  # retrieval still works
    with pytest.raises(RuntimeError, match="retrieval only"):
        pipeline.answer("q")


def test_source_previews_are_truncated(pipeline, fake_store, hit_factory):
    fake_store.hits = [hit_factory("a.md", text="word " * 400)]
    preview = pipeline.answer("q").sources[0]["text"]
    assert len(preview) <= 300
    assert preview.endswith("…")


def test_build_prompt_labels_every_chunk(hit_factory):
    hits = [hit_factory("a.md", text="alpha"), hit_factory("b.md", text="beta")]
    prompt = build_prompt("Why?", hits)
    assert "[source: a.md]" in prompt and "[source: b.md]" in prompt
    assert "[section: ## Heading]" in prompt
    assert "alpha" in prompt and "beta" in prompt
    assert prompt.rstrip().endswith("Question: Why?")
