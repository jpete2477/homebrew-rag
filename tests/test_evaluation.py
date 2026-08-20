from __future__ import annotations

import json

import pytest

from homebrew_rag.config import Settings
from homebrew_rag.evaluation import GoldenCase, evaluate, load_golden_set
from homebrew_rag.pipeline import RagPipeline


@pytest.fixture
def retrieval_pipeline(fake_embedder, fake_store):
    return RagPipeline(
        embedder=fake_embedder, store=fake_store, generator=None, settings=Settings()
    )


def test_load_golden_set_accepts_both_shapes(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {"query": "cost?", "expected_source": "pricing.md"},
                    {"query": "term?", "expected_sources": ["a.md", "b.md"], "source_tag": "x"},
                ]
            }
        )
    )
    cases = load_golden_set(path)
    assert cases[0].expected_sources == ["pricing.md"]
    assert cases[1].expected_sources == ["a.md", "b.md"]
    assert cases[1].source_tag == "x"


def test_case_without_expectation_is_rejected():
    with pytest.raises(ValueError, match="expected_source"):
        GoldenCase.from_dict({"query": "cost?"})


def test_recall_and_mrr(retrieval_pipeline, fake_store, hit_factory):
    fake_store.hits = [hit_factory("noise.md"), hit_factory("pricing.md")]
    cases = [
        GoldenCase(query="cost?", expected_sources=["pricing.md"]),  # found at rank 2
        GoldenCase(query="unrelated?", expected_sources=["missing.md"]),  # not found
    ]
    report = evaluate(cases, retrieval_pipeline, top_k=5)

    assert [r.rank for r in report.results] == [2, None]
    assert report.recall == 0.5
    assert report.mrr == pytest.approx(0.25)  # (1/2 + 0) / 2
    assert "Recall@5: 1/2 = 50.0%" in report.render()


def test_empty_report_does_not_divide_by_zero():
    from homebrew_rag.evaluation import EvalReport

    report = EvalReport(top_k=5)
    assert report.recall == 0.0 and report.mrr == 0.0
