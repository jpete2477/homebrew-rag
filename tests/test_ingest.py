from __future__ import annotations

import pytest

from homebrew_rag.config import Settings
from homebrew_rag.ingest import ingest_directory, iter_documents
from homebrew_rag.store import point_id


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "a.md").write_text("# Pricing\nThe audit costs a fixed fee.\n")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.txt").write_text("Turnaround is two weeks.")
    (tmp_path / "ignore.png").write_bytes(b"\x89PNG")
    (tmp_path / "blank.md").write_text("   \n")
    return tmp_path


def test_iter_documents_filters_by_suffix(corpus):
    names = sorted(p.name for p in iter_documents(corpus))
    assert names == ["a.md", "b.txt", "blank.md"]


def test_ingest_indexes_supported_files(corpus, fake_embedder, fake_store):
    report = ingest_directory(
        corpus,
        source_tag="test",
        embedder=fake_embedder,
        store=fake_store,
        settings=Settings(embed_dim=4),
    )
    assert report.files == 2
    assert report.points == report.chunks > 0
    assert report.skipped == ["blank.md"]
    assert fake_store.ensured is True


def test_ingest_uses_paths_relative_to_the_corpus_root(corpus, fake_embedder, fake_store):
    ingest_directory(
        corpus,
        source_tag="test",
        embedder=fake_embedder,
        store=fake_store,
        settings=Settings(embed_dim=4),
    )
    sources = {c.source for chunks, _, _ in fake_store.upserted for c in chunks}
    assert sources == {"a.md", "nested/b.txt"}


def test_ingest_clears_prior_chunks_for_each_document(corpus, fake_embedder, fake_store):
    ingest_directory(
        corpus,
        source_tag="test",
        embedder=fake_embedder,
        store=fake_store,
        settings=Settings(embed_dim=4),
    )
    assert ("a.md", "test") in fake_store.deleted_sources
    assert fake_store.deleted_tags == []


def test_replace_flag_wipes_the_whole_tag_once(corpus, fake_embedder, fake_store):
    ingest_directory(
        corpus,
        source_tag="test",
        embedder=fake_embedder,
        store=fake_store,
        settings=Settings(embed_dim=4),
        replace_tag=True,
    )
    assert fake_store.deleted_tags == ["test"]
    assert fake_store.deleted_sources == []  # no per-file deletes needed after a wipe


def test_missing_directory_raises(tmp_path, fake_embedder, fake_store):
    with pytest.raises(NotADirectoryError):
        ingest_directory(
            tmp_path / "nope", embedder=fake_embedder, store=fake_store, settings=Settings()
        )


def test_point_ids_are_stable_and_distinct():
    first = point_id("tag", "a.md", 0)
    assert first == point_id("tag", "a.md", 0)  # re-ingest overwrites, never duplicates
    assert first != point_id("tag", "a.md", 1)
    assert first != point_id("other", "a.md", 0)
