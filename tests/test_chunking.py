from __future__ import annotations

import pytest

from homebrew_rag.chunking import chunk_text, split_into_sections


def test_no_headings_returns_single_section():
    assert split_into_sections("just some prose") == [("", "just some prose")]


def test_empty_text_yields_nothing():
    assert split_into_sections("   ") == []
    assert chunk_text("", source="empty.md") == []


def test_sections_split_on_headings_and_keep_preamble():
    text = "intro line\n\n# First\nbody one\n\n## Second\nbody two\n"
    sections = split_into_sections(text)
    assert [heading for heading, _ in sections] == ["", "# First", "## Second"]
    assert "intro line" in sections[0][1]
    assert "body one" in sections[1][1]


def test_heading_is_prefixed_onto_each_chunk():
    text = "# Pricing\n" + " ".join(f"word{i}" for i in range(120))
    chunks = chunk_text(text, source="pricing.md", chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(chunk.text.startswith("# Pricing") for chunk in chunks)
    assert all(chunk.section == "# Pricing" for chunk in chunks)
    assert all(chunk.source == "pricing.md" for chunk in chunks)


def test_chunk_indices_are_sequential_across_sections():
    text = "# A\n" + " ".join(["a"] * 80) + "\n# B\n" + " ".join(["b"] * 80)
    chunks = chunk_text(text, source="doc.md", chunk_size=30, overlap=5)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap():
    words = [f"w{i}" for i in range(100)]
    chunks = chunk_text(" ".join(words), source="doc.md", chunk_size=40, overlap=10)
    first_tail = chunks[0].text.split()[-10:]
    second_head = chunks[1].text.split()[:10]
    assert first_tail == second_head


def test_short_document_produces_one_chunk():
    chunks = chunk_text("a short doc", source="short.md", chunk_size=500, overlap=75)
    assert len(chunks) == 1
    assert chunks[0].word_count == 3


def test_no_trailing_overlap_only_chunk():
    # 45 words with size 40 / overlap 10: windows at 0 and 30 cover everything;
    # a naive loop would also emit a redundant window starting at 60.
    words = [f"w{i}" for i in range(45)]
    chunks = chunk_text(" ".join(words), source="doc.md", chunk_size=40, overlap=10)
    assert len(chunks) == 2


def test_heading_without_body_is_not_indexed():
    chunks = chunk_text("# Lonely heading\n\n# Another\nreal body", source="d.md")
    assert len(chunks) == 1
    assert chunks[0].section == "# Another"


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, 15), (10, -1)])
def test_invalid_windows_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", source="d.md", chunk_size=size, overlap=overlap)
