"""Structure-aware chunking.

Retrieval quality is decided here more than anywhere else in the pipeline. The
strategy: split on markdown-style headings first so a chunk is a coherent idea,
then window inside long sections with overlap so no idea is severed at a
boundary. Every chunk keeps the heading it came from — both as retrieval signal
and so citations can point at a section, not just a filename.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

HEADING_RE = re.compile(r"^(#{1,6}[ \t]+\S.*)$", re.MULTILINE)


@dataclass
class Chunk:
    """One indexable unit of text plus the metadata needed to cite it."""

    text: str
    source: str
    section: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split on markdown headings, returning (heading, body) pairs.

    Text before the first heading is returned with an empty heading. A document
    with no headings comes back as a single ("", text) pair.
    """
    parts = HEADING_RE.split(text)
    if len(parts) == 1:
        return [("", text)] if text.strip() else []

    sections: list[tuple[str, str]] = []
    # parts alternates: [preamble, heading, body, heading, body, ...]
    if parts[0].strip():
        sections.append(("", parts[0]))
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((heading, body))
    return sections


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 75,
) -> list[Chunk]:
    """Chunk a document into overlapping, heading-prefixed windows.

    `chunk_size` and `overlap` are counted in whitespace-delimited words, which
    tracks token count closely enough for sizing decisions (roughly 1.3 tokens
    per word for English prose) without importing a tokenizer.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    stride = chunk_size - overlap
    chunks: list[Chunk] = []

    for heading, body in split_into_sections(text):
        words = body.split()
        if not words:
            # A heading with no body still carries meaning (e.g. a section title
            # answering "what topics exist?"), but not enough to index alone.
            continue
        for start in range(0, len(words), stride):
            window = words[start : start + chunk_size]
            if not window:
                continue
            body_text = " ".join(window)
            chunk_str = f"{heading}\n\n{body_text}" if heading else body_text
            chunks.append(
                Chunk(
                    text=chunk_str.strip(),
                    source=source,
                    section=heading,
                    chunk_index=len(chunks),
                    metadata={"word_offset": start},
                )
            )
            if start + chunk_size >= len(words):
                break  # last window covered the tail; don't emit an overlap-only chunk

    return chunks
