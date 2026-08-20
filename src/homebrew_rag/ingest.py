"""The offline pipeline: load → chunk → embed → store.

Run this whenever documents change. It is idempotent: re-ingesting the same
directory overwrites the previous points for those files and drops chunks that
no longer exist, so an index never silently accumulates two versions of a
document that disagree with each other.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunk, chunk_text
from .config import Settings, get_settings
from .embeddings import Embedder, build_embedder
from .store import QdrantStore

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".pdf"}


@dataclass
class IngestReport:
    files: int = 0
    chunks: int = 0
    points: int = 0
    skipped: list[str] = field(default_factory=list)

    def summary(self) -> str:
        line = f"{self.files} files → {self.chunks} chunks → {self.points} points upserted"
        if self.skipped:
            line += f" ({len(self.skipped)} skipped)"
        return line


def iter_documents(directory: Path) -> Iterator[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def load_file_text(path: Path) -> str:
    """Extract plain text from a supported file.

    PDF extraction is best-effort: scanned/image PDFs come back empty and are
    reported as skipped rather than indexed as blank chunks. Run them through
    OCR first if you need them.
    """
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def ingest_directory(
    directory: str | Path,
    source_tag: str = "",
    embedder: Embedder | None = None,
    store: QdrantStore | None = None,
    settings: Settings | None = None,
    replace_tag: bool = False,
) -> IngestReport:
    settings = settings or get_settings()
    embedder = embedder or build_embedder(settings)
    store = store or QdrantStore.from_settings(settings)

    dir_path = Path(directory).expanduser().resolve()
    if not dir_path.is_dir():
        raise NotADirectoryError(f"{dir_path} is not a directory")

    store.ensure_collection()
    if replace_tag:
        logger.info("Clearing existing points for source_tag=%r", source_tag)
        store.delete_tag(source_tag)

    report = IngestReport()

    for path in iter_documents(dir_path):
        relative = str(path.relative_to(dir_path))
        text = load_file_text(path)
        if not text.strip():
            logger.warning("No extractable text in %s — skipping", relative)
            report.skipped.append(relative)
            continue

        chunks: list[Chunk] = chunk_text(
            text,
            source=relative,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        if not chunks:
            report.skipped.append(relative)
            continue

        # Clear the previous version of this document before writing the new one,
        # so a shortened file does not leave orphaned chunks behind.
        if not replace_tag:
            store.delete_source(source=relative, source_tag=source_tag)

        written = 0
        batch_size = settings.embed_batch_size
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = embedder.embed_documents([c.text for c in batch])
            written += store.upsert_chunks(batch, vectors, source_tag=source_tag)

        report.files += 1
        report.chunks += len(chunks)
        report.points += written
        logger.info("%s → %d chunks", relative, len(chunks))

    logger.info("Ingest complete: %s", report.summary())
    return report
