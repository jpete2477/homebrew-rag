"""The query pipeline: embed → retrieve → generate.

This is the online half of the system. The offline half lives in `ingest.py`;
keeping them apart is the point — they run on different schedules, fail in
different ways, and share nothing but the collection schema.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .config import Settings, get_settings
from .embeddings import Embedder, build_embedder
from .generation import Generator, build_generator
from .store import QdrantStore, SearchHit

logger = logging.getLogger(__name__)


@dataclass
class RagResult:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    model: str = ""
    retrieval_ms: int = 0
    generation_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "model": self.model,
            "timings_ms": {"retrieval": self.retrieval_ms, "generation": self.generation_ms},
            "usage": {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens},
        }


NO_CONTEXT_ANSWER = (
    "No relevant documents were found for that question. Either the corpus does not "
    "cover it, or nothing has been ingested yet — check `homebrew-rag stats`."
)


class RagPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: QdrantStore,
        generator: Generator | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.generator = generator
        self.settings = settings or get_settings()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RagPipeline:
        settings = settings or get_settings()
        return cls(
            embedder=build_embedder(settings),
            store=QdrantStore.from_settings(settings),
            generator=build_generator(settings),
            settings=settings,
        )

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        source_tag: str | None = None,
    ) -> list[SearchHit]:
        vector = self.embedder.embed_query(question)
        return self.store.search(
            vector=vector,
            top_k=top_k or self.settings.top_k,
            source_tag=source_tag,
            score_threshold=self.settings.score_threshold,
        )

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        source_tag: str | None = None,
    ) -> RagResult:
        started = time.perf_counter()
        hits = self.retrieve(question, top_k=top_k, source_tag=source_tag)
        retrieval_ms = int((time.perf_counter() - started) * 1000)

        logger.info(
            "retrieved=%d source_tag=%s top_score=%s question=%r",
            len(hits),
            source_tag or "*",
            f"{hits[0].score:.3f}" if hits else "n/a",
            question,
        )

        if not hits:
            return RagResult(
                question=question,
                answer=NO_CONTEXT_ANSWER,
                retrieval_ms=retrieval_ms,
            )

        if self.generator is None:
            raise RuntimeError("This pipeline was built without a generator (retrieval only).")

        started = time.perf_counter()
        generated = self.generator.generate(question, hits)
        generation_ms = int((time.perf_counter() - started) * 1000)

        return RagResult(
            question=question,
            answer=generated.text,
            sources=[hit.to_dict() | {"text": _preview(hit.text)} for hit in hits],
            model=generated.model,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
        )


def _preview(text: str, limit: int = 300) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"
