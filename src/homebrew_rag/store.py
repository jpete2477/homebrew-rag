"""Qdrant-backed vector store.

Wraps the bits of the Qdrant client the pipeline actually uses, so ingestion,
retrieval and the API never touch `qdrant_client.models` directly. Two things
here are worth more than they look:

* **Deterministic point IDs.** A chunk's ID is a UUID5 of
  (source_tag, source, chunk_index), so re-ingesting a document overwrites its
  points instead of piling up duplicates next to the originals.
* **Stale-chunk deletion.** Before upserting a document's chunks we delete
  anything previously indexed under the same (source_tag, source). If an edit
  shortens a file, the chunks that no longer exist go away — otherwise the
  index keeps confidently citing text you deleted last month.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from .chunking import Chunk
from .config import Settings, get_settings

logger = logging.getLogger(__name__)

# Fixed namespace so point IDs are stable across machines and runs.
POINT_NAMESPACE = uuid.UUID("6f9d1a4e-6b3a-5f2e-9a3c-1d7c4b0e5a21")

# Payload fields we filter on; Qdrant needs an index on each for fast filtering.
INDEXED_FIELDS = ("source_tag", "source")


def point_id(source_tag: str, source: str, chunk_index: int) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{source_tag}|{source}|{chunk_index}"))


@dataclass
class SearchHit:
    text: str
    source: str
    section: str
    score: float
    source_tag: str = ""
    chunk_index: int = 0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "section": self.section,
            "score": round(self.score, 4),
            "source_tag": self.source_tag,
            "chunk_index": self.chunk_index,
        }


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int) -> None:
        self.client = client
        self.collection = collection
        self.dim = dim

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> QdrantStore:
        settings = settings or get_settings()
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=60,
        )
        return cls(client=client, collection=settings.collection, dim=settings.embed_dim)

    # --- schema ---------------------------------------------------------

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=self.dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info("Created collection %r (dim=%d, cosine)", self.collection, self.dim)

        for field in INDEXED_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception:  # noqa: BLE001 - index already exists is the common case
                logger.debug("Payload index on %r already present", field)

    # --- writes ---------------------------------------------------------

    def delete_source(self, source: str, source_tag: str = "") -> None:
        """Remove every point previously indexed for one document."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="source", match=qmodels.MatchValue(value=source)
                        ),
                        qmodels.FieldCondition(
                            key="source_tag", match=qmodels.MatchValue(value=source_tag)
                        ),
                    ]
                )
            ),
            wait=True,
        )

    def delete_tag(self, source_tag: str) -> None:
        """Remove an entire logical corpus (everything sharing one source_tag)."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="source_tag", match=qmodels.MatchValue(value=source_tag)
                        )
                    ]
                )
            ),
            wait=True,
        )

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        source_tag: str = "",
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return 0

        points = [
            qmodels.PointStruct(
                id=point_id(source_tag, chunk.source, chunk.chunk_index),
                vector=list(vector),
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                    "source_tag": source_tag,
                    **chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    # --- reads ----------------------------------------------------------

    def search(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        source_tag: str | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        query_filter = None
        if source_tag:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source_tag", match=qmodels.MatchValue(value=source_tag)
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
            SearchHit(
                text=point.payload.get("text", ""),
                source=point.payload.get("source", "unknown"),
                section=point.payload.get("section", ""),
                score=point.score,
                source_tag=point.payload.get("source_tag", ""),
                chunk_index=point.payload.get("chunk_index", 0),
            )
            for point in response.points
        ]

    def count(self, source_tag: str | None = None) -> int:
        count_filter = None
        if source_tag:
            count_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source_tag", match=qmodels.MatchValue(value=source_tag)
                    )
                ]
            )
        return self.client.count(
            collection_name=self.collection, count_filter=count_filter, exact=True
        ).count

    def source_tags(self) -> list[str]:
        """Distinct source_tag values currently indexed (scrolls payloads)."""
        tags: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=512,
                offset=offset,
                with_payload=["source_tag"],
                with_vectors=False,
            )
            tags.update(p.payload.get("source_tag", "") for p in points)
            if offset is None:
                break
        return sorted(t for t in tags if t)

    def healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001 - health check must never raise
            return False
