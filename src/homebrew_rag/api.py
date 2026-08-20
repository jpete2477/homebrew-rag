"""FastAPI layer.

Bound to the LAN, not the internet. Two things stand between this and the rest
of the network: the ufw rule (who can reach the port) and the optional API key
below (who may call it). Set RAG_API_KEY before anyone other than you can reach
the host, and definitely before it indexes anything client-owned.
"""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings, get_settings
from .generation import GenerationError
from .logging_setup import configure_logging
from .pipeline import RagPipeline

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"

_pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline.from_settings(get_settings())
    return _pipeline


def set_pipeline(pipeline: RagPipeline | None) -> None:
    """Injection point for tests and for embedding this app in a larger one."""
    global _pipeline
    _pipeline = pipeline


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """No-op when RAG_API_KEY is unset; enforced the moment it is set."""
    expected = settings.api_key
    if not expected:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
        )


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    source_tag: str | None = None


class SourceOut(BaseModel):
    source: str
    section: str = ""
    score: float
    source_tag: str = ""
    chunk_index: int = 0
    text: str = ""


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]
    model: str = ""
    timings_ms: dict = {}
    usage: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info(
        "Starting homebrew-rag %s (collection=%s, model=%s, auth=%s)",
        __version__,
        settings.collection,
        settings.claude_model,
        "on" if settings.api_key else "OFF",
    )
    if not settings.api_key:
        logger.warning("RAG_API_KEY is unset — /query is open to anyone who can reach this port.")
    # Load the embedding model at startup so the first user request is not the
    # one that pays the several-second model load.
    pipeline = get_pipeline()
    warm = getattr(pipeline.embedder, "warm", None)
    if callable(warm):
        warm()
    yield


app = FastAPI(
    title="Homebrew RAG",
    version=__version__,
    description="Retrieval-augmented Q&A over a local document collection.",
    lifespan=lifespan,
)


@app.exception_handler(GenerationError)
def generation_error_handler(request: Request, exc: GenerationError) -> JSONResponse:
    """Retrieval succeeded but generation did not — that is an upstream failure,
    not a bug in this service, so it gets a 502 and a readable message."""
    logger.error("Generation failed: %s", exc)
    return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content={"detail": str(exc)})


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict:
    store_ok = get_pipeline().store.healthy()
    return {
        "status": "ok" if store_ok else "degraded",
        "version": __version__,
        "qdrant": "up" if store_ok else "unreachable",
        "collection": settings.collection,
        "auth_required": bool(settings.api_key),
    }


@app.get("/stats", dependencies=[Depends(require_api_key)])
def stats() -> dict:
    store = get_pipeline().store
    tags = store.source_tags()
    return {
        "collection": store.collection,
        "points": store.count(),
        "source_tags": {tag: store.count(source_tag=tag) for tag in tags},
    }


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(req: QueryRequest, settings: Settings = Depends(get_settings)) -> dict:
    result = get_pipeline().answer(req.question, top_k=req.top_k, source_tag=req.source_tag)
    logger.info(
        "answered question=%r sources=%s retrieval_ms=%d generation_ms=%d tokens=%d/%d",
        result.question,
        [s["source"] for s in result.sources],
        result.retrieval_ms,
        result.generation_ms,
        result.input_tokens,
        result.output_tokens,
    )
    if settings.log_answers:
        logger.info("answer=%r", result.answer)
    return result.to_dict()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
