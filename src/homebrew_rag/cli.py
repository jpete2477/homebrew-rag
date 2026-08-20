"""Command line entry point: `homebrew-rag <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import get_settings
from .generation import GenerationError
from .logging_setup import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="homebrew-rag",
        description="Ingest documents, query them, evaluate retrieval, serve the API.",
    )
    parser.add_argument("--version", action="version", version=f"homebrew-rag {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="chunk, embed and index a directory")
    p_ingest.add_argument("directory", help="directory to ingest (recursive)")
    p_ingest.add_argument(
        "source_tag",
        nargs="?",
        default="",
        help="logical corpus label, e.g. 'agreements' — used for filtered retrieval",
    )
    p_ingest.add_argument(
        "--replace",
        action="store_true",
        help="delete everything under this source_tag first (full rebuild of that corpus)",
    )

    p_query = sub.add_parser("query", help="ask a question")
    p_query.add_argument("question", nargs="+")
    p_query.add_argument("--top-k", type=int, default=None)
    p_query.add_argument("--source-tag", default=None)
    p_query.add_argument("--json", action="store_true", help="emit the full result as JSON")
    p_query.add_argument(
        "--retrieve-only",
        action="store_true",
        help="show retrieved chunks without calling Claude (no API cost)",
    )

    p_eval = sub.add_parser("eval", help="score retrieval against a golden set")
    p_eval.add_argument("golden_set", nargs="?", default="eval/golden_set.json")
    p_eval.add_argument("--top-k", type=int, default=None)

    sub.add_parser("stats", help="show what is currently indexed")

    p_serve = sub.add_parser("serve", help="run the FastAPI app with uvicorn")
    p_serve.add_argument("--host", default="0.0.0.0")  # noqa: S104 - LAN service, see §11
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")

    return parser


def _cmd_ingest(args) -> int:
    from .ingest import ingest_directory

    report = ingest_directory(args.directory, source_tag=args.source_tag, replace_tag=args.replace)
    print(report.summary())
    for skipped in report.skipped:
        print(f"  skipped (no extractable text): {skipped}")
    return 0


def _cmd_query(args) -> int:
    from .pipeline import RagPipeline

    question = " ".join(args.question)
    pipeline = RagPipeline.from_settings()

    if args.retrieve_only:
        hits = pipeline.retrieve(question, top_k=args.top_k, source_tag=args.source_tag)
        if args.json:
            print(json.dumps([hit.to_dict() for hit in hits], indent=2))
            return 0
        if not hits:
            print("No chunks retrieved.")
            return 1
        for i, hit in enumerate(hits, start=1):
            print(f"{i}. {hit.source}  (score {hit.score:.3f})")
            body = hit.text
            if hit.section and body.startswith(hit.section):
                print(f"   {hit.section}")
                body = body[len(hit.section) :]
            print(f"   {' '.join(body.split())[:280]}\n")
        return 0

    result = pipeline.answer(question, top_k=args.top_k, source_tag=args.source_tag)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(result.answer)
    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(f"  - {source['source']} (score {source['score']})")
    print(
        f"\n[{result.model} · retrieval {result.retrieval_ms} ms · "
        f"generation {result.generation_ms} ms · "
        f"{result.input_tokens} in / {result.output_tokens} out tokens]"
    )
    return 0


def _cmd_eval(args) -> int:
    from .embeddings import build_embedder
    from .evaluation import evaluate, load_golden_set
    from .pipeline import RagPipeline
    from .store import QdrantStore

    path = Path(args.golden_set)
    if not path.exists():
        print(
            f"No golden set at {path}. Copy eval/golden_set.example.json to {path} "
            f"and replace the cases with questions about your own corpus.",
            file=sys.stderr,
        )
        return 2

    settings = get_settings()
    cases = load_golden_set(path)
    # Retrieval-only: no generator, so this never spends an API call.
    pipeline = RagPipeline(
        embedder=build_embedder(settings),
        store=QdrantStore.from_settings(settings),
        generator=None,
        settings=settings,
    )
    report = evaluate(cases, pipeline, top_k=args.top_k or settings.top_k)
    print(report.render())
    return 0 if report.recall == 1.0 else 1


def _cmd_stats(args) -> int:
    from .store import QdrantStore

    store = QdrantStore.from_settings()
    if not store.healthy():
        print(f"Qdrant unreachable at {get_settings().qdrant_url}", file=sys.stderr)
        return 2
    if not store.client.collection_exists(store.collection):
        print(f"Collection {store.collection!r} does not exist yet — nothing ingested.")
        return 0
    print(f"collection: {store.collection}")
    print(f"points:     {store.count()}")
    tags = store.source_tags()
    if tags:
        print("source tags:")
        for tag in tags:
            print(f"  {tag}: {store.count(source_tag=tag)}")
    return 0


def _cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run(
        "homebrew_rag.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


COMMANDS = {
    "ingest": _cmd_ingest,
    "query": _cmd_query,
    "eval": _cmd_eval,
    "stats": _cmd_stats,
    "serve": _cmd_serve,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging(get_settings().log_level)
    try:
        return COMMANDS[args.command](args)
    except GenerationError as exc:
        # An upstream failure (bad key, API down) is not a bug in this tool —
        # print it plainly instead of dumping a traceback at the user.
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
