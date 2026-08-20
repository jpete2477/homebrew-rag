# homebrew-rag

A Retrieval-Augmented Generation stack you can actually run: **Qdrant** for vectors,
**local embeddings** so documents never leave the machine, and **Claude** for the
final answer. No LangChain, no LlamaIndex — the pipeline is a few hundred lines of
plain Python, which is the point. You can read all of it.

[![CI](https://github.com/jpete2477/homebrew-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/jpete2477/homebrew-rag/actions/workflows/ci.yml)

The narrative walkthrough this repo grew out of lives in [docs/tutorial.md](docs/tutorial.md).

---

## Why these pieces

| Layer | Choice | Reason |
|---|---|---|
| Vector store | Qdrant (Docker) | Clean Python client, good metadata filtering, one-command start |
| Embeddings | `BAAI/bge-base-en-v1.5` via `sentence-transformers` | Runs on CPU, no API key, **no document text leaves the host** |
| Generation | Claude (`claude-opus-5`) via the Anthropic SDK | Direct API calls; nothing to debug through a framework |
| API | FastAPI + uvicorn | Small, async, trivially containerized |
| Orchestration | Plain Python | Hand-rolling the pipeline teaches you what the frameworks hide |

**Data exposure, concretely:** ingestion and retrieval are entirely local. The only
outbound network call is the generation request, which carries the question plus the
top-k retrieved snippets — not the corpus. That's a meaningfully smaller surface than
an all-API pipeline if you ever point this at client material.

---

## Architecture

```
  Documents ──▶  ingest.py  ──▶  chunking  ──▶  embeddings  ──▶  Qdrant
  (md/txt/pdf)   (offline)       (structure-     (local, CPU)     (vectors +
                                  aware, with                      payload)
                                  overlap)
                                                                     │
  Question  ──▶  pipeline.py ──▶  embed query  ──▶  similarity search ┘
  (online)                                              │
                                                        ▼
                                        generation.py ──▶  Claude API
                                                        │   (top-k snippets only)
                                                        ▼
                                            answer + cited sources
```

Two pipelines, deliberately separate: **ingestion** runs when documents change,
**query** runs per request. They share only the collection schema.

---

## Quickstart

```bash
git clone https://github.com/jpete2477/homebrew-rag.git
cd homebrew-rag

make setup      # venv + CPU-only PyTorch + deps + .env  (add your ANTHROPIC_API_KEY)
make up         # Qdrant on :6333
make ingest     # index documents/sample under the tag "demo"
make query Q="What does the audit cost?"
make serve      # API + web UI on :8000
```

`make` on its own lists every target. The equivalent without Make:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[local-embeddings,dev]"      # or: pip install -r requirements.txt
cp .env.example .env
docker compose up -d
homebrew-rag ingest documents/sample demo
homebrew-rag query "What does the audit cost?"
homebrew-rag serve
```

First run downloads PyTorch (~2 GB, once) and the embedding model into
`~/.cache/huggingface` (~440 MB, once). After that, ingestion and retrieval are
fully offline.

Open <http://localhost:8000> for the query page, or call it directly:

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What does the audit cost?", "top_k": 5}'
```

---

## Make targets

`make setup` · `make up` / `make down` · `make ingest` / `make reingest` ·
`make query Q="..."` · `make retrieve Q="..."` (free — no Claude call) ·
`make eval` · `make stats` · `make serve` / `make dev` ·
`make test` / `make test-integration` · `make lint` / `make format` / `make check` ·
`make docker-up` · `make install-service` · `make clean` / `make clean-index`

Variables override on the command line:

```bash
make ingest DIR=./docs/agreements TAG=agreements
make query Q="what does the agreement say about recording consent?" TOP_K=8
make serve PORT=9000
```

---

## CLI

Everything Make wraps is a plain CLI command, if you'd rather call it directly.

| Command | What it does |
|---|---|
| `homebrew-rag ingest DIR [TAG] [--replace]` | Chunk, embed and index a directory. Idempotent. |
| `homebrew-rag query "..." [--top-k N] [--source-tag T] [--json]` | Ask a question. |
| `homebrew-rag query "..." --retrieve-only` | Show retrieved chunks without calling Claude — free. |
| `homebrew-rag eval [PATH] [--top-k N]` | Score retrieval against a golden set. Exits non-zero on any miss. |
| `homebrew-rag stats` | What's indexed, broken down by source tag. |
| `homebrew-rag serve [--host H] [--port P] [--reload]` | Run the API. |

### Tagging a corpus

Ingest by category rather than flat, and retrieval can be filtered to the bucket the
answer actually lives in — meaningfully better precision than searching everything:

```bash
homebrew-rag ingest ./docs/services   services
homebrew-rag ingest ./docs/agreements agreements
homebrew-rag query "recording consent?" --source-tag agreements
```

**Re-ingesting is safe.** Point IDs are derived from `(source_tag, source, chunk_index)`,
and each document's old chunks are deleted before the new ones are written. Editing a
file updates its chunks in place; shortening one drops the chunks that no longer exist.
An index that keeps confidently citing last quarter's pricing is worse than no index.

---

## HTTP API

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | open | Liveness + Qdrant reachability. Safe for probes. |
| `POST /query` | `X-API-Key` when `RAG_API_KEY` is set | Answer a question with citations. |
| `GET /stats` | same | Point counts per source tag. |
| `GET /` | open | Single-page query UI. |
| `GET /docs` | open | OpenAPI schema (FastAPI). |

```jsonc
// POST /query
{ "question": "What does the audit cost?", "top_k": 5, "source_tag": "services" }

// 200
{
  "question": "...",
  "answer": "The audit is a fixed fee of $7,500. [source: audit-pricing.md]",
  "sources": [{ "source": "audit-pricing.md", "section": "# Pricing", "score": 0.81, "text": "…" }],
  "model": "claude-opus-5",
  "timings_ms": { "retrieval": 38, "generation": 2610 },
  "usage": { "input_tokens": 1204, "output_tokens": 96 }
}
```

---

## Configuration

Everything is environment-driven — see [.env.example](.env.example) for the full list
with comments. The ones you'll actually touch:

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for generation only. Retrieval and eval work without it. |
| `RAG_CLAUDE_MODEL` | `claude-opus-5` | |
| `QDRANT_URL` | `http://localhost:6333` | |
| `RAG_COLLECTION` | `documents` | Separate collections give stronger isolation than tags. |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `500` / `75` | Words, not tokens. Re-ingest and re-run the eval after changing. |
| `RAG_TOP_K` | `5` | |
| `RAG_EMBED_MODEL` / `RAG_EMBED_DIM` | `BAAI/bge-base-en-v1.5` / `768` | Change both together; re-create the collection. |
| `RAG_API_KEY` | unset | When unset, `/query` is open to anyone who can reach the port. |

---

## Testing

Two kinds of test, and they answer different questions.

**Plumbing** — does the pipeline run and return well-formed data? Fast, hermetic, no
Qdrant, no API key, no PyTorch (a fake embedder stands in):

```bash
make test                   # unit tests only, ~1s
make test-integration       # starts Qdrant first
make check                  # lint + format check + unit tests, i.e. what CI runs
```

The generation leg of the integration suite self-skips unless `ANTHROPIC_API_KEY` is
set, so nothing costs money by accident.

**Retrieval quality** — does it find the *right* chunks? This is the part most RAG
projects skip, and the part that decides whether the system is trustworthy:

```bash
make eval    # seeds eval/golden_set.json from the example on first run
```

```
[PASS] What does the AI Opportunity Audit cost?
        expected: audit-pricing.md (rank 1)
        got:      audit-pricing.md, engagement-terms.md
[FAIL] What are the payment terms?
        expected: engagement-terms.md (not retrieved)
        got:      audit-pricing.md, audit-pricing.md

Recall@5: 5/6 = 83.3%   MRR: 0.792
```

Build the golden set from questions you'd genuinely ask — 15–30 of them — and re-run
it after every change to chunking, the embedding model, or `top_k`. It's the only
thing that tells you whether a "smarter" tweak improved retrieval or just felt like it
did. `eval/golden_set.json` is gitignored, so real questions about real documents stay
local; the example file is the template.

---

## Deploying

```bash
make docker-up          # Qdrant + the API, both in containers
make install-service    # or: Qdrant in Docker, API under systemd (edit the unit first)
```

Firewall setup for LAN-only exposure is in [deploy/ufw.md](deploy/ufw.md).

### Before calling it deployed

- [ ] **Process management** — systemd or Docker, not a `uvicorn` in a terminal that dies with your SSH session.
- [ ] **Auth** — set `RAG_API_KEY`. The firewall controls *who can reach the port*; it authenticates nobody.
- [ ] **TLS** — not needed on a trusted LAN. The moment this is reachable from anywhere else, put Caddy or Nginx in front. Never expose uvicorn directly.
- [ ] **Secrets** — `.env` is gitignored; `chmod 600 .env` on a shared host.
- [ ] **Cost** — embeddings are free and local, but every `/query` is a Claude call. Rate-limit before sharing.
- [ ] **Backups** — the `qdrant_storage` volume is the index; keep the source documents too so you can always rebuild.
- [ ] **Re-ingest on change** — schedule it or hook it to file changes. A stale index that cites old pricing confidently is worse than no index.

---

## Repo layout

The tutorial's flat scripts map onto modules like this:

| Tutorial (`docs/tutorial.md`) | Here |
|---|---|
| `chunking.py` | [`src/homebrew_rag/chunking.py`](src/homebrew_rag/chunking.py) |
| `ingest.py` | [`ingest.py`](src/homebrew_rag/ingest.py) + [`embeddings.py`](src/homebrew_rag/embeddings.py) + [`store.py`](src/homebrew_rag/store.py) |
| `query.py` | [`pipeline.py`](src/homebrew_rag/pipeline.py) + [`generation.py`](src/homebrew_rag/generation.py) |
| `app.py` | [`api.py`](src/homebrew_rag/api.py) + [`web/index.html`](src/homebrew_rag/web/index.html) |
| `eval_retrieval.py` | [`evaluation.py`](src/homebrew_rag/evaluation.py) |
| `test_pipeline.py` | [`tests/`](tests/) |
| the `bash` blocks | [`Makefile`](Makefile) |

Splitting ingest into three modules isn't ceremony: `embeddings.py` is the seam where
you'd swap the local model for an API-based one (Voyage, say) without touching
anything else, and `store.py` is what lets the pipeline be tested without Qdrant.

**Documents are gitignored.** Anything in `documents/` other than `sample/` stays
local — this repo is public, and a business corpus doesn't belong in it.

---

## Where this goes next

- **Hybrid search** (vector + BM25) — Qdrant supports it natively. Reach for it when you
  notice pure-vector search missing exact terms: a dollar figure, a clause number.
- **Re-ranking** — retrieve top 20, re-rank to top 5 with a cross-encoder or Claude
  itself. Better precision on ambiguous questions.
- **Query rewriting** — for multi-turn use, have Claude rewrite "what about the second
  option?" into a standalone query before embedding it. Follow-ups embed terribly on
  their own.
- **Citation verification** — right now citations are enforced by prompt. Post-process
  the answer and check every cited source actually appears in the retrieved set.

## License

MIT — see [LICENSE](LICENSE).
