# Building & Deploying a RAG System on Ubuntu — End-to-End Tutorial

> **This is the narrative walkthrough the repo grew out of.** The working code
> lives in [`src/homebrew_rag/`](../src/homebrew_rag/) and differs in a few
> places where the shipped version is more careful than a tutorial needs to be:
> deterministic point IDs so re-ingesting cannot duplicate an index, settings
> pulled out into [`config.py`](../src/homebrew_rag/config.py), generation
> behind a protocol so retrieval can be evaluated without spending API calls,
> and current pins (`claude-opus-5`, `anthropic` 0.125) rather than the versions
> named below. Read this for the *why*; read the modules for the *what*.

This walks through a full Retrieval-Augmented Generation (RAG) stack: local setup, ingestion, retrieval, generation, a simple API, tests, and a real deployment scenario. Every piece is something you'd actually run in production, not a toy notebook.

**Stack choice and why:**
- **Vector DB: Qdrant** — runs great in Docker, has a clean Python client, handles metadata filtering well (useful once you have multiple document sources/clients).
- **Embeddings: local, via `sentence-transformers`** — no API key, no per-request cost, nothing leaves your server for the embedding step. Since your test corpus is real Tienta business documents (agreements, audit templates), keeping embedding fully offline means that content never transits a third-party API at all — only the final retrieved snippets go out, and only to Anthropic, when you call Claude for generation. Model: `BAAI/bge-base-en-v1.5` — solid open-weight quality, runs fine on CPU for this scale of corpus.
- **Generation: Claude via Anthropic API** — you already have this relationship; using the API directly (no LangChain) keeps the system legible and easy to debug.
- **API layer: FastAPI** — minimal, async, easy to containerize. Bound to your LAN, not the public internet (see §2 and §11).
- **Orchestration: plain Python** — no LangChain/LlamaIndex. For a learning project, hand-rolling the pipeline teaches you what those frameworks are actually doing under the hood. You can swap in a framework later once you understand the primitives.

---

## 1. Architecture Overview

```
                    ┌─────────────────┐
   Documents  ───▶  │   Ingestion      │
   (PDF/MD/etc)     │   - chunk        │
                     │   - embed        │──▶  Qdrant (vector store)
                     │   - store        │
                     └─────────────────┘

   User Query  ───▶  ┌─────────────────┐
                     │   Retrieval      │──▶  Qdrant similarity search
                     │   - embed query  │
                     │   - fetch top-k  │
                     └────────┬────────┘
                              │ retrieved chunks
                              ▼
                     ┌─────────────────┐
                     │   Generation     │──▶  Claude API (with context)
                     │   - build prompt │
                     │   - call Claude  │
                     └────────┬────────┘
                              ▼
                          Answer + citations
```

Two pipelines: **ingestion** (offline, run when documents change) and **query** (online, runs per user request). Keep them as separate scripts/modules — this is the single most common design mistake beginners make (jamming everything into one script).

---

## 2. Server Prep (Ubuntu)

SSH into your test server and get the baseline tooling in place.

```bash
sudo apt update && sudo apt upgrade -y

# Python 3.11+ and venv
sudo apt install -y python3 python3-venv python3-pip git curl

# Docker (for Qdrant)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker   # or log out/in to pick up the group change

docker --version
python3 --version
```

Create a project directory and virtual environment:

```bash
mkdir -p ~/homebrew-rag && cd ~/homebrew-rag
python3 -m venv venv
source venv/bin/activate
```

**LAN exposure:** since this will be reachable from other machines on your network (not just localhost), find the server's LAN IP now and open the firewall to your subnet only — not to the world.

```bash
ip addr show | grep "inet " | grep -v 127.0.0.1
# should show 10.10.0.130

sudo apt install -y ufw
sudo ufw allow from 10.10.0.0/24 to any port 8000 proto tcp   # FastAPI
sudo ufw allow from 10.10.0.0/24 to any port 6333 proto tcp   # Qdrant, only if you need to inspect it remotely
sudo ufw enable
sudo ufw status
```

You'll hit the API from other devices as `http://10.10.0.130:8000` — no TLS/domain needed for LAN-only use, just don't forward these ports on your router.

---

## 3. Stand Up Qdrant

Use Docker Compose so it's a one-command start/stop and easy to add services later (e.g., the FastAPI app itself).

```bash
cat > docker-compose.yml << 'EOF'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage

volumes:
  qdrant_storage:
EOF

docker compose up -d
```

Verify it's alive:

```bash
curl http://localhost:6333/collections
# should return {"result":{"collections":[]},"status":"ok",...}
```

---

## 4. Python Dependencies

```bash
cat > requirements.txt << 'EOF'
qdrant-client==1.11.3
anthropic==0.39.0
sentence-transformers==3.1.1
fastapi==0.115.0
uvicorn==0.31.0
pypdf==5.0.1
python-dotenv==1.0.1
pytest==8.3.3
EOF

pip install -r requirements.txt
```

First install will pull PyTorch (CPU build) as a dependency of `sentence-transformers` — expect a few minutes and a couple GB of download. The embedding model itself (`BAAI/bge-base-en-v1.5`) downloads once, on first use, and is cached locally (`~/.cache/huggingface`) — no further network calls after that.

Set your Anthropic key (never commit this — this is the only external API this system calls):

```bash
cat > .env << 'EOF'
ANTHROPIC_API_KEY=sk-ant-your-key-here
QDRANT_URL=http://localhost:6333
EOF

echo ".env" >> .gitignore
echo "venv/" >> .gitignore
```

> **If you later want API-based embeddings instead** (e.g., a client engagement where you'd rather offload compute), Voyage AI is Anthropic's recommended embeddings partner and pairs well with Claude. The embedding calls below are isolated in one function specifically so that swap is a five-minute change if you ever want it.

---

## 5. Chunking Strategy

This is the part people underinvest in and then wonder why retrieval quality is poor. Rules of thumb:

- **Chunk size**: 300–600 tokens for prose/technical docs. Too small loses context; too large dilutes relevance and wastes context window.
- **Overlap**: 10–20% overlap between chunks so you don't sever a sentence/idea at a boundary.
- **Respect structure**: split on headings/paragraphs first, then fall back to token-count splitting within long sections. Don't blindly split on raw character count.
- **Keep metadata**: source filename, section heading, page number, chunk index. You'll want this for citations and for filtering (e.g., "only search Client X's documents").

```python
# chunking.py
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    source: str
    section: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split on markdown-style headings; returns (heading, body) pairs."""
    pattern = re.compile(r"^(#{1,3}\s+.*)$", re.MULTILINE)
    parts = pattern.split(text)
    if len(parts) == 1:
        return [("", text)]
    sections = []
    # parts alternates: [preamble, heading, body, heading, body, ...]
    if parts[0].strip():
        sections.append(("", parts[0]))
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((heading, body))
    return sections


def chunk_text(text: str, source: str, chunk_size: int = 500, overlap: int = 75) -> list[Chunk]:
    chunks = []
    idx = 0
    for heading, body in split_into_sections(text):
        words = body.split()
        i = 0
        while i < len(words):
            chunk_words = words[i : i + chunk_size]
            if chunk_words:
                chunk_str = (f"{heading}\n\n" if heading else "") + " ".join(chunk_words)
                chunks.append(
                    Chunk(
                        text=chunk_str.strip(),
                        source=source,
                        section=heading,
                        chunk_index=idx,
                    )
                )
                idx += 1
            i += chunk_size - overlap
    return chunks
```

---

## 6. Ingestion Pipeline (embed + store)

```python
# ingest.py
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from chunking import chunk_text
from pypdf import PdfReader

load_dotenv()

COLLECTION_NAME = "documents"
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"  # 768-dim, runs fine on CPU
EMBED_DIM = 768

# Loaded once per process — this is the slow part (a few seconds), so
# don't call SentenceTransformer(...) inside a loop.
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
qdrant = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))


def ensure_collection():
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION_NAME}'")


def load_file_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(errors="ignore")


def embed_texts(texts: list[str]) -> list[list[float]]:
    # bge models expect a specific instruction prefix on the *document* side
    # for retrieval tasks — this is model-specific, check the model card if
    # you swap to a different embedding model.
    return embed_model.encode(texts, normalize_embeddings=True).tolist()


def ingest_directory(directory: str, source_tag: str = ""):
    ensure_collection()
    dir_path = Path(directory)
    files = [p for p in dir_path.rglob("*") if p.suffix.lower() in (".md", ".txt", ".pdf")]

    all_chunks = []
    for f in files:
        text = load_file_text(f)
        if not text.strip():
            continue
        chunks = chunk_text(text, source=str(f.relative_to(dir_path)))
        all_chunks.extend(chunks)

    print(f"Found {len(files)} files -> {len(all_chunks)} chunks")

    # Batch embed (Voyage batches internally, but keep requests reasonably sized)
    batch_size = 64
    points = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        vectors = embed_texts([c.text for c in batch])
        for chunk, vector in zip(batch, vectors):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk.text,
                        "source": chunk.source,
                        "section": chunk.section,
                        "chunk_index": chunk.chunk_index,
                        "source_tag": source_tag,
                    },
                )
            )
        print(f"Embedded {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}")

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Upserted {len(points)} points into Qdrant.")


if __name__ == "__main__":
    import sys

    directory = sys.argv[1] if len(sys.argv) > 1 else "./documents"
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    ingest_directory(directory, source_tag=tag)
```

Run it against a folder of test documents:

```bash
mkdir -p documents
# drop some .md/.txt/.pdf files in there
python ingest.py ./documents test-corpus
```

---

## 7. Retrieval + Generation

```python
# query.py
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import anthropic

load_dotenv()

COLLECTION_NAME = "documents"
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# bge models want a query-side instruction prefix for best retrieval quality —
# this is the local-model equivalent of Voyage's input_type="query" flag.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
qdrant = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def retrieve(query: str, top_k: int = 5, source_tag: str | None = None) -> list[dict]:
    query_vector = embed_model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()

    query_filter = None
    if source_tag:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_filter = Filter(
            must=[FieldCondition(key="source_tag", match=MatchValue(value=source_tag))]
        )

    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
    ).points

    return [
        {"text": h.payload["text"], "source": h.payload["source"], "score": h.score} for h in hits
    ]


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
    return f"""Answer the question using ONLY the context provided below. \
If the context doesn't contain enough information to answer, say so explicitly \
rather than guessing. Cite the source file for each claim you make.

<context>
{context_blocks}
</context>

Question: {query}"""


def answer(query: str, top_k: int = 5, source_tag: str | None = None) -> dict:
    chunks = retrieve(query, top_k=top_k, source_tag=source_tag)
    if not chunks:
        return {"answer": "No relevant documents found.", "sources": []}

    prompt = build_prompt(query, chunks)
    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "answer": response.content[0].text,
        "sources": [{"source": c["source"], "score": round(c["score"], 3)} for c in chunks],
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is this documentation about?"
    result = answer(q)
    print("ANSWER:\n", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  - {s['source']} (score: {s['score']})")
```

Test it from the command line:

```bash
python query.py "What are the main services Tienta offers?"
```

A few things worth noting about this code, since they're common failure points:
- **The query-side prefix matters.** `bge` models were trained with an asymmetric setup: documents get embedded plain, queries get a specific instruction prefix prepended. Forgetting the prefix (or using the wrong one) silently degrades retrieval quality without throwing an error — it'll just quietly return worse matches — so it's worth double-checking against the model card if you ever swap models.
- **The prompt explicitly permits "I don't know."** Without this instruction, Claude will often try to be helpful and answer from general knowledge even when the retrieved context is thin — which defeats the purpose of RAG (grounding answers in your specific documents) and reintroduces hallucination risk.
- **Citations are enforced by prompt, not by code.** For a stricter guarantee, you'd want to post-process the response to verify cited sources actually appear in the retrieved chunk list — worth adding once you move past testing.

---

## 8. Wrap It in an API

```python
# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from query import answer

app = FastAPI(title="RAG API")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    source_tag: str | None = None


@app.post("/query")
def query_endpoint(req: QueryRequest):
    return answer(req.question, top_k=req.top_k, source_tag=req.source_tag)


@app.get("/health")
def health():
    return {"status": "ok"}
```

Run it bound to all interfaces so other machines on your LAN can reach it (the `ufw` rule from §2 already restricts who can actually connect):

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Test locally first:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the AI Opportunity Audit?"}'
```

Then from another machine on your LAN:

```bash
curl -X POST http://10.10.0.130:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the AI Opportunity Audit?"}'
```

---

## 9. Testing

Two distinct kinds of tests matter for RAG, and they're not the same thing:

**a) Plumbing tests** — does the pipeline run without errors, return well-formed data.

```python
# test_pipeline.py
import pytest
from query import retrieve, answer


def test_retrieve_returns_results():
    results = retrieve("test query", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    if results:
        assert "text" in results[0] and "source" in results[0]


def test_answer_structure():
    result = answer("What is this about?")
    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["sources"], list)
```

```bash
pytest test_pipeline.py -v
```

**b) Retrieval quality tests** — does it actually find the right chunks. This is the part most tutorials skip, and it's the part that matters most. Build a small "golden set": questions where you know which document/section should be retrieved.

```python
# eval_retrieval.py
from query import retrieve

golden_set = [
    {"query": "What does the AI Opportunity Audit cost?", "expected_source": "audit-pricing.md"},
    {"query": "How long does the engagement take?", "expected_source": "audit-pricing.md"},
    # add 15-30 of these covering your real document set
]


def evaluate():
    hits, total = 0, len(golden_set)
    for case in golden_set:
        results = retrieve(case["query"], top_k=5)
        sources = [r["source"] for r in results]
        found = case["expected_source"] in sources
        hits += found
        print(f"{'✓' if found else '✗'} {case['query']!r} -> {sources}")
    print(f"\nRecall@5: {hits}/{total} = {hits / total:.1%}")


if __name__ == "__main__":
    evaluate()
```

Run this after any change to chunking strategy, embedding model, or top_k — it's your regression test for retrieval quality, and it's the thing that tells you whether a "smarter" prompt or model change actually improved anything or just felt like it did.

---

## 10. Real-World Scenario: Tienta Knowledge Base

Since you have real content to work with, the most useful test isn't a toy corpus — it's your own consulting materials. This also doubles as a legitimately useful internal tool.

**Scenario:** an internal Q&A assistant over Tienta's service docs, engagement agreements, past audit templates, and marketing materials — so that when you're prepping for a client call, you can ask "what's included in the AI Opportunity Audit deliverable?" instead of hunting through files.

Steps:

1. **Gather the corpus.** Export/collect: service line descriptions, the AI Opportunity Audit offer doc, engagement agreement templates, the SKILL.md report-generation file, marketing flyer copy, LinkedIn positioning docs. Convert anything not already text/markdown to `.md` or `.pdf`.

2. **Tag by category**, not just ingest flat. Modify `ingest_directory` calls to pass a meaningful `source_tag` per folder:
   ```bash
   python ingest.py ./docs/services services
   python ingest.py ./docs/agreements agreements
   python ingest.py ./docs/marketing marketing
   ```
   This lets you filter queries later (`source_tag="agreements"`) when you know which bucket the answer lives in — meaningfully better precision than searching everything at once.

3. **Build a golden eval set from real questions** you'd actually ask before a client call — "what's the audit turnaround time," "what does the engagement agreement say about recording consent," "what's the pitch for the effort-vs-impact matrix." This is more honest signal than a generic test set because you already know the right answers.

4. **Deploy for actual use**, not just as a demo:
   - Run `docker compose up -d` (Qdrant) and `uvicorn app:app` as a systemd service (below) so it survives reboots.
   - Point a simple frontend at it — even a single HTML page with a text box hitting `/query` is enough for personal use. If you want it usable from your phone, put it behind a reverse proxy with basic auth (see §11).

5. **Re-ingest on a schedule or on change.** Since your service docs/agreements will keep evolving (you've already iterated on the AI Automation Engagement Agreement and the audit SKILL.md multiple times), wire ingestion to run whenever those source files change rather than letting the index go stale silently — a stale RAG index that confidently cites an old pricing figure is worse than no RAG at all.

This scenario is also a legitimate stand-in for a client-facing version later: same architecture, swap Tienta's docs for a client's case files or SOPs, add per-client `source_tag` isolation (or separate Qdrant collections per client for stronger data separation) — relevant if you go this direction for Vero Legal or a similar engagement, since that already puts you in a security-conscious posture around case-file handling. Worth noting: with the local-embeddings setup here, your agreement templates and audit docs never leave the server during ingestion or retrieval — the only network call that ever happens is the final Claude generation request, which sends just the top-k retrieved snippets (not the full corpus) plus the question. That's a meaningfully smaller data-exposure surface than an all-API-calls pipeline, worth keeping in mind if you carry this pattern into a client engagement with data-handling requirements.

---

## 11. Production Hardening Checklist

Once past local testing, before calling this "deployed":

- **Process management**: run the API under `systemd` (below) or Docker, not a bare `uvicorn` in a terminal that dies when you disconnect.
  ```ini
  # /etc/systemd/system/rag-api.service
  [Unit]
  Description=RAG API
  After=network.target docker.service

  [Service]
  User=youruser
  WorkingDirectory=/home/homebrew-rag
  Environment="PATH=/home/homebrew-rag/venv/bin"
  ExecStart=/home/homebrew-rag/venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
  Restart=always

  [Install]
  WantedBy=multi-user.target
  ```
  ```bash
  sudo systemctl enable --now rag-api
  ```
- **Reverse proxy + TLS**: not required for LAN-only use as set up here. If you ever expose this beyond your LAN (port-forwarding, tailscale, a public domain), put Nginx or Caddy in front with TLS first — don't expose uvicorn directly to the internet.
- **Auth**: the `ufw` rule limits *who can reach the port*, but anyone on your LAN can currently call the API with no credentials. Fine for solo testing; add at minimum an API key check on `/query` before letting anyone else on the network use it, and definitely before it touches real client data (as opposed to your own Tienta docs).
- **Secrets**: `.env` should never be committed; on a shared server, lock file permissions (`chmod 600 .env`).
- **Cost control**: embeddings are free/local now, but every `/query` call still costs an Anthropic API call for generation — add basic rate limiting if this becomes reachable by more than just you.
- **Logging**: log query, retrieved sources, and response for every request — this is your debugging trail and your future eval data.
- **Backups**: Qdrant's `qdrant_storage` Docker volume is your index; back it up, and keep the raw source documents too so you can always re-ingest from scratch.

---

## Where to Go Next

- Swap in **hybrid search** (vector + keyword/BM25) once you notice pure-vector search missing exact-term queries (e.g., a specific dollar figure or clause number) — Qdrant supports this natively.
- Add **re-ranking**: retrieve top 20 with vector search, re-rank down to top 5 with a cross-encoder or Claude itself, for better precision on ambiguous queries.
- Try **query rewriting**: for multi-turn conversations, have Claude rewrite a follow-up question into a standalone query before embedding it, since "what about the second option?" embeds poorly on its own.

This gives you a working system end-to-end that you can point at real Tienta material immediately, plus the eval habit (§9b) that separates a RAG demo from a RAG system you can trust.
