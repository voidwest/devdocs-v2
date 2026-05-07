# devdocs-v2

![CI](https://github.com/west/devdocs-v2/actions/workflows/ci.yml/badge.svg)

local RAG pipeline for querying PDF documents. drop in a PDF, ask questions, get grounded answers with source references. no external APIs, runs entirely on your machine.

## features

- PDF ingestion with overlapping character-based chunking (per-page, preserving page numbers for source tracking)
- embedding-based vector retrieval (ChromaDB)
- context trimming to prevent LLM overflow
- source tracking per answer with file and page references
- retry logic with exponential backoff
- fully local inference via Ollama
- web UI with chat-style message history and source references
- incremental indexing (skips already-ingested docs via SHA-256 hashing)
- stale document cleanup (removes chunks for deleted PDFs)

## tech stack

- python 3.11
- ollama (default: phi3:mini, swappable)
- sentence-transformers (all-MiniLM-L6-v2)
- chromadb
- fastapi
- vanilla HTML/CSS/JS (frontend, served by static files)

## getting started

### prerequisites

- docker + docker compose
- 8GB RAM minimum

### setup

drop your PDFs into `app/data/docs/`, then:

```bash
# use sudo if needed
docker compose up --build
```

the app indexes your docs automatically when the vector collection is empty (first boot or after a volume wipe). watch the logs for `adding N chunks` to know it's working.

### usage

**web UI** — open http://localhost:8000 in your browser. type a question, get an answer with source references inline.

**API** — once you see `Application startup complete`, hit the endpoint:

```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"prompt": "what is this doc about?"}'
```

response:
```json
{
  "answer": "...",
  "sources": ["file.pdf#pages=3"]
}
```

**health check:**

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### configuration

set these via environment variables or a `.env` file:

| variable | default | description |
|---|---|---|
| `LLM_BASE_URL` | `http://ollama:11434` | ollama API endpoint |
| `LLM_MODEL` | `phi3:mini` | model name |
| `CHUNK_SIZE` | `512` | characters per chunk |
| `CHUNK_OVERLAP` | `64` | overlap between chunks |
| `TOP_K` | `5` | chunks retrieved per query |
| `TEMPERATURE` | `0.1` | LLM sampling temperature |
| `MAX_PROMPT_CHARS` | `12000` | max chars in prompt context |
| `REQUEST_TIMEOUT` | `300` | LLM request timeout in seconds |

### swapping the model

you can use any model available on ollama. two places need updating:

**1. `docker-compose.yml`** — change the model in the ollama entrypoint:

```yaml
# find this line in the ollama entrypoint:
if ! ollama list | grep -q phi3:mini; then
  ollama pull phi3:mini
fi

# replace phi3:mini with your model, e.g.:
if ! ollama list | grep -q llama3.2; then
  ollama pull llama3.2
fi
```

**2. `.env` or `docker-compose.yml` environment** — set the model name for the rag-app:

```bash
# .env
LLM_MODEL=llama3.2
```

or directly in `docker-compose.yml` under `rag-app.environment`:

```yaml
- LLM_MODEL=llama3.2
```

**3. prompt format** — if your model uses a different chat template, update `SYSTEM_PROMPT` and `build_prompt()` in `app/query.py`. phi-3 uses `<|system|> / <|user|> / <|assistant|>` tags. llama3 uses a different format. check your model's documentation.

restart after changes:

```bash
docker compose down
docker compose up --build
```

### troubleshooting

- **no pdfs found**: check your volume paths in `docker-compose.yml`.
- **timeout / 500 error**: if you're on a slow CPU, increase `REQUEST_TIMEOUT`.
- **permissions**: if the container can't read your files, run `chmod -R 755 app/data/docs`.
- **api unreachable in frontend**: ensure the backend started successfully. check `docker compose logs rag-app`.
- **chromadb keyerror / db errors after upgrading**: you likely have stale vector data from a previous chromadb version. run `docker compose down -v && docker compose up --build` to recreate the volume.
- **ollama model not found**: if the model name changed but the cached volume still has old data, run `docker compose down -v` to clear ollama's cache and re-pull.

## architecture

```text
┌──────────────┐
│   PDF Docs   │
└──────┬───────┘
       │
┌──────▼──────────────┐
│  Per-Page Chunking  │
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│  Embeddings Model   │
└──────┬──────────────┘
       │
┌──────▼──────┐
│  ChromaDB   │
└──────┬──────┘
       │
┌──────▼──────────────┐
│  Vector Retrieval   │
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│   Context Trim      │
└──────┬──────────────┘
       │
┌──────▼──────┐
│     LLM     │
└──────┬──────┘
       │
┌──────▼──────────┐
│  Answer+Sources │
└─────────────────┘
```

## design decisions

### why sentence-transformers over ollama embeddings?

latency and reliability. ollama embeddings require an HTTP call per chunk, slowing ingestion. `all-MiniLM-L6-v2` runs in-process via a thread-safe singleton and is roughly 3x faster. it also avoids a circular dependency: if the ollama container hadn't pulled the model yet, the embedding step could deadlock startup. decoupling embeddings from generation hardware removes that risk entirely.

### why character-based chunking?

predictability. word-based chunking produces unstable segment sizes and often breaks mid-sentence. a character-based sliding window with overlap produces more uniform blocks, which improves embedding consistency. each page is chunked independently so page numbers remain accurate in source metadata.

### why per-page chunking?

source fidelity. the previous approach concatenated all pages then chunked, which destroyed page boundaries — every source reference was `page=multi`. chunking each page separately preserves exact page numbers in metadata, so answers cite the right pages.

### why context trimming?

determinism. without a hard character limit, retrieved chunks can exceed the model's context window or drown the query in noise. trimming enforces a stable prompt size. the current implementation stops adding chunks once the cap is hit and will trim the last partial chunk if it has at least 100 chars of headroom — preventing tiny, useless fragments from polluting the prompt.

### why chromadb?

zero infrastructure. chromadb stores vectors, source text, and metadata in a single sqlite database — no separate service, no network config. fully local, fully persistent, and simpler to manage than faiss or pinecone at this scale. the collection is loaded via a global singleton with async locking so multiple concurrent requests share one client safely.

### why incremental indexing via SHA-256 hashing?

startup speed and data integrity. every PDF is hashed on disk and compared against existing chunk metadata before indexing. unchanged files are skipped. files that have been removed from `data/docs/` are automatically cleaned up — their chunks are deleted from chromadb by hash. this means restarting the container is fast after the first boot.

### why phi-3 as the default?

best quality/speed tradeoff for CPU inference. phi-3 mini handles instruction-following and grounded Q&A better than models in its 3.8B size class. the system prompt and chat template are tuned for phi-3's `<|system|>` / `<|user|>` / `<|assistant|>` format. swapping models requires updating the prompt format in `query.py` (see model swapping guide above).

### why vanilla HTML/CSS/JS for the frontend?

no build step, no dependency hell. a single static HTML file served by fastapi's `StaticFiles` mount. the entire frontend is under 300 lines — dark theme, chat bubbles, source tags, markdown rendering, and health polling (15s interval with visual connection indicator). no npm, no webpack, no python dependency. it loads instantly and works on any browser.

### why a unified container (backend + frontend)?

simplicity. the frontend is served by fastapi from within the same container, reducing the compose setup from 3 services to 2. no separate frontend build, no inter-container networking for the UI, and one less port to expose. the only split is rag-app + ollama, which is a genuine separation of concerns (inference hardware may differ).

## known limitations

- retrieval is vector-only (no BM25 hybrid search)
- no reranking stage — vector similarity used directly
- context trimming is length-based, not relevance-based
- chunking is not structure-aware (ignores headings/sections)
- no query rewriting for ambiguous inputs
- ingestion is coupled to app startup (slow on first run)

## development

### running tests

```bash
PYTHONPATH=app python3 -m pytest tests/ -v
```

a CI pipeline runs tests on every push and PR via GitHub Actions.

### project structure

```
.
├── app/                 # fastapi backend + static frontend
│   ├── main.py          # entry point, /ask, /health, serves /
│   ├── config.py        # pydantic-settings configuration
│   ├── ingest.py        # PDF parsing, chunking, chromadb indexing
│   ├── embedding.py     # sentence-transformers singleton
│   ├── query.py         # retrieval, prompt building, LLM call
│   ├── static/          # web UI (vanilla HTML/CSS/JS)
│   └── data/docs/       # drop PDFs here
├── tests/               # unit tests (chunking, config, query)
├── .dockerignore        # docker build exclusion rules
├── .github/
│   └── workflows/
│       └── ci.yml       # GitHub Actions CI (pytest on push/PR)
├── docker-compose.yml   # 2 services: rag-app, ollama
├── Dockerfile           # multi-stage python build
└── requirements.txt     # pinned python dependencies
```

## roadmap

- [ ] reranker (cross-encoder stage)
- [ ] hybrid retrieval (BM25 + vector)
- [ ] query rewriting
- [x] web UI (vanilla HTML/CSS/JS, served by fastapi)
- [ ] support for .txt and .docx
- [ ] decouple ingestion from app lifecycle

## license

MIT
