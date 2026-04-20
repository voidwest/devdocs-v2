# devdocs-v2

local RAG pipeline for querying PDF documents. drop in a PDF, ask questions, get grounded answers with source references. no external APIs, runs entirely on your machine.

## features

- PDF ingestion with semantic chunking
- embedding-based vector retrieval (ChromaDB)
- context trimming to prevent LLM overflow
- source tracking per answer
- retry logic with exponential backoff
- fully local inference via Ollama

## tech stack

- python 3.11
- ollama (phi3:mini)
- sentence-transformers (all-MiniLM-L6-v2)
- chromadb
- fastapi

## getting started

### prerequisites

- docker + docker compose
- 8GB RAM minimum

### setup

drop your PDFs into `app/data/docs/`, then:

```bash
docker compose up --build
```

the app indexes your docs automatically on first boot.

### query

```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"prompt": "what is disaggregation?"}'
```

response:
```json
{
  "answer": "...",
  "sources": ["file.pdf#page=2"]
}
```

## architecture
```text
        ┌──────────────┐
        │   PDF Docs   │
        └──────┬───────┘
               │
     ┌─────────▼─────────┐
     │ Semantic Chunking │
     └─────────┬─────────┘
               │
     ┌─────────▼─────────┐
     │ Embeddings Model  │
     └─────────┬─────────┘
               │
        ┌──────▼──────┐
        │  ChromaDB   │
        └──────┬──────┘
               │
        ┌──────▼────────────┐
        │ Vector Retrieval   │
        └──────┬────────────┘
               │
        ┌──────▼──────┐
        │  Reranker   │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │ Context Trim │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │     LLM     │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  Answer +   │
        │  Sources    │
        └─────────────┘
```


## design decisions

### why sentence-transformers over ollama embeddings?
latency. ollama embeddings require a network call per chunk, slowing ingestion. `all-MiniLM-L6-v2` runs in-process and is roughly 3x faster while keeping embedding logic decoupled from generation hardware.

### why character-based chunking?
word-based chunking produces unstable segment sizes and breaks semantic units mid-sentence. character-based sliding window with overlap produces more uniform blocks, improving embedding consistency and retrieval predictability.

### why context trimming?
without a hard limit, retrieved chunks can exceed the model's context window or dilute signal quality. trimming ensures deterministic prompt size and stable model behavior.

### why chromadb?
simplicity. unlike faiss, chromadb stores vectors and source text together. fully local, zero config, easier to manage than cloud-based alternatives at this scale.

### why phi-3?
best quality/speed tradeoff for CPU inference. handles instruction following and grounded Q&A better than models in its size class.

## known limitations

- retrieval is vector-only (no BM25 hybrid search)
- no reranking stage — vector similarity used directly
- context trimming is length-based, not relevance-based
- chunking is not structure-aware (ignores headings/sections)
- no query rewriting for ambiguous inputs
- ingestion is coupled to app startup (slow on first run)

## roadmap

- [ ] reranker (cross-encoder stage)
- [ ] hybrid retrieval (BM25 + vector)
- [ ] query rewriting
- [ ] streamlit frontend
- [ ] support for .txt and .docx
- [ ] decouple ingestion from app lifecycle

## license

MIT
