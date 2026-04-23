import logging
from typing import Any

import chromadb
import httpx
from config import get_settings
from embedding import get_embedding_function

logger = logging.getLogger(__name__)
_httpx_client: httpx.AsyncClient | None = None


def get_httpx_client() -> httpx.AsyncClient:
    global _httpx_client
    if _httpx_client is None:
        _httpx_client = httpx.AsyncClient(timeout=None)
    return _httpx_client


async def close_httpx_client():
    global _httpx_client
    if _httpx_client is not None:
        await _httpx_client.aclose()
        _httpx_client = None


def trim_context(docs: list[str], max_chars: int) -> list[str]:
    trimmed = []
    total = 0

    for d in docs:
        if total + len(d) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                trimmed.append(d[:remaining])
                total += remaining
                logger.info("context trimmed at %d chars (limit %d)", total, max_chars)
                break
        trimmed.append(d)
        total += len(d)

    return trimmed


async def get_context(query: str, n_result: int | None = None) -> tuple[str, list[str]]:

    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.vector_db_path)
    emb_fn = get_embedding_function()
    collection = client.get_or_create_collection(
        name="devdocs", embedding_function=emb_fn
    )

    results = collection.query(
        query_texts=[query], n_results=n_result or settings.top_k
    )

    docs_batch = results.get("documents") or []
    metas_batch = results.get("metadatas") or []

    if not docs_batch or not docs_batch[0]:
        return "", []

    docs = docs_batch[0]
    metas = metas_batch[0] if metas_batch and metas_batch[0] else []
    docs = trim_context(docs, settings.max_prompt_chars)

    sources: dict[str, None] = {}
    for m in metas:
        key = f"{m.get('source', 'unknown')}#pages={m.get('page', '?')}"
        sources[key] = None

    return "\n---\n".join(docs), list(sources.keys())


SYSTEM_PROMPT = (
    "You are a research assistant. Use the provided Context to answer the question.\n"
    "Rules:\n"
    "1. ONLY use the provided Context.\n"
    "2. If the answer is not in the Context, say 'I do not have that information.'\n"
    "3. Do not use outside knowledge."
)


def build_prompt(query: str, context: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\ncontext:\n{context}\n\n"
        f"question: {query}<|end|>\n"
        f"<|assistant|>\n"
    )


async def ask_llm(prompt: str, retries: int = 3, backoff: float = 2.0) -> str:
    settings = get_settings()
    llm_info = {
        "model": settings.llm_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": settings.temperature},
    }

    url = f"{settings.llm_base_url}/api/generate"
    client = get_httpx_client()

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = await client.post(
                url, json=llm_info, timeout=settings.request_timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            last_error = e
            logger.warning("llm request %d/%d failed: %s", attempt, retries, e)
            if attempt < retries:
                import asyncio

                await asyncio.sleep(backoff**attempt)

    logger.error("llm failed after %d retries %s", retries, last_error)
    return f"[llm failed after {retries} retries] {last_error}"


async def query_docs(user_query: str) -> dict[str, Any]:
    context, sources = await get_context(user_query)
    prompt = build_prompt(user_query, context)
    answer = await ask_llm(prompt)

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    test_query = "what's this doc about?"
    print(f"User: {test_query}")
    print(f"AI: {query_docs(test_query)}")
