import time
from typing import Any, cast

import chromadb
import config
import requests
from embedding import get_embedding_function

client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
emb_fn = get_embedding_function()


def get_context(query, n_result=config.TOP_K):
    collection = client.get_or_create_collection(
        name="devdocs", embedding_function=cast(Any, emb_fn)
    )

    results = collection.query(query_texts=[query], n_results=n_result)

    if not results:
        return "", []

    docs_batch = results.get("documents") or []
    metas_batch = results.get("metadatas") or []

    if not docs_batch or not docs_batch[0]:
        return "", []

    docs = docs_batch[0]
    metas = metas_batch[0] if metas_batch and metas_batch[0] else []

    sources = {f"{m.get('source', 'unknown')}#page={m.get('page', '?')}" for m in metas}

    return "\n---\n".join(docs), list(sources)


SYSTEM_PROMPT = """You are a focused assistant.
Use ONLY the following context from the book to answer the question. If the answer isn't there, say you don't know."""


def build_prompt(query, context):
    return f"""<|system|>
    You are a research assistant. Use the provided Context to answer the question.
    Rules:
    1. ONLY use the provided Context.
    2. If the answer is not in the Context, say "I do not have that information."
    3. Do not use outside knowledge.<|end|>
    <|user|>
    Context:
    {context}

    Question: {query}<|end|>
    <|assistant|>
    """


def ask_llm(prompt, retries=3, backoff=2):
    llm_info = {
        "model": config.LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": config.TEMPERATURE},
    }

    url = f"{config.LLM_BASE_URL}/api/generate"

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                url,
                json=llm_info,
                timeout=config.REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            return response.json().get("response", "")

        except requests.RequestException as e:
            if attempt == retries:
                return f"[LLM failed after {retries} retries] {str(e)}"

            time.sleep(backoff**attempt)


def query_docs(user_query: str):
    context, sources = get_context(user_query)
    prompt = build_prompt(user_query, context)
    answer = ask_llm(prompt)

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    test_query = "what's this doc about?"
    print(f"User: {test_query}")
    print(f"AI: {query_docs(test_query)}")
