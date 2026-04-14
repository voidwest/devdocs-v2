from typing import Any, cast

import chromadb
import config
import requests
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=config.EMBEDDING_MODEL_NAME
)


def get_context(query, n_result=3):
    collection = client.get_or_create_collection(
        name="devdocs", embedding_function=cast(Any, emb_fn)
    )

    results = collection.query(query_texts=[query], n_results=n_result)

    if not results["documents"] or not results["documents"][0]:
        return ""

    return "\n---\n".join(results["documents"][0])


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


def ask_llm(prompt):
    llm_info = {
        "model": config.LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": config.TEMPERATURE},
    }
    llm_response = requests.post(f"{config.LLM_BASE_URL}/api/generate", json=llm_info)

    return llm_response.json().get("response")


def query_docs(user_query: str):
    context = get_context(user_query)
    prompt = build_prompt(user_query, context)
    answer = ask_llm(prompt)

    return answer


if __name__ == "__main__":
    test_query = "what's this doc about?"
    print(f"User: {test_query}")
    print(f"AI: {query_docs(test_query)}")
