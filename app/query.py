from typing import Any, cast

import chromadb
import config
import requests
from chromadb.utils import embedding_functions


def get_context(query, n_result=3):
    client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL_NAME
    )
    collection = client.get_collection(
        name="devdocs", embedding_function=cast(Any, emb_fn)
    )

    results = collection.query(query_texts=[query], n_results=n_result)

    return "\n---\n".join(results["documents"][0])


SYSTEM_PROMPT = """you are a helpful assistant. use the provided context to answer the question.
If the answer isn't in the context, say you don't know."""


def build_prompt(query, context):
    return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Context:
{context}

Question: {query}<|im_end|>
<|im_start|>assistant
"""


def ask_llm(prompt):
    llm_info = {
        "model": config.LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": config.TEMPERATURE},
    }
    llm_response = requests.post(f"{config.LLM_BASE_URL}/api/generate", json=llm_info)

    return llm_response.json().get("llm_response")
