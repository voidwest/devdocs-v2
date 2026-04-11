import os
from typing import Any, cast

import chromadb
import config
import pypdf
from chromadb.utils import embedding_functions


def get_text(path):
    docs = []
    reader = pypdf.PdfReader(path)

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            docs.append({"text": text, "metadata": {"page": i + 1}})
    return docs


def chunk_text(text, chunk_size, overlap):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    return chunks


def run_ingest():
    client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)

    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL_NAME
    )

    collection = client.get_or_create_collection(
        name="devdocs", embedding_function=cast(Any, emb_fn)
    )

    for filename in os.listdir(config.DATA_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(config.DATA_DIR, filename)

            pages = get_text(file_path)

            for page in pages:
                chunks = chunk_text(
                    page["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP
                )
                ids = [
                    f"{filename}_{page['metadata']['page']}_{i}"
                    for i in range(len(chunks))
                ]
                collection.add(
                    documents=chunks,
                    ids=ids,
                    metadatas=[page["metadata"]] * len(chunks),
                )


if __name__ == "__main__":
    run_ingest()
