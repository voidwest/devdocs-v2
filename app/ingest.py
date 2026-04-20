import hashlib
import os
from typing import Any, cast

import chromadb
import config
import pypdf
from embedding import get_embedding_function


def get_text(path):
    docs = []
    reader = pypdf.PdfReader(path)

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            docs.append({"text": text, "metadata": {"page": i + 1}})
    return docs


def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def run_ingest():
    client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)

    emb_fn = get_embedding_function()
    
    collection = client.get_or_create_collection(
        name="devdocs", embedding_function=cast(Any, emb_fn)
    )

    existing_ids = set(collection.get().get("ids", []))

    for filename in os.listdir(config.DATA_DIR):
        if not filename.endswith(".pdf"):
            continue

        file_path = os.path.join(config.DATA_DIR, filename)
        filehash = file_hash(file_path)

        pages = get_text(file_path)

        for page in pages:
            chunks = chunk_text(page["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)

            ids = [
                f"{filehash}_{page['metadata']['page']}_{i}" for i in range(len(chunks))
            ]

            new_docs = []
            new_ids = []
            new_meta = []

            for chunk, id_ in zip(chunks, ids):
                if id_ in existing_ids:
                    continue
                new_docs.append(chunk)
                new_ids.append(id_)
                new_meta.append({"source": filename, "page": page["metadata"]["page"]})
            if new_docs:
                collection.add(
                    documents=new_docs,
                    ids=new_ids,
                    metadatas=new_meta,
                )


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if __name__ == "__main__":
    run_ingest()
