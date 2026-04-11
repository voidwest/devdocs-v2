import os

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
            docs.append({"text:": text})
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

    collection = client.get
