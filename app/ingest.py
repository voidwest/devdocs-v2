import hashlib
import logging
import os
from typing import Iterable

import chromadb
import config
import pypdf
from chromadb.api.models.Collection import Collection
from embedding import get_embedding_function

logger = logging.getLogger(__name__)


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
        return h.hexdigest()[:16]


def get_text(path: str) -> list[dict]:
    try:
        reader = pypdf.PdfReader(path)
    except Exception as e:
        logger.error("failed to read pdf %s: %s", path, e)
        return []

    docs = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception as e:
            logger.warning("failed to extract page %d from %s: %s", i, path, e)

            continue
        if text and text.strip():
            docs.append({"text": text, "metadata": {"page": i + 1}})
    return docs


def chunk_text_stream(text: Iterable[str], chunk_size: int, overlap: int):
    full_text = "\n".join(text)
    start = 0
    text_len = len(full_text)

    while start < text_len:
        end = start + chunk_size
        yield full_text[start:end]
        if end >= text_len:
            break
        start += chunk_size - overlap


def get_existing_doc_hashes(collection: Collection) -> set[str]:
    result = collection.get(where={"dochash": {"$ne": ""}}, include=["metadatas"])

    metadatas = result.get("metadatas") if result else None
    if not metadatas:
        return set()

    hashes: set[str] = set()
    for m in metadatas:
        val = m.get("dochash")
        if isinstance(val, str):
            hashes.add(val)
    return hashes


def run_ingest():

    settings = config.get_settings()
    os.makedirs(settings.data_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=settings.vector_db_path)
    emb_fn = get_embedding_function()

    try:
        collection = client.get_collection(name="devdocs")
        logger.info("Found existing collection, using persisted embedding function.")
    except Exception:
        collection = client.create_collection(name="devdocs", embedding_function=emb_fn)  # type: ignore
        logger.info("Created new collection with sentence_transformer.")

    pdf_files = {
        f: os.path.join(settings.data_dir, f)
        for f in os.listdir(settings.data_dir)
        if f.endswith(".pdf")
    }

    if not pdf_files:
        logger.warning("No PDFs found in %s", settings.data_dir)
        return

    current_hashes = {file_hash(p) for p in pdf_files.values()}
    existing_hashes = get_existing_doc_hashes(collection)

    stale_hashes = existing_hashes - current_hashes
    if stale_hashes:
        for stale in stale_hashes:
            logger.info("Removing stale documents for hash %s", stale)
            collection.delete(where={"dochash": stale})

    for filename, filepath in pdf_files.items():
        doc_hash = file_hash(filepath)
        if doc_hash in existing_hashes:
            logger.info("skipping %s (already in database)", filename)
            continue

        pages = get_text(filepath)
        if not pages:
            continue

        all_chunks: list[str] = []
        all_metadatas: list[dict] = []

        for p in pages:
            page_num = p["metadata"]["page"]
            page_chunks = list(
                chunk_text_stream([p["text"]], settings.chunk_size, settings.chunk_overlap)
            )
            for chunk_text in page_chunks:
                if not chunk_text.strip():
                    continue
                all_chunks.append(chunk_text)
                all_metadatas.append({
                    "source": filename,
                    "page": str(page_num),
                    "dochash": doc_hash,
                })

        ids = [f"{doc_hash}_{i}" for i in range(len(all_chunks))]
        logger.info("adding %d chunks for %s", len(all_chunks), filename)
        collection.add(
            documents=all_chunks,
            ids=ids,
            metadatas=all_metadatas,  # type: ignore[arg-type]
        )


if __name__ == "__main__":
    run_ingest()
