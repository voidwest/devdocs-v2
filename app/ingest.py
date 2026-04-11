import os

import config
import pypdf


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
