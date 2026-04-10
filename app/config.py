import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/docs"
DB_DIR = BASE_DIR / "vector_db"

LLM_MODEL = "modeldiscussed"
LLM_BASE_URL = "urlillfindsomehow"
TEMPERATURE = 0.1  #


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = "cpu"

VECTOR_DB_PATH = str(DB_DIR / "chroma_store")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
