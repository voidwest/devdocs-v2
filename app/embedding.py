import threading

from chromadb.utils import embedding_functions
from config import get_settings

_lock = threading.Lock()
_embedding_fn = None


def get_embedding_function():
    global _embedding_fn
    if _embedding_fn is not None:
        return _embedding_fn
    with _lock:
        if _embedding_fn is not None:
            return _embedding_fn
        settings = get_settings()
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model_name
        )
        return _embedding_fn
