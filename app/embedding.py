import config
from chromadb.utils import embedding_functions

_embedding_fn = None


def get_embedding_function():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL_NAME
        )
    return _embedding_fn
