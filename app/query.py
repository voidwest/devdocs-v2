import chromadb
import config
import requests
from chromadb.utils import embedding_functions


def get_context(query, n_result=3):
    client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL_NAME
    )
    collection = client.get_collection(name="devdocs", embedding_function=emb_fn)
