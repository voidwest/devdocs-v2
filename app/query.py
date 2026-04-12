import chromadb
import config
import requests
from chromadb.utils import embedding_functions


def get_context(query, n_result=3):
    client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
