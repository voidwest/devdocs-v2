import config
import chromadb
from chromadb.utils import embedding_functions
import requests

def get_context(query, n_result=3):
