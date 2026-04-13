import os

from ingest import run_ingest
from query import query_docs


def main():
    if not os.path.exists("./db_storage"):
        run_ingest()
