import os

import config
from ingest import run_ingest
from query import query_docs


def main():
    if not os.path.exists(config.VECTOR_DB_PATH):
        run_ingest()

    print("test msg")
    while True:
        user_input = input("msg: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = query_docs(user_input)
        print(f"\nAI: {response}\n")
