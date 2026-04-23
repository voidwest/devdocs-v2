import logging
from contextlib import asynccontextmanager

from config import get_settings
from fastapi import FastAPI, HTTPException
from ingest import run_ingest
from pydantic import BaseModel, Field
from query import close_httpx_client, query_docs


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("init")

    import chromadb

    client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)

    try:
        collection = client.get_collection(name="devdocs")
        count = collection.count()
    except:
        count = 0

    if count == 0:
        print(f"empty or missing collection. ingesting from {config.DATA_DIR}")
        run_ingest()
    else:
        print(f"found {count} chunks in 'devdocs' collection.")

    yield


app = FastAPI(lifespan=lifespan, title="DevDocs V2")


class userRequest(BaseModel):
    prompt: str


class aiResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ask", response_model=aiResponse)
async def ask_rag(request: userRequest):
    return query_docs(request.prompt)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
