import os

import config
from fastapi import FastAPI
from fastapi.routing import asynccontextmanager
from ingest import run_ingest
from pydantic import BaseModel
from query import query_docs


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"init using {config.LLM_BASE_URL}")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.DB_DIR.mkdir(parents=True, exist_ok=True)

    if not (config.DB_DIR.exists() and any(config.DB_DIR.iterdir())):
        print(f"databse is empty, processing from {config.DATA_DIR}")
        run_ingest()
    else:
        print(f"data exists at {config.VECTOR_DB_PATH}")

    yield


app = FastAPI(lifespan=lifespan, title="DevDocs V2")


class userRequest(BaseModel):
    prompt: str


class aiResponse(BaseModel):
    answer: str
    status: str = "success"


@app.post("/ask", response_model=aiResponse)
async def ask_rag(request: userRequest):
    result = query_docs(request.prompt)
    return {"answer": result}
