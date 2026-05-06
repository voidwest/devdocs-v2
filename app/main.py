import asyncio
import logging
import os
from contextlib import asynccontextmanager

import chromadb.errors
from config import get_settings
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from ingest import run_ingest
from pydantic import BaseModel, Field
from query import close_httpx_client, get_chroma_client, query_docs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    os.makedirs(settings.vector_db_path, exist_ok=True)
    logger.info("starting service, vector db path is at %s", settings.vector_db_path)

    try:
        client = await get_chroma_client()
        collection = await asyncio.to_thread(client.get_or_create_collection, "devdocs")
        count = await asyncio.to_thread(collection.count)

        if count == 0:
            logger.info("collection empty, running ingestion")
            await asyncio.to_thread(run_ingest)
        else:
            logger.info("found collection with %d chunks", count)

    except asyncio.CancelledError:
        raise
    except (OSError, chromadb.errors.ChromaError) as e:
        logger.exception("vector store initialization failed")
        raise RuntimeError("failed to init vector store") from e
    except Exception:
        logger.exception("unexpected error during startup")
        raise

    yield

    logger.info("shutting down")
    await close_httpx_client()


app = FastAPI(lifespan=lifespan, title="DevDocs V2")

app.mount("/static", StaticFiles(directory="static", html=True), name="static")


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


class UserRequest(BaseModel):
    prompt: str = Field(
        ..., min_length=1, max_length=4000, description="user query for the llm"
    )


class AIResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ask", response_model=AIResponse)
async def ask_rag(request: UserRequest):
    try:
        return await query_docs(request.prompt)
    except Exception as e:
        logger.exception("query failed")
        raise HTTPException(status_code=500, detail="internal query error") from e
