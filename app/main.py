import asyncio
import logging
from contextlib import asynccontextmanager

import chromadb.errors
from config import get_settings
from fastapi import FastAPI, HTTPException
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
    logger.info("starting service, vector db path is at %s", settings.vector_db_path)

    try:
        client = await get_chroma_client()
        existing = await asyncio.to_thread(
            lambda: {c.name for c in client.list_collections()}
        )

        if "devdocs" not in existing:
            logger.info("collection missing, running ingestion")
            await asyncio.to_thread(run_ingest)
        else:
            collection = await asyncio.to_thread(client.get_collection, "devdocs")
            count = await asyncio.to_thread(collection.count)
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
