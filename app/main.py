import os

import config
from fastapi import FastAPI
from ingest import run_ingest
from pydantic import BaseModel
from query import query_docs

app = FastAPI(title="DevDocs V2")


class userRequest(BaseModel):
    prompt: str


class aiResponse(BaseModel):
    answer: str
    status: str = "success"


@app.post("/ask", response_model=aiResponse)
async def ask_rag(request: userRequest):
    result = query_docs(request.prompt)
    return {"answer": result}
