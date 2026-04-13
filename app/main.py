import os

import config
from fastapi import FastAPI
from ingest import run_ingest
from pydantic import BaseModel
from query import query_docs
