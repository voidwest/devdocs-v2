from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = base_dir / "data/docs"
    db_dir: Path = base_dir / "vector_db"
    llm_model: str = "phi3:mini"
    llm_base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    top_k: int = 5
    request_timeout: int = 30

    embedding_model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"

    vector_db_path: str = str(db_dir / "chroma_store")
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_prompt_chars: int = 12000

    class Config:
        env_prefix = ""
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
