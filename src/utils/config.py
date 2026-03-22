from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "llama3-8b-8192"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    chroma_db_path: str = "./data/processed/chroma_db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True
    )

settings = Settings()