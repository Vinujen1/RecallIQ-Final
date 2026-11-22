"""Application settings loaded from environment variables (or a local .env file)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RECALLIQ_", extra="ignore")

    app_name: str = "RecallIQ API"
    app_version: str = "0.1.0"
    environment: str = "development"
    # Comma-separated list of origins allowed to call the API from a browser.
    cors_origins: str = "http://localhost:3000"
    # SQLite database produced by the ml/ data pipeline (opened read-only by the API).
    database_path: Path = REPO_ROOT / "data" / "processed" / "recalliq.db"
    # Precomputed MiniLM embeddings built by `python -m retrieval.embed` (git-ignored).
    embeddings_dir: Path = REPO_ROOT / "ml" / "artifacts" / "embeddings" / "all-MiniLM-L6-v2"
    # Fine-tuned models (git-ignored artifacts): checkpoint dir + index built with it
    # (`python -m retrieval.finetuned`). Missing files -> that model answers 503.
    model_b_checkpoint: Path = REPO_ROOT / "ml" / "artifacts" / "models" / "model-b-v1" / "checkpoints" / "best"
    model_b_embeddings_dir: Path = REPO_ROOT / "ml" / "artifacts" / "embeddings" / "model-b-v1"
    model_c_checkpoint: Path = REPO_ROOT / "ml" / "artifacts" / "models" / "model-c-v1" / "checkpoints" / "best"
    model_c_embeddings_dir: Path = REPO_ROOT / "ml" / "artifacts" / "embeddings" / "model-c-v1"

    @property
    def finetuned_models(self) -> dict[str, tuple[Path, Path]]:
        return {"minilm_finetuned": (self.model_b_checkpoint, self.model_b_embeddings_dir),
                "minilm_hard_negative": (self.model_c_checkpoint, self.model_c_embeddings_dir)}

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
