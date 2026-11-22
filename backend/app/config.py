"""Application settings loaded from environment variables (or a local .env file)."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag.llm import DEFAULT_MODEL, LLMConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime configuration, from RECALLIQ_* environment variables or backend/.env.

    Paths default to the repository layout, so a fresh checkout runs with no .env at all;
    missing artifacts disable individual models rather than failing startup."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RECALLIQ_", extra="ignore",
                                      populate_by_name=True)

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

    # --- RAG generation (POST /explain) --------------------------------------------------
    # "none" = deterministic template only (default, no network); "openai" = GPT answer that
    # must pass the grounding guards, else the template is shown with a fallback reason.
    llm_provider: str = "none"
    # Never logged or returned. Read from RECALLIQ_OPENAI_API_KEY or the standard OPENAI_API_KEY.
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("RECALLIQ_OPENAI_API_KEY", "OPENAI_API_KEY"))
    openai_model: str = DEFAULT_MODEL
    llm_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    llm_max_output_tokens: int = Field(default=500, ge=64, le=4000)
    llm_max_retries: int = Field(default=1, ge=0, le=5)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    # Results retrieved for /explain when the request omits top_k, and how many of them
    # (after dropping identifier conflicts) are given to the generator as evidence.
    rag_top_k: int = Field(default=5, ge=1, le=5)
    rag_max_evidence: int = Field(default=3, ge=1, le=5)
    # Level of the `recalliq.*` loggers (e.g. RAG accept/reject lines). DEBUG adds guard
    # excerpts of rejected answers, which may echo query content: use only when diagnosing.
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def _log_level(cls, value: str) -> str:
        value = (value or "INFO").strip().upper()
        if value not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("RECALLIQ_LOG_LEVEL must be DEBUG, INFO, WARNING or ERROR")
        return value

    @field_validator("llm_provider")
    @classmethod
    def _provider(cls, value: str) -> str:
        value = (value or "none").strip().lower()
        if value not in {"none", "openai"}:
            raise ValueError('RECALLIQ_LLM_PROVIDER must be "none" or "openai"')
        return value

    @property
    def llm_config(self) -> LLMConfig:
        key = self.openai_api_key.get_secret_value().strip() if self.openai_api_key else ""
        return LLMConfig(provider=self.llm_provider, model=self.openai_model.strip(),
                         api_key=key or None, timeout_s=self.llm_timeout_seconds,
                         max_output_tokens=self.llm_max_output_tokens,
                         max_retries=self.llm_max_retries, temperature=self.llm_temperature)

    @property
    def finetuned_models(self) -> dict[str, tuple[Path, Path]]:
        return {"minilm_finetuned": (self.model_b_checkpoint, self.model_b_embeddings_dir),
                "minilm_hard_negative": (self.model_c_checkpoint, self.model_c_embeddings_dir)}

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Settings are read once per process; tests replace this dependency with overrides."""
    return Settings()
