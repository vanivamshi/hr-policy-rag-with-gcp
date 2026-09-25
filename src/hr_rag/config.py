from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, read from environment variables (or a local .env)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Google Cloud ---
    gcp_project: str = ""
    gcp_location: str = "us-central1"
    gcs_bucket: str = ""
    gcs_prefix: str = "hr-policies/"

    # --- Qdrant Cloud ---
    qdrant_url: str = ""
    qdrant_api_key: str | None = None
    qdrant_collection: str = "hr_policies"
    cache_collection: str = "hr_semantic_cache"

    # --- Jina AI (embeddings + reranker) ---
    jina_api_key: str = ""
    jina_embedding_model: str = "jina-embeddings-v3"
    jina_embedding_dim: int = 1024
    jina_reranker_model: str = "jina-reranker-v2-base-multilingual"

    # --- Sparse BM25 (fastembed) ---
    bm25_model: str = "Qdrant/bm25"
    fastembed_cache_dir: str | None = None

    # --- LiteLLM router ---
    primary_model: str = "vertex_ai/gemini-2.5-flash"
    fallback_model: str = "groq/llama-3.3-70b-versatile"
    groq_api_key: str | None = None
    llm_temperature: float = 0.1
    llm_timeout_seconds: float = 60.0

    # --- Model Armor guardrails ---
    model_armor_enabled: bool = True
    model_armor_location: str = "us-central1"
    model_armor_template: str = "hr-rag-template"
    guardrail_fail_open: bool = False

    # --- Semantic cache ---
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 7 * 24 * 3600

    # --- Retrieval ---
    retrieval_prefetch_k: int = 30
    rerank_top_n: int = 5
    rerank_min_score: float = 0.15

    # --- Ingestion ---
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # --- Agent ---
    agent_max_steps: int = 4
    agent_max_history_messages: int = 6

    # --- Access control ---
    allowed_emails: str = ""  # comma-separated, e.g. "a@corp.com,b@corp.com"
    allowed_domains: str = ""  # comma-separated, e.g. "corp.com"
    auth_disabled: bool = False  # local development only, never on Cloud Run


@lru_cache
def get_settings() -> Settings:
    return Settings()
