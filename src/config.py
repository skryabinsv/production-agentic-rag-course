import os
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class BaseConfigSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )


class ArxivSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="ARXIV__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    base_url: str = "https://export.arxiv.org/api/query"
    pdf_cache_dir: str = "./data/arxiv_pdfs"
    rate_limit_delay: float = 3.0
    timeout_seconds: int = 30
    max_results: int = 15
    search_category: str = "cs.AI"
    download_max_retries: int = 3
    download_retry_delay_base: float = 5.0
    max_concurrent_downloads: int = 5
    max_concurrent_parsing: int = 1

    namespaces: dict = {
        "atom": "http://www.w3.org/2005/Atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    @field_validator("pdf_cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: str) -> str:
        os.makedirs(v, exist_ok=True)
        return v


class PDFParserSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="PDF_PARSER__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    max_pages: int = 30
    max_file_size_mb: int = 20
    do_ocr: bool = False
    do_table_structure: bool = True


class ChunkingSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="CHUNKING__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    chunk_size: int = 600  # Target words per chunk
    overlap_size: int = 100  # Words to overlap between chunks
    min_chunk_size: int = 100  # Minimum words for a valid chunk
    section_based: bool = True  # Use section-based chunking when available
    # Legacy behaviour prefixed *every* section chunk with the full abstract, which makes all
    # chunks of a paper embed alike. Default: title-only header + one dedicated abstract chunk.
    include_abstract_in_chunks: bool = False


class OpenSearchSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="OPENSEARCH__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "http://localhost:9200"
    index_name: str = "arxiv-papers"
    chunk_index_suffix: str = "chunks"  # Creates single hybrid index: {index_name}-{suffix}
    max_text_size: int = 1000000

    # Vector search settings
    vector_dimension: int = 1024  # Jina embeddings dimension
    vector_space_type: str = "cosinesimil"  # cosinesimil, l2, innerproduct

    # Hybrid search settings
    rrf_pipeline_name: str = "hybrid-rrf-pipeline"
    hybrid_search_size_multiplier: int = 2  # Get k*multiplier for better recall

    # Optional authentication / TLS for a secured cluster (the bundled dev cluster runs
    # with the security plugin disabled and is only reachable on 127.0.0.1)
    username: str = ""
    password: str = ""
    use_ssl: bool = False
    verify_certs: bool = True


class EmbeddingsSettings(BaseConfigSettings):
    """Embedding provider. ``jina`` = hosted API (texts leave the machine), ``local`` =
    sentence-transformers model running in-process (fully private, needs RAM/CPU)."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="EMBEDDINGS__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    provider: Literal["jina", "local"] = "jina"
    # 1024-dim English model (MIT) - same dimension as the default index mapping
    local_model: str = "BAAI/bge-large-en-v1.5"
    local_query_prefix: str = "Represent this sentence for searching relevant passages: "
    local_passage_prefix: str = ""
    local_device: str = ""  # "", "cpu", "cuda", "mps"; empty = auto
    local_batch_size: int = 16


class RerankerSettings(BaseConfigSettings):
    """Second-stage reranking of retrieved chunks (cross-encoder)."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="RERANKER__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    enabled: bool = False
    provider: Literal["jina", "local"] = "local"
    # Empty = provider default (jina-reranker-v2-base-multilingual / cross-encoder/ms-marco-MiniLM-L-6-v2)
    model: str = ""
    candidates: int = 20  # chunks fetched from OpenSearch before reranking down to top_k
    timeout_seconds: float = 15.0


class ConversationSettings(BaseConfigSettings):
    """Multi-turn memory for the agentic endpoint (stored in Redis, per-process fallback)."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="CONVERSATION__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    enabled: bool = True
    max_turns: int = 5
    ttl_hours: int = 24


def _langfuse_env(name: str) -> AliasChoices:
    """Accept both ``LANGFUSE__<NAME>`` (project convention) and ``LANGFUSE_<NAME>`` (official SDK).

    ``.env.example`` documents the single-underscore SDK names, while this settings class
    only read the double-underscore prefix, so Langfuse silently stayed disabled.
    The field name itself stays valid so the parent ``Settings`` can still populate it.
    """
    return AliasChoices(name, f"LANGFUSE__{name.upper()}", f"LANGFUSE_{name.upper()}")


class LangfuseSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="LANGFUSE__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
        populate_by_name=True,
    )

    public_key: str = Field(default="", validation_alias=_langfuse_env("public_key"))
    secret_key: str = Field(default="", validation_alias=_langfuse_env("secret_key"))
    host: str = Field(default="http://localhost:3001", validation_alias=_langfuse_env("host"))  # compose maps 3001 -> 3000
    enabled: bool = Field(default=True, validation_alias=_langfuse_env("enabled"))
    flush_at: int = Field(default=15, validation_alias=_langfuse_env("flush_at"))  # Number of events before flushing
    flush_interval: float = Field(default=1.0, validation_alias=_langfuse_env("flush_interval"))  # Seconds between flushes
    max_retries: int = 3
    timeout: int = 30
    debug: bool = Field(default=False, validation_alias=_langfuse_env("debug"))


class RedisSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="REDIS__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    decode_responses: bool = True
    socket_timeout: int = 30
    socket_connect_timeout: int = 30

    # Cache settings
    ttl_hours: int = 6  # Cache TTL in hours


class TelegramSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="TELEGRAM__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    bot_token: str = ""
    enabled: bool = False
    # Comma-separated Telegram user ids allowed to use the bot (every message costs LLM time
    # and embedding-API quota).
    allowed_user_ids: str = ""
    # With an empty allow-list the bot stays private and tells users their id;
    # set to true only if you really want a public bot.
    allow_public: bool = False

    @property
    def allowed_user_id_set(self) -> set[int]:
        return {int(x) for x in self.allowed_user_ids.replace(" ", "").split(",") if x}


class Settings(BaseConfigSettings):
    app_version: str = "0.1.0"
    debug: bool = True
    environment: Literal["development", "staging", "production"] = "development"
    service_name: str = "rag-api"

    postgres_database_url: str = "postgresql://rag_user:rag_password@localhost:5432/rag_db"
    postgres_echo_sql: bool = False
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 0

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    ollama_timeout: int = 300
    # Context window passed to Ollama. Its default is small; 3 chunks of 600-800 words
    # plus the system prompt would otherwise be truncated without any error.
    ollama_num_ctx: int = 8192
    # Low temperature for grounded answers (was 0.7 and, in fact, never applied).
    ollama_temperature: float = 0.2
    # Comma-separated allow-list of models clients may request (ollama_model is always allowed).
    # Default keeps the course notebooks working (they use llama3.2:1b and llama3.2:3b).
    ollama_allowed_models: str = "llama3.2:3b"

    @property
    def allowed_models(self) -> set[str]:
        models = {m.strip() for m in self.ollama_allowed_models.split(",") if m.strip()}
        models.add(self.ollama_model)
        return models

    # Jina AI embeddings configuration
    jina_api_key: str = ""

    # API protection. Comma-separated keys accepted in `X-API-Key` / `Authorization: Bearer`.
    # Empty = no authentication (fine on 127.0.0.1, never on a reachable host).
    api_keys: str = ""
    # Requests per minute per client (API key or IP) on /api/v1 endpoints; 0 = unlimited.
    rate_limit_per_minute: int = 0

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    pdf_parser: PDFParserSettings = Field(default_factory=PDFParserSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)

    @field_validator("postgres_database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not (v.startswith("postgresql://") or v.startswith("postgresql+psycopg2://")):
            raise ValueError("Database URL must start with 'postgresql://' or 'postgresql+psycopg2://'")
        return v


def get_settings() -> Settings:
    return Settings()
