import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI
from src.config import get_settings
from src.db.factory import make_database
from src.dependencies import protect_api
from src.routers import agentic_ask, hybrid_search, ping
from src.routers.ask import ask_router, stream_router
from src.services.arxiv.factory import make_arxiv_client
from src.services.cache.factory import make_cache_client
from src.services.embeddings.factory import make_embeddings_service
from src.services.langfuse.factory import make_langfuse_tracer
from src.services.memory import ConversationMemory
from src.services.ollama.factory import make_ollama_client
from src.services.opensearch.factory import make_opensearch_client
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.services.retrieval import Retriever, make_reranker
from src.services.telegram.factory import make_telegram_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _warn_on_index_mismatch(opensearch_client, embeddings_client) -> None:
    """Log loud warnings for index/config mismatches that silently ruin retrieval quality."""
    try:
        index = opensearch_client.index_name
        mapping = opensearch_client.client.indices.get_mapping(index=index)
        embedding = mapping[index]["mappings"]["properties"]["embedding"]
        if embedding.get("method", {}).get("engine") == "nmslib":
            logger.warning(
                f"Index '{index}' uses the deprecated nmslib k-NN engine (cannot be created on OpenSearch 3.x). "
                "Recreate it with setup_indices(force=True) and re-run ingestion."
            )
        sample = opensearch_client.client.search(index=index, body={"size": 1, "_source": ["embedding_model"]})
        hits = sample["hits"]["hits"]
        indexed_model = hits[0]["_source"].get("embedding_model") if hits else None
        current_model = getattr(embeddings_client, "model_name", None)
        if indexed_model and current_model and indexed_model != current_model:
            logger.warning(
                f"Index was built with embeddings '{indexed_model}' but queries use '{current_model}': "
                "vector search results will be meaningless until the index is rebuilt."
            )
    except Exception as e:
        logger.debug(f"Index consistency check skipped: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan for the API.
    """
    logger.info("Starting RAG API...")

    settings = get_settings()
    app.state.settings = settings
    if not settings.api_key_set:
        logger.warning("API_KEYS is empty: the API accepts unauthenticated requests. Set it before exposing the API.")

    database = make_database()
    app.state.database = database
    logger.info("Database connected")

    # Initialize search service
    opensearch_client = make_opensearch_client()
    app.state.opensearch_client = opensearch_client

    # Verify OpenSearch connectivity and create index if needed
    if opensearch_client.health_check():
        logger.info("OpenSearch connected successfully")

        # Setup hybrid index (supports all search types)
        setup_results = opensearch_client.setup_indices(force=False)
        if setup_results.get("hybrid_index"):
            logger.info("Hybrid index created")
        else:
            logger.info("Hybrid index already exists")

        # Get simple statistics
        try:
            stats = opensearch_client.client.count(index=opensearch_client.index_name)
            logger.info(f"OpenSearch ready: {stats['count']} documents indexed")
        except Exception:
            logger.info("OpenSearch index ready (stats unavailable)")
    else:
        logger.warning("OpenSearch connection failed - search features will be limited")

    # Initialize other services (kept for future endpoints and notebook demos)
    app.state.arxiv_client = make_arxiv_client()
    app.state.pdf_parser = make_pdf_parser_service()
    app.state.embeddings_service = make_embeddings_service()
    app.state.ollama_client = make_ollama_client()
    app.state.langfuse_tracer = make_langfuse_tracer()
    app.state.cache_client = make_cache_client(settings)  # None if Redis is unavailable
    app.state.reranker = make_reranker(settings)
    app.state.retriever = Retriever(
        opensearch_client,
        app.state.embeddings_service,
        reranker=app.state.reranker,
        rerank_candidates=settings.reranker.candidates,
    )
    if settings.conversation.enabled:
        app.state.conversation_memory = ConversationMemory(
            redis_client=getattr(app.state.cache_client, "redis", None),
            max_turns=settings.conversation.max_turns,
            ttl_seconds=settings.conversation.ttl_hours * 3600,
        )
    logger.info(
        "Services initialized: arXiv API client, PDF parser, OpenSearch, Embeddings "
        f"({settings.embeddings.provider}), Ollama, Langfuse, Cache, Reranker "
        f"({'on: ' + app.state.reranker.model_name if app.state.reranker else 'off'})"
    )
    _warn_on_index_mismatch(opensearch_client, app.state.embeddings_service)

    # Initialize Telegram bot (Week 7)
    telegram_service = make_telegram_service(
        opensearch_client=app.state.opensearch_client,
        embeddings_client=app.state.embeddings_service,
        ollama_client=app.state.ollama_client,
        cache_client=app.state.cache_client,
        langfuse_tracer=app.state.langfuse_tracer,
        retriever=app.state.retriever,
    )

    if telegram_service:
        try:
            # Only one uvicorn worker wins the polling lock; the others skip the bot.
            if await telegram_service.start():
                app.state.telegram_service = telegram_service
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
    else:
        logger.info("Telegram bot not configured - skipping initialization")

    logger.info("API ready")
    yield

    # Cleanup
    if hasattr(app.state, "telegram_service") and app.state.telegram_service:
        await app.state.telegram_service.stop()
        logger.info("Telegram bot stopped")

    database.teardown()
    logger.info("API shutdown complete")


app = FastAPI(
    title="arXiv Paper Curator API",
    description="Personal arXiv CS.AI paper curator with RAG capabilities",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
)

# Include routers. Everything except the health probe goes through API-key auth and
# rate limiting (both disabled unless API_KEYS / RATE_LIMIT_PER_MINUTE are set).
protected = [Depends(protect_api)]
app.include_router(ping.router, prefix="/api/v1")  # Health check endpoint (unauthenticated for probes)
app.include_router(hybrid_search.router, prefix="/api/v1", dependencies=protected)  # Search chunks with BM25/hybrid
app.include_router(ask_router, prefix="/api/v1", dependencies=protected)  # RAG question answering with LLM
app.include_router(stream_router, prefix="/api/v1", dependencies=protected)  # Streaming RAG responses
app.include_router(agentic_ask.router, dependencies=protected)  # Agentic RAG with intelligent retrieval


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
