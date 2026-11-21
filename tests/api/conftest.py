from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from src.main import app

SAMPLE_HITS = [
    {
        "arxiv_id": "1706.03762v7",
        "title": "Attention Is All You Need",
        "authors": "Ashish Vaswani, Noam Shazeer",
        "abstract": "The dominant sequence transduction models...",
        "published_date": "2017-06-12",
        "chunk_text": "Transformers rely entirely on self-attention.",
        "section_title": "Introduction",
        "score": 0.9,
    }
]


def _mock_opensearch():
    client = MagicMock()
    client.health_check.return_value = True
    client.setup_indices.return_value = {"hybrid_index": False, "rrf_pipeline": False}
    client.index_name = "arxiv-papers-chunks"
    client.client.count.return_value = {"count": len(SAMPLE_HITS)}
    client.get_index_stats.return_value = {"index_name": "arxiv-papers-chunks", "document_count": len(SAMPLE_HITS)}
    client.search_unified.return_value = {"total": len(SAMPLE_HITS), "hits": [dict(h) for h in SAMPLE_HITS]}
    return client


def _mock_ollama():
    client = MagicMock()
    client.generate_rag_answer = AsyncMock(return_value={"answer": "Transformers use self-attention [arXiv:1706.03762]."})

    async def _stream(**_kwargs):
        yield {"response": "Transformers ", "done": False}
        yield {"response": "use attention.", "done": True}

    client.generate_rag_answer_stream = _stream
    return client


@pytest.fixture
async def client():
    """HTTP client for API testing with mocked services.

    The previous version patched factory modules (e.g. ``src.services.opensearch.factory``)
    while ``src.main`` had already imported the functions by name, so the patches never
    applied and the "API tests" silently required live Postgres/OpenSearch/Redis.
    """
    embeddings = MagicMock()
    embeddings.embed_query = AsyncMock(return_value=[0.1] * 1024)
    tracer = MagicMock()
    tracer.client = None
    tracer.create_span.return_value = None

    from contextlib import contextmanager

    @contextmanager
    def _no_trace(**_kwargs):
        yield None

    tracer.trace_rag_request = _no_trace

    with (
        patch("src.db.interfaces.postgresql.PostgreSQLDatabase.startup") as mock_startup,
        patch("src.db.interfaces.postgresql.PostgreSQLDatabase.get_session") as mock_get_session,
        patch("src.main.make_opensearch_client", return_value=_mock_opensearch()),
        patch("src.main.make_arxiv_client", return_value=MagicMock()),
        patch("src.main.make_pdf_parser_service", return_value=MagicMock()),
        patch("src.main.make_embeddings_service", return_value=embeddings),
        patch("src.main.make_ollama_client", return_value=_mock_ollama()),
        patch("src.main.make_langfuse_tracer", return_value=tracer),
        patch("src.main.make_cache_client", return_value=None),
        patch("src.main.make_telegram_service", return_value=None),
        patch("src.routers.ping.OllamaClient") as mock_ping_ollama,
        patch("src.repositories.paper.PaperRepository.get_by_arxiv_id") as mock_get_by_id,
    ):
        mock_startup.return_value = None

        mock_session = MagicMock()
        mock_get_session.return_value.__enter__.return_value = mock_session
        mock_get_session.return_value.__exit__.return_value = None

        mock_get_by_id.return_value = None
        mock_ping_ollama.return_value.health_check = AsyncMock(
            return_value={"status": "healthy", "message": "Ollama service is running"}
        )

        async with LifespanManager(app) as manager:
            async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client:
                yield client
