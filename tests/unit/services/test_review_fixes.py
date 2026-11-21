"""Regression tests for the bugs fixed after the course review (docs/COURSE_REVIEW_RU.md)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.config import LangfuseSettings, Settings
from src.schemas.api.ask import AskRequest
from src.services.agents.agentic_rag import AgenticRAGService
from src.services.agents.models import SourceItem
from src.services.cache.client import CacheClient
from src.services.cache.factory import make_cache_client
from src.services.evaluation import evaluate_rankings, ndcg_at_k, recall_at_k, reciprocal_rank
from src.services.indexing.text_chunker import TextChunker
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient
from src.services.opensearch.index_config_hybrid import build_chunks_index_mapping
from src.utils.arxiv import arxiv_pdf_url, strip_arxiv_version

# --- arXiv ids ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2401.12345v2", "2401.12345"),
        ("2401.12345", "2401.12345"),
        ("solv-int/9901001v1", "solv-int/9901001"),  # old split("v") logic returned "sol"
        ("cs/0501001v3", "cs/0501001"),
    ],
)
def test_strip_arxiv_version(raw, expected):
    assert strip_arxiv_version(raw) == expected


def test_arxiv_pdf_url():
    assert arxiv_pdf_url("1706.03762v7") == "https://arxiv.org/pdf/1706.03762.pdf"


# --- chunking ----------------------------------------------------------------------------


def test_chunk_text_short_document_does_not_crash():
    """Texts shorter than min_chunk_size used to raise TypeError (_reconstruct_text got 2 args)."""
    chunker = TextChunker(chunk_size=600, overlap_size=100, min_chunk_size=100)
    chunks = chunker.chunk_text("a short paper body " * 5, arxiv_id="2401.00001", paper_id="1")
    assert len(chunks) == 1
    assert chunks[0].metadata.word_count == 20


def test_combined_small_sections_use_real_newlines():
    """Combined chunks used to contain the literal characters '\\n\\n' instead of line breaks."""
    chunker = TextChunker(chunk_size=600, overlap_size=100, min_chunk_size=100)
    sections = {
        "Introduction": "intro words " * 30,
        "Related Work": "related words " * 30,
        "Conclusion": "final words " * 30,
    }
    chunks = chunker.chunk_paper(
        title="T", abstract="An abstract about something else entirely " * 3, full_text="", arxiv_id="x", paper_id="1",
        sections=sections,
    )
    assert chunks
    for chunk in chunks:
        assert "\\n" not in chunk.text
    assert any("Section: Introduction\n\n" in chunk.text for chunk in chunks)


def test_empty_abstract_does_not_drop_all_sections():
    chunker = TextChunker()
    assert chunker._is_duplicate_abstract("some section content", "", set()) is False


# --- OpenSearch --------------------------------------------------------------------------


def _opensearch_client() -> OpenSearchClient:
    with patch("src.services.opensearch.client.OpenSearch"):
        client = OpenSearchClient(host="http://localhost:9200", settings=Settings())
    client.client = MagicMock()
    client.client.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}
    return client


def test_hybrid_search_applies_category_filter_to_vector_branch():
    client = _opensearch_client()
    client.search_unified(query="q", query_embedding=[0.1] * 4, size=5, categories=["cs.AI"], use_hybrid=True)

    body = client.client.search.call_args.kwargs["body"]
    bm25_branch, vector_branch = body["query"]["hybrid"]["queries"]
    assert {"terms": {"categories": ["cs.AI"]}} in bm25_branch["bool"]["filter"]
    assert vector_branch["bool"]["filter"] == [{"terms": {"categories": ["cs.AI"]}}]
    assert "knn" in vector_branch["bool"]["must"][0]


def test_hybrid_search_without_categories_keeps_plain_knn():
    client = _opensearch_client()
    client.search_unified(query="q", query_embedding=[0.1] * 4, size=5, use_hybrid=True)
    vector_branch = client.client.search.call_args.kwargs["body"]["query"]["hybrid"]["queries"][1]
    assert "knn" in vector_branch


def test_bulk_index_uses_deterministic_ids():
    client = _opensearch_client()
    with patch("opensearchpy.helpers.bulk", return_value=(2, [])) as bulk:
        client.bulk_index_chunks(
            [
                {"chunk_data": {"arxiv_id": "2401.1", "chunk_index": 0}, "embedding": [0.1]},
                {"chunk_data": {"arxiv_id": "2401.1", "chunk_index": 1}, "embedding": [0.2]},
            ]
        )
    actions = bulk.call_args.args[1]
    assert [a["_id"] for a in actions] == ["2401.1_0", "2401.1_1"]


def test_index_mapping_follows_settings_and_avoids_nmslib():
    mapping = build_chunks_index_mapping(vector_dimension=768, space_type="l2")
    embedding = mapping["mappings"]["properties"]["embedding"]
    assert embedding["dimension"] == 768
    assert embedding["method"]["space_type"] == "l2"
    assert embedding["method"]["engine"] != "nmslib"
    assert "index.knn.space_type" not in mapping["settings"]
    # The module-level template is not mutated
    assert build_chunks_index_mapping()["mappings"]["properties"]["embedding"]["dimension"] == 1024


def test_rrf_pipeline_existence_checked_via_search_pipeline_api():
    client = _opensearch_client()
    client._create_rrf_pipeline(force=False)
    first_call = client.client.transport.perform_request.call_args_list[0]
    assert first_call.args == ("GET", "/_search/pipeline/hybrid-rrf-pipeline")
    client.client.ingest.get_pipeline.assert_not_called()


# --- Ollama ------------------------------------------------------------------------------


def test_ollama_payload_puts_sampling_params_into_options():
    """Top-level temperature/top_p are ignored by Ollama; they must live under `options`."""
    client = OllamaClient(Settings())
    payload = client._build_payload("llama3.2:1b", "hi", False, {"temperature": 0.1, "top_p": 0.9, "format": {"type": "object"}})
    assert "temperature" not in payload
    assert payload["options"]["temperature"] == 0.1
    assert payload["options"]["top_p"] == 0.9
    assert payload["options"]["num_ctx"] == Settings().ollama_num_ctx
    assert payload["format"] == {"type": "object"}


def test_ollama_client_provides_langchain_model():
    """Agent nodes call get_langchain_model(); it did not exist, so every node fell back."""
    llm = OllamaClient(Settings()).get_langchain_model(model="llama3.2:1b", temperature=0.0)
    assert llm.model == "llama3.2:1b"
    assert llm.temperature == 0.0
    assert llm.num_ctx == Settings().ollama_num_ctx


# --- Config ------------------------------------------------------------------------------


def test_langfuse_settings_accept_sdk_style_env_names(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    settings = LangfuseSettings()
    assert settings.public_key == "pk-test"
    assert settings.secret_key == "sk-test"


def test_langfuse_settings_accept_double_underscore_env_names(monkeypatch):
    monkeypatch.setenv("LANGFUSE__PUBLIC_KEY", "pk-double")
    assert LangfuseSettings().public_key == "pk-double"


def test_allowed_models_always_include_default_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_ALLOWED_MODELS", "a, b")
    settings = Settings()
    assert settings.allowed_models == {"a", "b", settings.ollama_model}


# --- Cache -------------------------------------------------------------------------------


def test_cache_key_normalises_case_and_whitespace():
    cache = CacheClient(MagicMock(), Settings().redis)
    key_a = cache._generate_cache_key(AskRequest(query="What is  RAG?"))
    key_b = cache._generate_cache_key(AskRequest(query="  what is rag? "))
    assert key_a == key_b


def test_cache_factory_returns_none_when_redis_unavailable():
    with patch("src.services.cache.factory.make_redis_client", side_effect=ConnectionError("no redis")):
        assert make_cache_client(Settings()) is None


# --- Agentic service ---------------------------------------------------------------------


def test_agentic_sources_are_urls_and_chunks_counted():
    result = {
        "relevant_sources": [
            SourceItem(arxiv_id="1", title="t", url="https://arxiv.org/pdf/1.pdf"),
            SourceItem(arxiv_id="1", title="t", url="https://arxiv.org/pdf/1.pdf"),
        ],
        "messages": [
            HumanMessage(content="q"),
            ToolMessage(content="ctx", tool_call_id="1", artifact=[Document(page_content="a"), Document(page_content="b")]),
            AIMessage(content="answer"),
        ],
    }
    assert AgenticRAGService._extract_sources(None, result) == ["https://arxiv.org/pdf/1.pdf"]
    assert AgenticRAGService._count_retrieved_chunks(result) == 2


# --- Retrieval evaluation ----------------------------------------------------------------


def test_retrieval_metrics_basic():
    assert recall_at_k(["a", "b", "c"], ["a", "x"], k=2) == 0.5
    assert reciprocal_rank(["b", "a"], ["a"]) == 0.5
    assert ndcg_at_k(["a"], ["a"], k=10) == 1.0


def test_evaluate_rankings_collapses_duplicate_chunks():
    # Two chunks of paper "a" must not count twice or push "b" out of the top-2.
    report = evaluate_rankings([["a", "a", "b"]], [["a", "b"]], k=2)
    assert report.metrics["recall@2"] == 1.0
    assert report.metrics["precision@2"] == 1.0
    assert report.metrics["mrr"] == 1.0
