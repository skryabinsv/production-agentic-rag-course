"""Shared fixtures for agentic RAG tests.

These fixtures were referenced by the agent test modules but never defined, so 35 tests
errored at collection time and the agentic code path was effectively untested.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.services.agents.context import Context


@pytest.fixture
def sample_search_hits():
    return [
        {
            "chunk_text": "Transformers are neural network architectures based on self-attention mechanisms.",
            "arxiv_id": "1706.03762",
            "title": "Attention Is All You Need",
            "authors": "Ashish Vaswani, Noam Shazeer",
            "score": 0.95,
            "section_title": "Introduction",
        },
        {
            "chunk_text": "BERT pre-trains deep bidirectional representations from unlabeled text.",
            "arxiv_id": "1810.04805v2",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": "Jacob Devlin, Ming-Wei Chang",
            "score": 0.87,
            "section_title": "Method",
        },
    ]


@pytest.fixture
def mock_opensearch_client(sample_search_hits):
    client = Mock()
    client.search_unified = Mock(return_value={"total": len(sample_search_hits), "hits": sample_search_hits})
    return client


@pytest.fixture
def mock_jina_embeddings_client():
    client = Mock()
    client.embed_query = AsyncMock(return_value=[0.1] * 1024)
    return client


@pytest.fixture
def mock_llm():
    """LLM double: `with_structured_output()` returns the same object so tests can set `ainvoke`."""
    llm = Mock()
    llm.with_structured_output = Mock(return_value=llm)
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="mock answer"))
    return llm


@pytest.fixture
def mock_ollama_client(mock_llm):
    client = Mock()
    client.get_langchain_model = Mock(return_value=mock_llm)
    return client


@pytest.fixture
def test_context(mock_ollama_client, mock_opensearch_client, mock_jina_embeddings_client):
    return Context(
        ollama_client=mock_ollama_client,
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        langfuse_tracer=None,
        trace=None,
        langfuse_enabled=False,
        model_name="llama3.2:1b",
        temperature=0.0,
        top_k=3,
        max_retrieval_attempts=2,
        guardrail_threshold=60,
    )


@pytest.fixture
def sample_human_message():
    return HumanMessage(content="What is machine learning?")


@pytest.fixture
def sample_ai_message():
    return AIMessage(content="Machine learning is a field of AI.")


@pytest.fixture
def sample_tool_message():
    documents = [
        Document(
            page_content="Transformers are neural network architectures based on self-attention mechanisms.",
            metadata={
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need",
                "authors": "Ashish Vaswani, Noam Shazeer",
                "score": 0.95,
                "source": "https://arxiv.org/pdf/1706.03762.pdf",
            },
        )
    ]
    return ToolMessage(
        content="<document>\n[1] arXiv:1706.03762 | Attention Is All You Need\n"
        "Transformers are neural network architectures based on self-attention mechanisms.\n</document>",
        tool_call_id="retrieve_1",
        name="retrieve_papers",
        artifact=documents,
    )
