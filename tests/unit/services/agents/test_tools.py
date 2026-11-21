from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document
from src.services.agents.tools import create_retriever_tool


async def _invoke_documents(tool, args):
    """Invoke the tool the way ToolNode does and return its document artefact.

    The tool uses response_format="content_and_artifact": the LLM-facing content is a formatted
    string, the Documents travel as the ToolMessage artefact.
    """
    message = await tool.ainvoke({"type": "tool_call", "id": "call_1", "name": tool.name, "args": args})
    return message.artifact, message.content


@pytest.mark.asyncio
async def test_create_retriever_tool_basic(mock_opensearch_client, mock_jina_embeddings_client):
    """Test basic retriever tool creation and invocation."""
    tool = create_retriever_tool(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        top_k=2,
        use_hybrid=True,
    )

    # Verify tool properties
    assert tool.name == "retrieve_papers"
    assert "Search and return relevant arXiv research papers" in tool.description

    # Invoke tool
    result, content = await _invoke_documents(tool, {"query": "machine learning"})

    # Verify result
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(doc, Document) for doc in result)

    # Verify first document
    first_doc = result[0]
    assert first_doc.page_content == "Transformers are neural network architectures based on self-attention mechanisms."
    assert first_doc.metadata["arxiv_id"] == "1706.03762"
    assert first_doc.metadata["title"] == "Attention Is All You Need"
    assert first_doc.metadata["score"] == 0.95

    # The LLM sees a clean, delimited context instead of a Python repr of Documents
    assert content.startswith("<document>")
    assert "arXiv:1706.03762" in content
    assert "metadata=" not in content
    # Sources are version-less canonical URLs
    assert result[1].metadata["source"] == "https://arxiv.org/pdf/1810.04805.pdf"

    # Verify embeddings were generated
    mock_jina_embeddings_client.embed_query.assert_called_once_with("machine learning")

    # Verify search was called correctly
    mock_opensearch_client.search_unified.assert_called_once()
    call_args = mock_opensearch_client.search_unified.call_args
    assert call_args.kwargs["query"] == "machine learning"
    assert call_args.kwargs["size"] == 2  # search_unified uses 'size', not 'top_k'
    assert call_args.kwargs["use_hybrid"] is True


@pytest.mark.asyncio
async def test_retriever_tool_empty_results(mock_opensearch_client, mock_jina_embeddings_client):
    """Test retriever tool with no results."""
    from unittest.mock import Mock
    mock_opensearch_client.search_unified = Mock(return_value={"hits": []})

    tool = create_retriever_tool(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
    )

    result, content = await _invoke_documents(tool, {"query": "nonexistent topic"})

    assert isinstance(result, list)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_retriever_tool_custom_top_k(mock_opensearch_client, mock_jina_embeddings_client):
    """Test retriever tool with custom top_k parameter."""
    tool = create_retriever_tool(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        top_k=5,
        use_hybrid=False,
    )

    await _invoke_documents(tool, {"query": "test query"})

    call_args = mock_opensearch_client.search_unified.call_args
    # search_unified uses 'size' parameter, not 'top_k'
    assert call_args.kwargs["size"] == 5
    assert call_args.kwargs["use_hybrid"] is False


@pytest.mark.asyncio
async def test_retriever_tool_metadata_fields(mock_opensearch_client, mock_jina_embeddings_client):
    """Test that all expected metadata fields are present."""
    from unittest.mock import Mock
    mock_opensearch_client.search_unified = Mock(return_value={
        "hits": [
            {
                "chunk_text": "Test content",
                "arxiv_id": "2301.00001",
                "title": "Test Paper",
                "authors": "Author One, Author Two",
                "score": 0.95,
                "section_name": "Introduction",
            }
        ]
    })

    tool = create_retriever_tool(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
    )

    result, content = await _invoke_documents(tool, {"query": "test"})

    doc = result[0]
    assert "arxiv_id" in doc.metadata
    assert "title" in doc.metadata
    assert "authors" in doc.metadata
    assert "score" in doc.metadata
    assert "source" in doc.metadata
    assert "section" in doc.metadata


@pytest.mark.asyncio
async def test_retriever_tool_falls_back_to_bm25_when_embeddings_fail(mock_opensearch_client, mock_jina_embeddings_client):
    """An embedding-API outage must degrade to BM25 instead of failing the agent run."""
    mock_jina_embeddings_client.embed_query = AsyncMock(side_effect=RuntimeError("jina down"))
    tool = create_retriever_tool(opensearch_client=mock_opensearch_client, embeddings_client=mock_jina_embeddings_client)

    documents, _ = await _invoke_documents(tool, {"query": "transformers"})

    assert len(documents) == 2
    call_kwargs = mock_opensearch_client.search_unified.call_args.kwargs
    assert call_kwargs["use_hybrid"] is False
    assert call_kwargs["query_embedding"] is None


@pytest.mark.asyncio
async def test_retriever_tool_per_call_overrides(mock_opensearch_client, mock_jina_embeddings_client):
    """Per-request top_k / categories passed by the retrieve node reach OpenSearch."""
    tool = create_retriever_tool(opensearch_client=mock_opensearch_client, embeddings_client=mock_jina_embeddings_client, top_k=3)

    await _invoke_documents(tool, {"query": "q", "top_k": 7, "categories": ["cs.CL"]})

    call_kwargs = mock_opensearch_client.search_unified.call_args.kwargs
    assert call_kwargs["size"] == 7
    assert call_kwargs["categories"] == ["cs.CL"]
