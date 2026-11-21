"""Tests for AgenticRAGService using LangGraph 2.0 Runtime pattern."""

from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.services.agents.agentic_rag import AgenticRAGService
from src.services.agents.config import GraphConfig
from src.services.agents.models import GuardrailScoring


@pytest.fixture
def test_service(mock_opensearch_client, mock_ollama_client, mock_jina_embeddings_client):
    """Create AgenticRAGService with mocked dependencies."""
    config = GraphConfig(
        model="llama3.2:1b",
        temperature=0.0,
        top_k=3,
        use_hybrid=True,
        max_retrieval_attempts=2,
        guardrail_threshold=60,
    )
    return AgenticRAGService(
        opensearch_client=mock_opensearch_client,
        ollama_client=mock_ollama_client,
        embeddings_client=mock_jina_embeddings_client,
        langfuse_tracer=None,
        graph_config=config,
    )


class TestAgenticRAGServiceInitialization:
    """Tests for service initialization."""

    def test_service_initialization(self, test_service):
        """Test that service initializes correctly."""
        assert test_service.opensearch is not None
        assert test_service.ollama is not None
        assert test_service.embeddings is not None
        assert test_service.graph is not None
        assert test_service.graph_config is not None

    def test_graph_config_values(self, test_service):
        """Test graph configuration values."""
        assert test_service.graph_config.model == "llama3.2:1b"
        assert test_service.graph_config.top_k == 3
        assert test_service.graph_config.use_hybrid is True
        assert test_service.graph_config.max_retrieval_attempts == 2
        assert test_service.graph_config.guardrail_threshold == 60


class TestAgenticRAGAskMethod:
    """Tests for the ask() method."""

    @pytest.mark.asyncio
    async def test_ask_empty_query_validation(self, test_service):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await test_service.ask(query="")

        with pytest.raises(ValueError, match="Query cannot be empty"):
            await test_service.ask(query="   ")

    @pytest.mark.asyncio
    async def test_ask_with_model_override(self, test_service):
        """Test ask method with model parameter override."""
        mock_final_state = {
            "messages": [
                HumanMessage(content="Test query"),
                AIMessage(content="Test answer"),
            ],
            "retrieval_attempts": 0,
            "guardrail_result": GuardrailScoring(score=85, reason="Relevant"),
            "sources": [],
            "relevant_sources": [],
            "grading_results": [],
            "metadata": {},
            "original_query": "Test query",
            "rewritten_query": None,
            "routing_decision": "generate_answer",
            "relevant_tool_artefacts": None,
        }

        test_service.graph.ainvoke = AsyncMock(return_value=mock_final_state)

        result = await test_service.ask(query="Test query", model="llama3.2:3b")

        assert result is not None
        # Verify graph was called
        test_service.graph.ainvoke.assert_called_once()


class TestAgenticRAGGraphVisualization:
    """Tests for graph visualization methods."""

    def test_get_graph_mermaid(self, test_service):
        """Test Mermaid diagram generation."""
        mermaid = test_service.get_graph_mermaid()

        assert isinstance(mermaid, str)
        assert len(mermaid) > 0
        assert "graph" in mermaid.lower() or "flowchart" in mermaid.lower()


class TestAgenticRAGErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_ask_with_graph_execution_error(self, test_service):
        """Test error handling when graph execution fails."""
        # Mock graph to raise an exception
        test_service.graph.ainvoke = AsyncMock(side_effect=Exception("Graph execution failed"))

        with pytest.raises(Exception, match="Graph execution failed"):
            await test_service.ask(query="Test query")


class _FakeStructuredLLM:
    def __init__(self, schema, responses):
        self.schema = schema
        self.responses = responses

    async def ainvoke(self, _prompt):
        return self.responses[self.schema.__name__]


class _FakeLLM:
    """Deterministic stand-in for ChatOllama covering structured and plain calls."""

    def __init__(self, responses):
        self.responses = responses
        self.prompts = []

    def with_structured_output(self, schema):
        return _FakeStructuredLLM(schema, self.responses)

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content="Transformers rely on self-attention [arXiv:1706.03762].")


class TestAgenticRAGEndToEnd:
    """Runs the real compiled graph with fake LLM/search to cover the full Week 7 flow."""

    @pytest.mark.asyncio
    async def test_full_flow_returns_answer_sources_and_chunk_count(self, mock_opensearch_client, mock_jina_embeddings_client):
        from src.services.agents.models import GradeDocuments

        fake_llm = _FakeLLM(
            {
                "GuardrailScoring": GuardrailScoring(score=90, reason="research question"),
                "GradeDocuments": GradeDocuments(binary_score="yes", reasoning="relevant"),
            }
        )
        ollama = Mock()
        ollama.get_langchain_model = Mock(return_value=fake_llm)

        service = AgenticRAGService(
            opensearch_client=mock_opensearch_client,
            ollama_client=ollama,
            embeddings_client=mock_jina_embeddings_client,
            langfuse_tracer=None,
            graph_config=GraphConfig(top_k=3),
        )

        result = await service.ask(query="What are transformers?", top_k=2, categories=["cs.CL"])

        assert result["answer"].startswith("Transformers rely on self-attention")
        assert result["sources"] == ["https://arxiv.org/pdf/1706.03762.pdf", "https://arxiv.org/pdf/1810.04805.pdf"]
        assert result["chunks_used"] == 2
        assert result["retrieval_attempts"] == 1
        assert result["reasoning_steps"][-1] == "Generated answer from context"
        # Per-request retrieval overrides reached OpenSearch
        search_kwargs = mock_opensearch_client.search_unified.call_args.kwargs
        assert search_kwargs["size"] == 2
        assert search_kwargs["categories"] == ["cs.CL"]
        # The generation prompt contains formatted, delimited context
        assert "<document>" in fake_llm.prompts[-1]

    @pytest.mark.asyncio
    async def test_out_of_scope_question_is_rejected(self, mock_opensearch_client, mock_jina_embeddings_client):
        fake_llm = _FakeLLM({"GuardrailScoring": GuardrailScoring(score=10, reason="not research")})
        ollama = Mock()
        ollama.get_langchain_model = Mock(return_value=fake_llm)
        service = AgenticRAGService(
            opensearch_client=mock_opensearch_client,
            ollama_client=ollama,
            embeddings_client=mock_jina_embeddings_client,
        )

        result = await service.ask(query="What is a good pizza recipe?")

        assert "only help with questions about academic research papers" in result["answer"]
        assert result["sources"] == []
        assert result["reasoning_steps"][-1] == "Rejected as out of scope"
        mock_opensearch_client.search_unified.assert_not_called()
