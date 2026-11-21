import pytest


async def test_ask_endpoint_basic(client):
    response = await client.post("/api/v1/ask", json={"query": "What is machine learning?", "model": "llama3.2:3b"})

    assert response.status_code in [200, 500, 503]

    if response.status_code == 200:
        data = response.json()

        assert "query" in data
        assert "answer" in data
        assert "sources" in data
        assert "chunks_used" in data
        assert "search_mode" in data

        assert data["query"] == "What is machine learning?"
        assert isinstance(data["sources"], list)
        assert isinstance(data["chunks_used"], int)


async def test_ask_endpoint_returns_answer_and_versionless_sources(client):
    """/ask used to crash with AttributeError (missing Langfuse helpers) on every request."""
    response = await client.post("/api/v1/ask", json={"query": "What are transformers?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"].startswith("Transformers use self-attention")
    assert data["sources"] == ["https://arxiv.org/pdf/1706.03762.pdf"]
    assert data["chunks_used"] == 1
    assert data["search_mode"] == "hybrid"


async def test_stream_endpoint_emits_sse_events(client):
    response = await client.post("/api/v1/stream", json={"query": "What are transformers?"})

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    events = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert '"done": true' in events[-1]
    assert "error" not in response.text


async def test_ask_endpoint_with_hybrid_search(client):
    response = await client.post(
        "/api/v1/ask", json={"query": "neural networks", "model": "llama3.2:3b", "use_hybrid": True, "top_k": 5}
    )

    assert response.status_code in [200, 500, 503]

    if response.status_code == 200:
        data = response.json()
        assert data["query"] == "neural networks"


async def test_ask_endpoint_with_categories(client):
    response = await client.post(
        "/api/v1/ask", json={"query": "computer vision", "model": "llama3.2:3b", "categories": ["cs.CV", "cs.AI"], "top_k": 3}
    )

    assert response.status_code in [200, 500, 503]


async def test_ask_endpoint_validation_errors(client):
    response = await client.post("/api/v1/ask", json={"query": "", "model": "llama3.2:3b"})
    assert response.status_code == 422

    response = await client.post("/api/v1/ask", json={"model": "llama3.2:3b"})
    assert response.status_code == 422

    response = await client.post("/api/v1/ask", json={"query": "test", "model": "llama3.2:3b", "top_k": 0})
    assert response.status_code == 422


async def test_stream_endpoint_basic(client):
    response = await client.post("/api/v1/stream", json={"query": "What is deep learning?", "model": "llama3.2:3b"})

    assert response.status_code in [200, 500, 503]

    if response.status_code == 200:
        assert "text/event-stream" in response.headers.get("content-type", "")


async def test_ask_endpoint_rejects_unknown_model(client):
    response = await client.post("/api/v1/ask", json={"query": "test", "model": "some-huge-model:405b"})
    assert response.status_code == 422


async def test_stream_endpoint_validation_errors(client):
    response = await client.post("/api/v1/stream", json={"query": "", "model": "llama3.2:3b"})
    assert response.status_code == 422
