"""Single retrieval path shared by /ask, /stream, the agent tool and the Telegram bot.

Before, each of them had its own copy of "embed -> search -> collect chunks", with different
fallback behaviour, and all called the synchronous OpenSearch client directly from async code
(blocking the event loop). This module:

- embeds the query, degrading to BM25 if the embedding provider fails;
- runs the blocking OpenSearch call in a worker thread;
- optionally over-fetches candidates and reranks them with a cross-encoder.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    hits: List[Dict[str, Any]]
    search_mode: str  # "bm25" | "hybrid"
    total: int = 0
    reranked: bool = False
    errors: List[str] = field(default_factory=list)


class Retriever:
    def __init__(self, opensearch_client, embeddings_client, reranker=None, rerank_candidates: int = 20):
        self.opensearch = opensearch_client
        self.embeddings = embeddings_client
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates

    async def embed_query(self, query: str) -> Optional[List[float]]:
        """Query embedding, or None if the provider is unavailable (caller falls back to BM25)."""
        try:
            return await self.embeddings.embed_query(query)
        except Exception as e:
            logger.warning(f"Embedding failed, falling back to BM25: {e}")
            return None

    def fetch_size(self, top_k: int, rerank: bool) -> int:
        return max(top_k, self.rerank_candidates) if (rerank and self.reranker) else top_k

    async def search(
        self,
        query: str,
        query_embedding: Optional[List[float]],
        size: int,
        categories: Optional[List[str]] = None,
        use_hybrid: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run the (synchronous) OpenSearch query in a worker thread."""
        return await asyncio.to_thread(
            self.opensearch.search_unified,
            query=query,
            query_embedding=query_embedding,
            size=size,
            categories=categories,
            use_hybrid=use_hybrid and query_embedding is not None,
            **kwargs,
        )

    async def rerank(self, query: str, hits: List[Dict[str, Any]], top_k: int) -> tuple[List[Dict[str, Any]], bool]:
        """Rerank hits down to top_k; on reranker failure keep the retrieval order."""
        if not self.reranker or not hits:
            return hits[:top_k], False
        try:
            return await self.reranker.rerank(query, hits, top_k), True
        except Exception as e:
            logger.warning(f"Reranking failed, keeping retrieval order: {e}")
            return hits[:top_k], False

    async def retrieve(
        self,
        query: str,
        top_k: int,
        use_hybrid: bool = True,
        categories: Optional[List[str]] = None,
        rerank: bool = True,
    ) -> RetrievalResult:
        embedding = await self.embed_query(query) if use_hybrid else None
        results = await self.search(
            query, embedding, self.fetch_size(top_k, rerank), categories=categories, use_hybrid=use_hybrid
        )
        hits, reranked = await self.rerank(query, results.get("hits", []), top_k) if rerank else (results.get("hits", [])[:top_k], False)
        return RetrievalResult(
            hits=hits,
            search_mode="hybrid" if embedding is not None else "bm25",
            total=results.get("total", 0),
            reranked=reranked,
        )
