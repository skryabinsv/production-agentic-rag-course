"""Second-stage rerankers (cross-encoders) for retrieved chunks.

Hybrid retrieval optimises recall; a cross-encoder reads query and chunk *together* and is
much better at ordering the final top-k that goes into the prompt.
"""

import asyncio
import logging
import threading
from typing import Dict, List, Optional, Protocol

import httpx
from src.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_JINA_RERANKER = "jina-reranker-v2-base-multilingual"
DEFAULT_LOCAL_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _hit_text(hit: Dict) -> str:
    return hit.get("chunk_text") or hit.get("abstract") or hit.get("title") or ""


class Reranker(Protocol):
    model_name: str

    async def rerank(self, query: str, hits: List[Dict], top_k: int) -> List[Dict]: ...


def _apply_scores(hits: List[Dict], scores: List[float], top_k: int) -> List[Dict]:
    ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
    result = []
    for hit, score in ranked:
        hit = dict(hit)
        hit["retrieval_score"] = hit.get("score")
        hit["score"] = float(score)
        hit["rerank_score"] = float(score)
        result.append(hit)
    return result


class JinaReranker:
    """Hosted reranker (same API key as Jina embeddings)."""

    def __init__(self, api_key: str, model_name: str = DEFAULT_JINA_RERANKER, timeout: float = 15.0):
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout

    async def rerank(self, query: str, hits: List[Dict], top_k: int) -> List[Dict]:
        if not hits:
            return []
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [_hit_text(h) for h in hits],
            "top_n": len(hits),
            "return_documents": False,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
        scores = [0.0] * len(hits)
        for item in response.json().get("results", []):
            scores[item["index"]] = item["relevance_score"]
        return _apply_scores(hits, scores, top_k)


class LocalCrossEncoderReranker:
    """In-process cross-encoder via sentence-transformers (private, CPU-friendly default)."""

    def __init__(self, model_name: str = DEFAULT_LOCAL_RERANKER, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or None
        self._model = None
        self._lock = threading.Lock()

    def _score(self, query: str, texts: List[str]) -> List[float]:
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                logger.info(f"Loading local reranker {self.model_name} (first use)...")
                self._model = CrossEncoder(self.model_name, device=self.device)
        return [float(s) for s in self._model.predict([(query, t) for t in texts], show_progress_bar=False)]

    async def rerank(self, query: str, hits: List[Dict], top_k: int) -> List[Dict]:
        if not hits:
            return []
        scores = await asyncio.to_thread(self._score, query, [_hit_text(h) for h in hits])
        return _apply_scores(hits, scores, top_k)


def make_reranker(settings: Settings) -> Optional[Reranker]:
    """Build the configured reranker, or None when ``RERANKER__ENABLED`` is false."""
    cfg = settings.reranker
    if not cfg.enabled:
        return None
    if cfg.provider == "local":
        return LocalCrossEncoderReranker(model_name=cfg.model or DEFAULT_LOCAL_RERANKER, device=settings.embeddings.local_device)
    if not settings.jina_api_key:
        logger.warning("RERANKER__PROVIDER=jina but JINA_API_KEY is empty - reranking disabled")
        return None
    return JinaReranker(api_key=settings.jina_api_key, model_name=cfg.model or DEFAULT_JINA_RERANKER, timeout=cfg.timeout_seconds)
