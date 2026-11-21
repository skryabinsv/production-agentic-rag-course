import asyncio
import logging
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)


class LocalEmbeddingsClient:
    """In-process embeddings with sentence-transformers (no text leaves the machine).

    Same async interface as ``JinaEmbeddingsClient``. The model is loaded lazily on first use
    and inference runs in a worker thread so the event loop is never blocked.

    Switching provider changes the vector space: the index must be rebuilt
    (``setup_indices(force=True)`` + re-ingest), otherwise query and passage vectors don't match.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        query_prefix: str = "",
        passage_prefix: str = "",
        device: Optional[str] = None,
        batch_size: int = 16,
        expected_dimension: Optional[int] = None,
    ):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.device = device or None
        self.batch_size = batch_size
        self.expected_dimension = expected_dimension
        self._model = None
        self._lock = threading.Lock()
        logger.info(f"Local embeddings client configured with model {model_name}")

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading local embedding model {self.model_name} (first use)...")
                model = SentenceTransformer(self.model_name, device=self.device)
                dimension = model.get_sentence_embedding_dimension()
                if self.expected_dimension and dimension != self.expected_dimension:
                    raise ValueError(
                        f"Embedding model {self.model_name} produces {dimension}-d vectors but the index expects "
                        f"{self.expected_dimension}. Set OPENSEARCH__VECTOR_DIMENSION={dimension} and recreate the index."
                    )
                self._model = model
        return self._model

    def _encode(self, texts: List[str]) -> List[List[float]]:
        model = self._load()
        vectors = model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    async def embed_passages(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        prefixed = [f"{self.passage_prefix}{t}" for t in texts]
        embeddings = await asyncio.to_thread(self._encode, prefixed)
        logger.info(f"Successfully embedded {len(texts)} passages locally")
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        embeddings = await asyncio.to_thread(self._encode, [f"{self.query_prefix}{query}"])
        return embeddings[0]

    async def close(self):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
