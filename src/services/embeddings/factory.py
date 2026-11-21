from typing import Optional, Union

from src.config import Settings, get_settings

from .jina_client import JinaEmbeddingsClient
from .local_client import LocalEmbeddingsClient

EmbeddingsClient = Union[JinaEmbeddingsClient, LocalEmbeddingsClient]


def make_embeddings_client(settings: Optional[Settings] = None) -> EmbeddingsClient:
    """Create the configured embeddings client (``EMBEDDINGS__PROVIDER=jina|local``).

    Creates a new client instance each time to avoid closed client issues.

    :param settings: Optional settings instance
    :returns: Embeddings client with async ``embed_query`` / ``embed_passages``
    """
    if settings is None:
        settings = get_settings()

    if settings.embeddings.provider == "local":
        cfg = settings.embeddings
        return LocalEmbeddingsClient(
            model_name=cfg.local_model,
            query_prefix=cfg.local_query_prefix,
            passage_prefix=cfg.local_passage_prefix,
            device=cfg.local_device or None,
            batch_size=cfg.local_batch_size,
            expected_dimension=settings.opensearch.vector_dimension,
        )

    return JinaEmbeddingsClient(api_key=settings.jina_api_key)


def make_embeddings_service(settings: Optional[Settings] = None) -> EmbeddingsClient:
    """Alias kept for backwards compatibility (used by the API lifespan and notebooks)."""
    return make_embeddings_client(settings)
