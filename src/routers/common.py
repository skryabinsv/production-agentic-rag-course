"""Helpers shared by API routers."""

from fastapi import HTTPException
from src.config import Settings


def ensure_model_allowed(model: str, settings: Settings) -> None:
    """Reject model names outside the configured allow-list.

    Without this any client could make Ollama try to pull/load arbitrary (huge) models.
    Configure extra models with ``OLLAMA_ALLOWED_MODELS=llama3.2:1b,llama3.2:3b``.
    """
    if model not in settings.allowed_models:
        raise HTTPException(
            status_code=422,
            detail=f"Model '{model}' is not allowed. Allowed models: {sorted(settings.allowed_models)}",
        )
