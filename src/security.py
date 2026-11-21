"""API protection: API-key authentication and per-client rate limiting.

Both are opt-in through settings so the local learning setup keeps working unchanged:

- ``API_KEYS=key1,key2``  -> every /api/v1 endpoint except /health requires
  ``X-API-Key: <key>`` or ``Authorization: Bearer <key>``.
- ``RATE_LIMIT_PER_MINUTE=30`` -> fixed one-minute window per client (API key, else IP).
  Counters live in Redis when the cache is available (shared by all uvicorn workers),
  otherwise in process memory.
"""

import hashlib
import logging
import secrets
import threading
import time
from typing import Dict, Optional, Tuple

from fastapi import HTTPException, Request, status
from src.config import Settings

logger = logging.getLogger(__name__)


def _extract_api_key(request: Request) -> Optional[str]:
    key = request.headers.get("x-api-key")
    if key:
        return key.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def verify_api_key(request: Request, settings: Settings) -> Optional[str]:
    """Return the presented key if valid, None when auth is disabled; raise 401 otherwise."""
    allowed = settings.api_key_set
    if not allowed:
        return None
    presented = _extract_api_key(request)
    if presented and any(secrets.compare_digest(presented, key) for key in allowed):
        return presented
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


class _MemoryWindowCounter:
    """Per-process fixed-window counter (fallback when Redis is unavailable)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Dict[str, Tuple[int, int]] = {}

    def hit(self, key: str, window: int) -> int:
        with self._lock:
            stored_window, count = self._counts.get(key, (window, 0))
            if stored_window != window:
                count = 0
            count += 1
            self._counts[key] = (window, count)
            if len(self._counts) > 10_000:  # drop stale windows
                self._counts = {k: v for k, v in self._counts.items() if v[0] == window}
            return count


_memory_counter = _MemoryWindowCounter()


def _client_id(request: Request, api_key: Optional[str]) -> str:
    if api_key:
        return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


def enforce_rate_limit(request: Request, settings: Settings, api_key: Optional[str]) -> None:
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return

    now = time.time()
    window = int(now // 60)
    key = f"ratelimit:{_client_id(request, api_key)}:{window}"

    count = None
    cache = getattr(request.app.state, "cache_client", None)
    redis_client = getattr(cache, "redis", None)
    if redis_client is not None:
        try:
            pipe = redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, 120)
            count = int(pipe.execute()[0])
        except Exception as e:  # Redis hiccup: degrade to per-process counting
            logger.warning(f"Rate limiter falling back to memory: {e}")
    if count is None:
        count = _memory_counter.hit(key, window)

    if count > limit:
        retry_after = max(1, int(60 - (now % 60)))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({limit} requests per minute)",
            headers={"Retry-After": str(retry_after)},
        )
