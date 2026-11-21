"""Short-term conversation memory for multi-turn questions.

Stores the last N (question, answer) turns per ``session_id``. Redis is used when the cache is
available, so every uvicorn worker sees the same history; otherwise a per-process store is used
(fine for local single-worker runs, inconsistent across workers - a warning is logged).
"""

import asyncio
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    question: str
    answer: str


class ConversationMemory:
    def __init__(self, redis_client=None, max_turns: int = 5, ttl_seconds: int = 24 * 3600, max_sessions: int = 10_000):
        self.redis = redis_client
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._local: "OrderedDict[str, tuple[float, List[Turn]]]" = OrderedDict()
        self._lock = threading.Lock()
        if redis_client is None:
            logger.warning("Conversation memory uses per-process storage (no Redis): history is not shared between workers")

    @staticmethod
    def _key(session_id: str) -> str:
        return f"conversation:{session_id}"

    # --- sync implementations (run in a worker thread when Redis is used) ---

    def _get_sync(self, session_id: str) -> List[Turn]:
        if self.redis is not None:
            raw = self.redis.lrange(self._key(session_id), -self.max_turns, -1)
            turns = []
            for item in raw:
                try:
                    data = json.loads(item)
                    turns.append(Turn(question=data["q"], answer=data["a"]))
                except (ValueError, KeyError, TypeError):
                    continue
            return turns
        with self._lock:
            entry = self._local.get(session_id)
            if not entry or time.time() - entry[0] > self.ttl_seconds:
                return []
            return list(entry[1][-self.max_turns :])

    def _append_sync(self, session_id: str, question: str, answer: str) -> None:
        if self.redis is not None:
            key = self._key(session_id)
            pipe = self.redis.pipeline()
            pipe.rpush(key, json.dumps({"q": question, "a": answer}, ensure_ascii=False))
            pipe.ltrim(key, -self.max_turns, -1)
            pipe.expire(key, self.ttl_seconds)
            pipe.execute()
            return
        with self._lock:
            _, turns = self._local.pop(session_id, (0.0, []))
            turns = (turns + [Turn(question, answer)])[-self.max_turns :]
            self._local[session_id] = (time.time(), turns)
            while len(self._local) > self.max_sessions:
                self._local.popitem(last=False)

    def _clear_sync(self, session_id: str) -> None:
        if self.redis is not None:
            self.redis.delete(self._key(session_id))
        else:
            with self._lock:
                self._local.pop(session_id, None)

    # --- async API (never raises: memory is best-effort) ---

    async def get_history(self, session_id: Optional[str]) -> List[Turn]:
        if not session_id:
            return []
        try:
            return await asyncio.to_thread(self._get_sync, session_id)
        except Exception as e:
            logger.warning(f"Failed to load conversation history: {e}")
            return []

    async def append(self, session_id: Optional[str], question: str, answer: str) -> None:
        if not session_id:
            return
        try:
            await asyncio.to_thread(self._append_sync, session_id, question, answer)
        except Exception as e:
            logger.warning(f"Failed to store conversation turn: {e}")

    async def clear(self, session_id: str) -> None:
        try:
            await asyncio.to_thread(self._clear_sync, session_id)
        except Exception as e:
            logger.warning(f"Failed to clear conversation: {e}")


def format_history(turns: List[Turn], max_answer_chars: int = 600) -> str:
    """Render history compactly for the contextualisation prompt."""
    lines = []
    for turn in turns:
        answer = turn.answer if len(turn.answer) <= max_answer_chars else turn.answer[:max_answer_chars] + "..."
        lines.append(f"User: {turn.question}\nAssistant: {answer}")
    return "\n\n".join(lines)
