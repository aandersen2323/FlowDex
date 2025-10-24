"""Pluggable components for tokenization and semantic recall."""
from __future__ import annotations

import importlib
import inspect
import math
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import tiktoken  # type: ignore
except ImportError:  # pragma: no cover - gracefully degrade when tiktoken missing
    tiktoken = None


class BaseTokenizer(ABC):
    """Interface for counting, truncating, and iterating text tokens."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return the number of tokens contained within ``text``."""

    @abstractmethod
    def truncate(self, text: str, budget_tokens: int) -> Tuple[str, int]:
        """Truncate ``text`` so it fits in ``budget_tokens`` tokens."""

    @abstractmethod
    def iter_tokens(self, text: str) -> Iterable[str]:
        """Yield the textual representation of each token in ``text``."""

    @abstractmethod
    def tokenize_semantic(self, text: str) -> List[str]:
        """Return tokens for semantic similarity comparisons."""


class DefaultTokenizer(BaseTokenizer):
    """Tokenizer that prefers ``tiktoken`` but degrades gracefully."""

    _SEMANTIC_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)

    def __init__(self, encoding: str = "cl100k_base") -> None:
        self._encoding_name = encoding
        self._encoder = None
        if tiktoken is not None:
            try:
                self._encoder = tiktoken.get_encoding(encoding)
            except Exception:  # pragma: no cover - fallback when encoding missing
                self._encoder = None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return max(1, len(text) // 4)

    def truncate(self, text: str, budget_tokens: int) -> Tuple[str, int]:
        if not text or budget_tokens <= 0:
            return "", 0

        if self._encoder is None:
            char_budget = max(budget_tokens * 4, 0)
            truncated_text = text[-char_budget:] if char_budget else ""
            return truncated_text, self.count_tokens(truncated_text)

        tokens = self._encoder.encode(text)
        if len(tokens) <= budget_tokens:
            return text, len(tokens)

        truncated_tokens = tokens[-budget_tokens:]
        try:
            truncated_text = self._encoder.decode(truncated_tokens)
        except Exception:  # pragma: no cover - defensive decoding fallback
            start_char = max(
                0,
                len(text)
                - int(len(text) * (budget_tokens / max(len(tokens), 1)) * 1.1),
            )
            truncated_text = text[start_char:]
            truncated_tokens = self._encoder.encode(truncated_text)[-budget_tokens:]
            truncated_text = self._encoder.decode(truncated_tokens)

        return truncated_text, len(truncated_tokens)

    def iter_tokens(self, text: str) -> Iterable[str]:
        if not text:
            return []
        if self._encoder is None:
            return (token for token in text.split())
        tokens = self._encoder.encode(text)
        return (self._encoder.decode([token]) for token in tokens)

    def tokenize_semantic(self, text: str) -> List[str]:
        if not text:
            return []
        return [tok for tok in self._SEMANTIC_PATTERN.findall(text.lower()) if tok]


class BaseSemanticRecall(ABC):
    """Interface for retrieving semantically-similar memory snippets."""

    @abstractmethod
    def recall(self, query: str, exclude_ids: List[str], limit: int) -> List[Dict[str, Any]]:
        """Return up to ``limit`` memory entries relevant to ``query``."""


class RedisTFIDFSemanticRecall(BaseSemanticRecall):
    """Default semantic recall implementation backed by Redis."""

    def __init__(
        self,
        redis_getter: Callable[[], "redis.Redis"],
        tokenizer: BaseTokenizer,
        memory_index_key: str,
        memory_key_prefix: str,
    ) -> None:
        self._redis_getter = redis_getter
        self._tokenizer = tokenizer
        self._memory_index_key = memory_index_key
        self._memory_key_prefix = memory_key_prefix

    def _latest_memory_version(self, client: "redis.Redis", memory_id: str) -> Optional[Dict[str, Any]]:
        import json
        from redis.exceptions import RedisError

        try:
            payload = client.lindex(f"{self._memory_key_prefix}{memory_id}", -1)
        except RedisError as exc:
            raise RuntimeError(f"Failed to load memory {memory_id}: {exc}") from exc
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {"data": payload, "ts": None}

    def recall(self, query: str, exclude_ids: List[str], limit: int) -> List[Dict[str, Any]]:
        from collections import Counter

        import redis
        from redis.exceptions import RedisError

        if not query or not query.strip():
            return []

        tokens_query = self._tokenizer.tokenize_semantic(query)
        if not tokens_query:
            return []

        client = self._redis_getter()
        try:
            memory_ids = client.smembers(self._memory_index_key)
        except RedisError as exc:
            raise RuntimeError(f"Failed to enumerate memory: {exc}") from exc

        query_counter = Counter(tokens_query)
        query_norm = math.sqrt(sum(v * v for v in query_counter.values())) or 1.0

        scored: List[Dict[str, Any]] = []
        for mem_id in memory_ids:
            latest = self._latest_memory_version(client, mem_id)
            if not latest:
                continue
            doc_text = latest.get("data", "")
            doc_tokens = self._tokenizer.tokenize_semantic(doc_text)
            if not doc_tokens:
                continue
            doc_counter = Counter(doc_tokens)
            doc_norm = math.sqrt(sum(v * v for v in doc_counter.values())) or 1.0
            dot = sum(
                query_counter[token] * doc_counter.get(token, 0)
                for token in query_counter
            )
            score = dot / (query_norm * doc_norm)
            if score <= 0:
                continue
            scored.append(
                {
                    "id": mem_id,
                    "score": round(score, 6),
                    "data": doc_text,
                    "ts": latest.get("ts"),
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        results: List[Dict[str, Any]] = []
        for entry in scored:
            if entry["id"] in exclude_ids:
                continue
            results.append(entry)
            if len(results) >= limit:
                break
        return results


def instantiate_component(
    path: Optional[str],
    default_factory: Callable[..., Any],
    **kwargs: Any,
) -> Any:
    """Instantiate a component class or factory from ``path``."""

    if not path:
        return default_factory(**kwargs)

    module_path, _, attr_name = path.partition(":")
    if not module_path or not attr_name:
        raise ValueError(
            "Component path must be in 'module:attribute' format, got %r" % (path,)
        )
    module = importlib.import_module(module_path)
    attr = getattr(module, attr_name)

    if inspect.isclass(attr):
        return attr(**kwargs)
    if callable(attr):
        return attr(**kwargs)
    raise TypeError(f"Component {path!r} is not instantiable")
