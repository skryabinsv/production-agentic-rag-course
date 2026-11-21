"""Offline retrieval-quality metrics.

The course publishes Recall@10 / Precision@10 numbers (week 4) but ships no code or dataset
to reproduce them, so changes to chunking, embeddings or fusion can't be measured.
These helpers turn a small labelled query set into comparable numbers; see
``scripts/evaluate_retrieval.py`` and ``evals/README.md``.

All metrics work at *paper* level: a query is labelled with the arXiv ids that answer it,
and a ranked list of retrieved ids (chunks collapsed to their paper, first occurrence wins)
is scored against those labels.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence


def dedupe_preserving_order(ids: Iterable[str]) -> List[str]:
    """Collapse repeated ids (several chunks of one paper) keeping the best rank."""
    seen = set()
    result = []
    for item in ids:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant_set)
    return hits / len(relevant_set)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    hits = len(set(retrieved[:k]) & relevant_set)
    return hits / k


def hit_rate_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if at least one relevant item is in the top-k (a.k.a. success@k)."""
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    for rank, item in enumerate(retrieved, 1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    relevant_set = set(relevant)
    if not relevant_set or k <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(retrieved[:k], 1) if item in relevant_set)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


@dataclass
class RetrievalReport:
    """Aggregated metrics over a query set."""

    k: int
    num_queries: int
    metrics: Dict[str, float]
    per_query: List[Dict] = field(default_factory=list)


def evaluate_rankings(rankings: Sequence[Sequence[str]], labels: Sequence[Iterable[str]], k: int = 10) -> RetrievalReport:
    """Score ranked id lists against relevance labels and return mean metrics.

    :param rankings: one ranked list of retrieved ids per query (duplicates allowed)
    :param labels: one collection of relevant ids per query
    :param k: cut-off for @k metrics
    """
    if len(rankings) != len(labels):
        raise ValueError("rankings and labels must have the same length")

    per_query = []
    for retrieved, relevant in zip(rankings, labels):
        ranked = dedupe_preserving_order(retrieved)
        relevant = list(relevant)
        per_query.append(
            {
                f"recall@{k}": recall_at_k(ranked, relevant, k),
                f"precision@{k}": precision_at_k(ranked, relevant, k),
                f"hit_rate@{k}": hit_rate_at_k(ranked, relevant, k),
                "mrr": reciprocal_rank(ranked, relevant),
                f"ndcg@{k}": ndcg_at_k(ranked, relevant, k),
            }
        )

    n = len(per_query)
    keys = per_query[0].keys() if per_query else []
    means = {key: (sum(q[key] for q in per_query) / n) for key in keys}
    return RetrievalReport(k=k, num_queries=n, metrics=means, per_query=per_query)
