"""Answer-level (generation) metrics that need no extra model.

- citation validity: every arXiv id cited in the answer must be one of the retrieved sources
  (a cited id that wasn't retrieved is a hallucinated citation);
- source recall: did the sources include at least one paper labelled relevant;
- refusal behaviour: for questions the corpus can't answer the system should say so.

Faithfulness (are the claims supported by the context?) needs a judge model; see
``scripts/evaluate_answers.py --judge-model``.
"""

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from src.utils.arxiv import strip_arxiv_version

# New-style ids (2401.12345, optional version) and old-style (cs/0501001)
_ARXIV_ID = re.compile(r"(?<![\w/.])((?:\d{4}\.\d{4,5})|(?:[a-z\-]+(?:\.[A-Z]{2})?/\d{7}))(?:v\d+)?(?![\w.])")

_REFUSAL_PATTERNS = [
    r"\bdon'?t contain enough information\b",
    r"\bdo not contain enough information\b",
    r"\b(?:no|not enough) (?:relevant )?information\b",
    r"\bcouldn'?t find\b",
    r"\bcould not find\b",
    r"\bunable to (?:find|answer)\b",
    r"\bcannot answer\b",
    r"\bcan'?t answer\b",
    r"\bnot (?:mentioned|covered|discussed) in the (?:provided )?(?:excerpts|papers|context)\b",
    r"\boutside (?:my|the) (?:domain|scope)\b",
]
_REFUSAL = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def extract_cited_ids(answer: str) -> List[str]:
    """arXiv ids mentioned in an answer (version stripped, de-duplicated, in order)."""
    seen: List[str] = []
    for match in _ARXIV_ID.finditer(answer or ""):
        arxiv_id = strip_arxiv_version(match.group(1))
        if arxiv_id not in seen:
            seen.append(arxiv_id)
    return seen


def ids_from_source_urls(urls: Iterable[str]) -> List[str]:
    ids = []
    for url in urls:
        tail = url.rstrip("/").split("/pdf/")[-1].split("/abs/")[-1]
        ids.append(strip_arxiv_version(tail.removesuffix(".pdf")))
    return ids


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL.search(answer or ""))


@dataclass
class AnswerScores:
    cited_ids: List[str]
    citation_validity: Optional[float]  # None when nothing was cited
    source_hit: Optional[float]  # None when the item has no relevance labels
    refused: bool
    refusal_correct: Optional[bool]  # None when answerability isn't labelled


def score_answer(
    answer: str,
    source_urls: Sequence[str],
    relevant_ids: Optional[Iterable[str]] = None,
    answerable: Optional[bool] = None,
) -> AnswerScores:
    cited = extract_cited_ids(answer)
    retrieved = set(ids_from_source_urls(source_urls))
    validity = (sum(1 for c in cited if c in retrieved) / len(cited)) if cited else None
    relevant = {strip_arxiv_version(x) for x in (relevant_ids or [])}
    source_hit = (1.0 if retrieved & relevant else 0.0) if relevant else None
    refused = is_refusal(answer)
    refusal_correct = None if answerable is None else (refused == (not answerable))
    return AnswerScores(cited, validity, source_hit, refused, refusal_correct)


def aggregate(scores: Sequence[AnswerScores]) -> Dict[str, Optional[float]]:
    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    return {
        "citation_validity": mean([s.citation_validity for s in scores]),
        "answers_with_citations": mean([1.0 if s.cited_ids else 0.0 for s in scores]),
        "source_hit_rate": mean([s.source_hit for s in scores]),
        "refusal_rate": mean([1.0 if s.refused else 0.0 for s in scores]),
        "refusal_accuracy": mean([None if s.refusal_correct is None else float(s.refusal_correct) for s in scores]),
    }
