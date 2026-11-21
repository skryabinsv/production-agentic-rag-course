# Retrieval evaluation

The course reports Recall@10 / Precision@10 for BM25, vector and hybrid search (week 4),
measured on 3 papers / 81 chunks, but ships neither the dataset nor the code. Numbers from such
a sample say nothing about your index, and without a repeatable measurement there is no way to
tell whether a change to chunking, embeddings or fusion helped or hurt.

`scripts/evaluate_retrieval.py` closes that gap: it sends labelled queries to the running API
(`/api/v1/hybrid-search/`) and reports paper-level Recall@k, Precision@k, Hit-rate@k, MRR and nDCG@k
for BM25 and hybrid modes side by side.

```bash
make start                       # API + OpenSearch with some ingested papers
make eval GOLDEN=evals/my_set.jsonl
# or
uv run python scripts/evaluate_retrieval.py --golden evals/my_set.jsonl --k 10 --output report.json
```

## Building a golden set

`golden_set.example.jsonl` only illustrates the format: it uses classic papers that are almost
certainly **not** in an index built from recent cs.AI submissions, so its scores will be ~0.
Build your own set from papers you actually ingested:

1. Pick 30–100 realistic questions (mix keyword-heavy and paraphrased/semantic ones).
2. For each, list the arXiv ids (without version) of the papers that answer it. Check them in
   OpenSearch Dashboards or with `/api/v1/hybrid-search/` using generous `size`.
3. Keep the file in version control and re-run the script on every retrieval change.

```json
{"query": "how do agents decide when to retrieve", "relevant_arxiv_ids": ["2410.01234"], "categories": ["cs.AI"]}
```

Add `hybrid+rerank` to `--modes` (with `RERANKER__ENABLED=true`) to measure the reranker, and
re-run after changing chunking (`CHUNKING__INCLUDE_ABSTRACT_IN_CHUNKS`) or the embedding provider.

## Answer quality

`scripts/evaluate_answers.py` (`make eval-answers`) uses the same golden set against `/ask` or
`/ask-agentic` and reports:

| metric | meaning |
|---|---|
| `citation_validity` | share of arXiv ids cited in answers that are among the returned sources (hallucinated citations lower it) |
| `answers_with_citations` | share of answers that cite anything |
| `source_hit_rate` | share of questions whose sources include a labelled relevant paper |
| `refusal_rate`, `refusal_accuracy` | "I don't know" behaviour; label out-of-corpus questions with `"answerable": false` |
| `judge_faithfulness` | with `JUDGE=<ollama model>`: share of answers whose claims the judge finds supported by the retrieved excerpts |

```bash
make eval-answers GOLDEN=evals/my_set.jsonl                       # deterministic metrics only
make eval-answers GOLDEN=evals/my_set.jsonl JUDGE=llama3.1:8b     # + LLM-as-judge faithfulness
uv run python scripts/evaluate_answers.py --golden evals/my_set.jsonl --endpoint ask-agentic --output answers.json
```

The judge should be a stronger model than the one under test, and its verdicts are noisy:
spot-check the `per_question` section of the JSON report. User feedback collected via
`POST /api/v1/feedback` in Langfuse is a good source of new golden-set questions.
