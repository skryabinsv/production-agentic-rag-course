"""Measure retrieval quality of the running API against a labelled query set.

Usage (API must be running, e.g. `make start`):

    uv run python scripts/evaluate_retrieval.py --golden evals/golden_set.example.jsonl
    uv run python scripts/evaluate_retrieval.py --golden evals/my_set.jsonl --k 5 --modes bm25 hybrid hybrid+rerank --output report.json

Golden set format (JSONL, one query per line):

    {"query": "...", "relevant_arxiv_ids": ["2401.01234", "2312.05678"], "categories": ["cs.AI"]}

`categories` is optional. Labels must refer to papers that are actually in *your* index;
see evals/README.md for how to build a set.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.evaluation import evaluate_rankings  # noqa: E402
from src.utils.arxiv import strip_arxiv_version  # noqa: E402


def load_golden_set(path: Path) -> List[Dict]:
    items = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = json.loads(line)
        if not item.get("query") or not item.get("relevant_arxiv_ids"):
            raise ValueError(f"{path}:{line_no}: 'query' and 'relevant_arxiv_ids' are required")
        items.append(item)
    if not items:
        raise ValueError(f"{path}: no queries found")
    return items


def retrieve(client: httpx.Client, api_url: str, item: Dict, mode: str, k: int) -> Dict:
    payload = {
        "query": item["query"],
        # Over-fetch chunks: several chunks usually belong to the same paper
        "size": min(k * 3, 100),
        "use_hybrid": mode.startswith("hybrid"),
        "rerank": mode.endswith("+rerank"),
    }
    if item.get("categories"):
        payload["categories"] = item["categories"]
    response = client.post(f"{api_url.rstrip('/')}/api/v1/hybrid-search/", json=payload)
    response.raise_for_status()
    data = response.json()
    return {
        "ids": [strip_arxiv_version(hit["arxiv_id"]) for hit in data.get("hits", []) if hit.get("arxiv_id")],
        "search_mode": data.get("search_mode"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", type=Path, required=True, help="JSONL file with labelled queries")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--modes", nargs="+", default=["bm25", "hybrid"], choices=["bm25", "hybrid", "bm25+rerank", "hybrid+rerank"]
    )
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""), help="X-API-Key if the API uses API_KEYS")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON report")
    args = parser.parse_args()

    golden = load_golden_set(args.golden)
    labels = [[strip_arxiv_version(x) for x in item["relevant_arxiv_ids"]] for item in golden]

    report: Dict[str, Dict] = {}
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    with httpx.Client(timeout=60.0, headers=headers) as client:
        for mode in args.modes:
            results = [retrieve(client, args.api_url, item, mode, args.k) for item in golden]
            fallbacks = sum(1 for r in results if r["search_mode"] != mode)
            if fallbacks:
                print(f"WARNING: {fallbacks}/{len(results)} '{mode}' queries were served in another mode "
                      f"(e.g. embeddings unavailable -> BM25). Numbers for '{mode}' are not pure.")
            evaluation = evaluate_rankings([r["ids"] for r in results], labels, k=args.k)
            report[mode] = {
                "num_queries": evaluation.num_queries,
                "metrics": evaluation.metrics,
                "per_query": [
                    {"query": item["query"], **scores} for item, scores in zip(golden, evaluation.per_query)
                ],
            }

    metric_names = list(next(iter(report.values()))["metrics"].keys())
    print(f"\nRetrieval evaluation on {len(golden)} queries (paper-level, k={args.k})\n")
    print("| mode | " + " | ".join(metric_names) + " |")
    print("|---|" + "---|" * len(metric_names))
    for mode, data in report.items():
        print(f"| {mode} | " + " | ".join(f"{data['metrics'][m]:.3f}" for m in metric_names) + " |")
    if len(golden) < 30:
        print(f"\nNote: {len(golden)} queries is too few for stable conclusions; aim for 30-100.")

    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nFull report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
