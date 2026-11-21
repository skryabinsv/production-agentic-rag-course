"""OpenSearch index configuration for hybrid search (BM25 + Vector).

This configuration supports both keyword search (BM25) and vector similarity search
using HNSW algorithm for approximate nearest neighbor search.

Engine note: the ``nmslib`` k-NN engine is deprecated since OpenSearch 2.19 and new
indices with it cannot be created on OpenSearch 3.x; it also does not support
efficient k-NN filtering. New indices therefore use the ``lucene`` engine, which
supports ``cosinesimil`` and works on both 2.19 and 3.x. Existing indices keep
their engine until they are recreated (``setup_indices(force=True)`` + re-ingest).
"""

import copy
from typing import Any, Dict

ARXIV_PAPERS_CHUNKS_INDEX = "arxiv-papers-chunks"

# Index mapping for chunked papers with vector embeddings
ARXIV_PAPERS_CHUNKS_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index.knn": True,
        "analysis": {
            "analyzer": {
                "standard_analyzer": {"type": "standard", "stopwords": "_english_"},
                "text_analyzer": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "stop", "snowball"]},
            }
        },
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "chunk_id": {"type": "keyword"},
            "arxiv_id": {"type": "keyword"},
            "paper_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "chunk_text": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "chunk_word_count": {"type": "integer"},
            "start_char": {"type": "integer"},
            "end_char": {"type": "integer"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,  # Jina v3 embeddings dimension (overridden from settings)
                "method": {
                    "name": "hnsw",  # Hierarchical Navigable Small World
                    "space_type": "cosinesimil",  # Cosine similarity (overridden from settings)
                    "engine": "lucene",  # nmslib is deprecated (2.19) and rejected for new indices in 3.x
                    "parameters": {
                        "ef_construction": 512,  # Higher value = better recall, slower indexing
                        "m": 16,  # Number of bi-directional links
                    },
                },
            },
            "title": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "authors": {
                "type": "text",
                "analyzer": "standard_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "abstract": {"type": "text", "analyzer": "text_analyzer"},
            "categories": {"type": "keyword"},
            "published_date": {"type": "date"},
            "section_title": {"type": "keyword"},
            "embedding_model": {"type": "keyword"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
        },
    },
}



def build_chunks_index_mapping(vector_dimension: int = 1024, space_type: str = "cosinesimil") -> Dict[str, Any]:
    """Return a copy of the chunk index mapping with vector settings taken from config.

    Keeps ``OPENSEARCH__VECTOR_DIMENSION`` / ``OPENSEARCH__VECTOR_SPACE_TYPE`` and the
    actual index mapping in sync (previously the mapping hard-coded 1024 and ignored config).
    """
    mapping = copy.deepcopy(ARXIV_PAPERS_CHUNKS_MAPPING)
    embedding = mapping["mappings"]["properties"]["embedding"]
    embedding["dimension"] = vector_dimension
    embedding["method"]["space_type"] = space_type
    return mapping


HYBRID_RRF_PIPELINE = {
    "id": "hybrid-rrf-pipeline",
    "description": "Post processor for hybrid RRF search",
    "phase_results_processors": [
        {
            "score-ranker-processor": {
                "combination": {
                    "technique": "rrf",  # Reciprocal Rank Fusion
                    "rank_constant": 60,  # Default k=60 for RRF formula: 1/(k+rank)
                }
            }
        }
    ],
}

# Alternative: Weighted average pipeline (commented out - not used by default)
# This could be used if you need explicit control over BM25 vs vector weights
# However, RRF generally provides better results without manual weight tuning
"""
HYBRID_SEARCH_PIPELINE = {
    "id": "hybrid-ranking-pipeline",
    "description": "Hybrid search pipeline using weighted average for BM25 and vector similarity",
    "phase_results_processors": [
        {
            "normalization-processor": {
                "normalization": {
                    "technique": "l2"  # L2 normalization for better score distribution
                },
                "combination": {
                    "technique": "harmonic_mean",  # Harmonic mean often works better than arithmetic
                    "parameters": {
                        "weights": [0.3, 0.7]  # 30% BM25, 70% vector similarity
                    }
                }
            }
        }
    ]
}
"""
