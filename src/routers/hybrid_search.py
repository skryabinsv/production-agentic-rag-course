import asyncio
import logging

from fastapi import APIRouter, HTTPException
from src.dependencies import OpenSearchDep, RetrieverDep
from src.schemas.api.search import HybridSearchRequest, SearchHit, SearchResponse
from src.utils.arxiv import arxiv_pdf_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hybrid-search", tags=["hybrid-search"])


@router.post("/", response_model=SearchResponse)
async def hybrid_search(
    request: HybridSearchRequest, opensearch_client: OpenSearchDep, retriever: RetrieverDep
) -> SearchResponse:
    """
    Hybrid search endpoint supporting multiple search modes (optionally reranked).
    """
    try:
        # Blocking OpenSearch calls run in a worker thread instead of stalling the event loop
        if not await asyncio.to_thread(opensearch_client.health_check):
            raise HTTPException(status_code=503, detail="Search service is currently unavailable")

        query_embedding = await retriever.embed_query(request.query) if request.use_hybrid else None

        logger.info(f"Hybrid search: '{request.query}' (hybrid: {request.use_hybrid and query_embedding is not None})")

        rerank = request.rerank and retriever.reranker is not None and request.from_ == 0
        results = await retriever.search(
            query=request.query,
            query_embedding=query_embedding,
            size=retriever.fetch_size(request.size, rerank=rerank),
            categories=request.categories,
            use_hybrid=request.use_hybrid,
            from_=request.from_,
            latest=request.latest_papers,
            min_score=request.min_score,
        )
        raw_hits = results.get("hits", [])
        if rerank:
            raw_hits, rerank = await retriever.rerank(request.query, raw_hits, request.size)

        hits = []
        for hit in raw_hits:
            hits.append(
                SearchHit(
                    arxiv_id=hit.get("arxiv_id", ""),
                    title=hit.get("title", ""),
                    authors=hit.get("authors"),
                    abstract=hit.get("abstract"),
                    published_date=hit.get("published_date"),
                    # Chunks don't store pdf_url, so derive it from the arXiv id.
                    pdf_url=hit.get("pdf_url") or (arxiv_pdf_url(hit["arxiv_id"]) if hit.get("arxiv_id") else None),
                    score=hit.get("score", 0.0),
                    highlights=hit.get("highlights"),
                    chunk_text=hit.get("chunk_text"),
                    chunk_id=hit.get("chunk_id"),
                    # The index stores the field as `section_title`; `section_name` was always empty.
                    section_name=hit.get("section_name") or hit.get("section_title"),
                )
            )

        search_response = SearchResponse(
            query=request.query,
            total=results.get("total", 0),
            hits=hits,
            size=request.size,
            **{"from": request.from_},
            search_mode=("hybrid" if (request.use_hybrid and query_embedding is not None) else "bm25")
            + ("+rerank" if rerank else ""),
        )

        logger.info(f"Search completed: {search_response.total} results returned")
        return search_response

    except HTTPException:
        raise
    except Exception:
        logger.exception("Hybrid search error")
        raise HTTPException(status_code=500, detail="Search failed due to an internal error")
