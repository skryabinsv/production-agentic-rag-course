import json
import logging
import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.dependencies import CacheDep, LangfuseDep, OllamaDep, RetrieverDep, SettingsDep
from src.routers.common import ensure_model_allowed
from src.schemas.api.ask import AskRequest, AskResponse
from src.services.langfuse.tracer import RAGTracer
from src.utils.arxiv import arxiv_pdf_url

logger = logging.getLogger(__name__)

# Two separate routers - one for regular ask, one for streaming
ask_router = APIRouter(tags=["ask"])
stream_router = APIRouter(tags=["stream"])


async def _prepare_chunks_and_sources(
    request: AskRequest,
    retriever,
    rag_tracer: RAGTracer,
    trace=None,
) -> tuple[List[Dict], List[str], List[str], str]:
    """Retrieve (and optionally rerank) chunks for RAG with clean tracing.

    Returns ``(chunks, sources, arxiv_ids, search_mode)``; ``search_mode`` reflects what actually
    ran (hybrid silently degrades to BM25 when embeddings fail).
    """

    # Handle embeddings for hybrid search
    query_embedding = None
    if request.use_hybrid:
        with rag_tracer.trace_embedding(trace, request.query) as embedding_span:
            query_embedding = await retriever.embed_query(request.query)
            if query_embedding is None and embedding_span:
                rag_tracer.tracer.update_span(embedding_span, output={"success": False, "fallback": "bm25"})

    # Search with tracing (blocking OpenSearch call runs in a worker thread)
    with rag_tracer.trace_search(trace, request.query, request.top_k) as search_span:
        search_results = await retriever.search(
            query=request.query,
            query_embedding=query_embedding,
            size=retriever.fetch_size(request.top_k, rerank=True),
            categories=request.categories,
            use_hybrid=request.use_hybrid,
            from_=0,
            min_score=0.0,
        )
        hits, _reranked = await retriever.rerank(request.query, search_results.get("hits", []), request.top_k)

        # Extract essential data for LLM
        chunks = []
        arxiv_ids = []
        sources: List[str] = []  # ordered by rank, de-duplicated

        for hit in hits:
            arxiv_id = hit.get("arxiv_id", "")

            # Minimal chunk data for LLM
            chunks.append(
                {
                    "arxiv_id": arxiv_id,
                    "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
                }
            )

            if arxiv_id:
                arxiv_ids.append(arxiv_id)
                pdf_url = arxiv_pdf_url(arxiv_id)
                if pdf_url not in sources:
                    sources.append(pdf_url)

        # End search span with essential metadata
        rag_tracer.end_search(search_span, chunks, arxiv_ids, search_results.get("total", 0))

    search_mode = "hybrid" if query_embedding is not None else "bm25"
    return chunks, sources, arxiv_ids, search_mode


@ask_router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    retriever: RetrieverDep,
    ollama_client: OllamaDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
    settings: SettingsDep,
) -> AskResponse:
    """Clean RAG endpoint with essential tracing and exact match caching."""

    ensure_model_allowed(request.model, settings)
    rag_tracer = RAGTracer(langfuse_tracer)
    start_time = time.time()

    with rag_tracer.trace_request("api_user", request.query) as trace:
        try:
            # Check exact cache first
            cached_response = None
            if cache_client:
                try:
                    cached_response = await cache_client.find_cached_response(request)
                    if cached_response:
                        logger.info("Returning cached response for exact query match")
                        return cached_response
                except Exception as e:
                    logger.warning(f"Cache check failed, proceeding with normal flow: {e}")

            # Retrieve chunks
            chunks, sources, _, search_mode = await _prepare_chunks_and_sources(request, retriever, rag_tracer, trace)

            if not chunks:
                response = AskResponse(
                    query=request.query,
                    answer="I couldn't find any relevant information in the papers to answer your question.",
                    sources=[],
                    chunks_used=0,
                    search_mode=search_mode,
                )
                rag_tracer.end_request(trace, response.answer, time.time() - start_time)
                return response

            # Build prompt
            with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                from src.services.ollama.prompts import RAGPromptBuilder

                prompt_builder = RAGPromptBuilder()

                try:
                    prompt_data = prompt_builder.create_structured_prompt(request.query, chunks)
                    final_prompt = prompt_data["prompt"]
                except Exception:
                    final_prompt = prompt_builder.create_rag_prompt(request.query, chunks)

                rag_tracer.end_prompt(prompt_span, final_prompt)

            # Generate answer
            with rag_tracer.trace_generation(trace, request.model, final_prompt) as gen_span:
                rag_response = await ollama_client.generate_rag_answer(query=request.query, chunks=chunks, model=request.model)
                answer = rag_response.get("answer", "Unable to generate answer")
                rag_tracer.end_generation(gen_span, answer, request.model)

            # Prepare response
            response = AskResponse(
                query=request.query,
                answer=answer,
                sources=sources,
                chunks_used=len(chunks),
                search_mode=search_mode,
            )

            rag_tracer.end_request(trace, answer, time.time() - start_time)

            # Store response in exact match cache
            if cache_client:
                try:
                    await cache_client.store_response(request, response)
                except Exception as e:
                    logger.warning(f"Failed to store response in cache: {e}")

            return response

        except HTTPException:
            raise
        except Exception:
            # Log the full error server-side; don't leak internals (hosts, stack details) to clients.
            logger.exception("Error processing /ask request")
            raise HTTPException(status_code=500, detail="Internal error while answering the question")


@stream_router.post("/stream")
async def ask_question_stream(
    request: AskRequest,
    retriever: RetrieverDep,
    ollama_client: OllamaDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
    settings: SettingsDep,
) -> StreamingResponse:
    """Clean streaming RAG endpoint (Server-Sent Events)."""

    ensure_model_allowed(request.model, settings)

    async def generate_stream():
        rag_tracer = RAGTracer(langfuse_tracer)
        start_time = time.time()

        with rag_tracer.trace_request("api_user", request.query) as trace:
            try:
                # Check exact cache first
                if cache_client:
                    try:
                        cached_response = await cache_client.find_cached_response(request)
                        if cached_response:
                            logger.info("Returning cached response for exact streaming query match")

                            # Send metadata first (same format as non-cached)
                            metadata_response = {
                                "sources": cached_response.sources,
                                "chunks_used": cached_response.chunks_used,
                                "search_mode": cached_response.search_mode,
                            }
                            yield f"data: {json.dumps(metadata_response)}\n\n"

                            # Stream the cached response in chunks
                            for chunk in cached_response.answer.split():
                                yield f"data: {json.dumps({'chunk': chunk + ' '})}\n\n"

                            # Send completion signal with just the final answer
                            yield f"data: {json.dumps({'answer': cached_response.answer, 'done': True})}\n\n"
                            return
                    except Exception as e:
                        logger.warning(f"Cache check failed, proceeding with normal flow: {e}")

                # Retrieve chunks
                chunks, sources, _, search_mode = await _prepare_chunks_and_sources(request, retriever, rag_tracer, trace)

                if not chunks:
                    yield f"data: {json.dumps({'answer': 'No relevant information found.', 'sources': [], 'done': True})}\n\n"
                    return

                # Send metadata first
                metadata_response = {"sources": sources, "chunks_used": len(chunks), "search_mode": search_mode}
                yield f"data: {json.dumps(metadata_response)}\n\n"

                # Build prompt
                with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                    from src.services.ollama.prompts import RAGPromptBuilder

                    prompt_builder = RAGPromptBuilder()
                    final_prompt = prompt_builder.create_rag_prompt(request.query, chunks)
                    rag_tracer.end_prompt(prompt_span, final_prompt)

                # Stream generation
                with rag_tracer.trace_generation(trace, request.model, final_prompt) as gen_span:
                    full_response = ""
                    async for chunk in ollama_client.generate_rag_answer_stream(
                        query=request.query, chunks=chunks, model=request.model
                    ):
                        if chunk.get("response"):
                            text_chunk = chunk["response"]
                            full_response += text_chunk
                            yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"

                        if chunk.get("done", False):
                            rag_tracer.end_generation(gen_span, full_response, request.model)
                            yield f"data: {json.dumps({'answer': full_response, 'done': True})}\n\n"
                            break

                rag_tracer.end_request(trace, full_response, time.time() - start_time)

                # Store response in exact match cache
                if cache_client and full_response:
                    try:
                        response_to_cache = AskResponse(
                            query=request.query,
                            answer=full_response,
                            sources=sources,
                            chunks_used=len(chunks),
                            search_mode=search_mode,
                        )
                        await cache_client.store_response(request, response_to_cache)
                    except Exception as e:
                        logger.warning(f"Failed to store streaming response in cache: {e}")

            except Exception:
                logger.exception("Streaming error")
                yield f"data: {json.dumps({'error': 'Internal error while generating the answer'})}\n\n"

    # The body is SSE-framed ("data: ...\n\n"), so advertise it as such: browsers' EventSource
    # and proxies (e.g. nginx buffering) rely on the text/event-stream content type.
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
