import logging

from fastapi import APIRouter, HTTPException
from src.dependencies import AgenticRAGDep, ConversationMemoryDep, LangfuseDep, SettingsDep
from src.routers.common import ensure_model_allowed
from src.schemas.api.ask import AgenticAskResponse, AskRequest, FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["agentic-rag"])


@router.post("/ask-agentic", response_model=AgenticAskResponse)
async def ask_agentic(
    request: AskRequest,
    agentic_rag: AgenticRAGDep,
    settings: SettingsDep,
    memory: ConversationMemoryDep,
) -> AgenticAskResponse:
    """
    Agentic RAG endpoint with intelligent retrieval and query refinement.

    Features:
    - Decides if retrieval is needed
    - Grades document relevance
    - Rewrites queries if needed
    - Provides reasoning transparency

    The agent will automatically:
    1. Determine if the question requires research paper retrieval
    2. If needed, search for relevant papers
    3. Grade retrieved documents for relevance
    4. Rewrite the query if documents aren't relevant
    5. Generate an answer with citations

    Request parameters (model, top_k, use_hybrid, categories) are honoured; previously
    they were accepted by the schema but silently ignored. Pass the same ``session_id`` on
    consecutive requests for multi-turn conversations.

    Args:
        request: Question and parameters
        agentic_rag: Injected agentic RAG service

    Returns:
        Answer with sources and reasoning steps

    Raises:
        HTTPException: If processing fails
    """
    ensure_model_allowed(request.model, settings)

    use_memory = bool(request.session_id and memory)
    try:
        history = await memory.get_history(request.session_id) if use_memory else []
        ask_kwargs = dict(
            query=request.query,
            model=request.model,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            categories=request.categories,
        )
        if history:
            ask_kwargs["history"] = history
        if request.session_id:
            ask_kwargs["user_id"] = request.session_id
        result = await agentic_rag.ask(**ask_kwargs)

        if use_memory:
            await memory.append(request.session_id, request.query, result["answer"])

        return AgenticAskResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result.get("sources", []),
            # Report what was actually retrieved, not what was requested
            chunks_used=result.get("chunks_used", 0),
            search_mode="hybrid" if request.use_hybrid else "bm25",
            reasoning_steps=result.get("reasoning_steps", []),
            retrieval_attempts=result.get("retrieval_attempts", 0),
            rewritten_query=result.get("rewritten_query"),
            standalone_query=result.get("standalone_query"),
            session_id=request.session_id,
            trace_id=result.get("trace_id"),
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Agentic RAG request failed")
        raise HTTPException(status_code=500, detail="Error processing question")


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    langfuse_tracer: LangfuseDep,
) -> FeedbackResponse:
    """
    Submit user feedback for an agentic RAG response.

    This endpoint allows users to rate the quality of answers and provide
    optional comments. Feedback is tracked in Langfuse for continuous improvement.

    Args:
        request: Feedback data including trace_id, score, and optional comment
        langfuse_tracer: Injected Langfuse tracer service

    Returns:
        FeedbackResponse indicating success or failure

    Raises:
        HTTPException: If feedback submission fails
    """
    try:
        if not langfuse_tracer or not getattr(langfuse_tracer, "client", None):
            raise HTTPException(status_code=503, detail="Langfuse tracing is disabled. Cannot submit feedback.")

        success = langfuse_tracer.submit_feedback(
            trace_id=request.trace_id,
            score=request.score,
            comment=request.comment,
        )

        if success:
            # Flush to ensure feedback is sent immediately
            langfuse_tracer.flush()

            return FeedbackResponse(success=True, message="Feedback recorded successfully")
        else:
            raise HTTPException(status_code=500, detail="Failed to submit feedback to Langfuse")

    except HTTPException:
        raise
    except Exception:
        logger.exception("Feedback submission failed")
        raise HTTPException(status_code=500, detail="Error submitting feedback")
