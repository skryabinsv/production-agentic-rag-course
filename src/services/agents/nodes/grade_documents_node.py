import asyncio
import logging
import time
from typing import Dict

from langgraph.runtime import Runtime

from ..context import Context
from ..models import GradeDocuments, GradingResult
from ..prompts import GRADE_DOCUMENTS_PROMPT
from ..state import AgentState
from ..tools import format_documents_for_llm
from .utils import documents_to_sources, get_latest_context, get_latest_tool_documents, get_original_query

logger = logging.getLogger(__name__)


async def ainvoke_grade_documents_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, str | list]:
    """Grade retrieved documents for relevance using LLM.

    This function uses an LLM to evaluate whether the retrieved documents
    are relevant to the user's query and decides whether to generate an
    answer or rewrite the query for better results.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary with routing_decision and grading_results
    """
    logger.info("NODE: grade_documents")
    start_time = time.time()

    # Grade against what the user asked, not against a retrieval-oriented rewrite
    question = get_original_query(state)
    context = get_latest_context(state["messages"])

    # Extract document chunks from context for logging
    chunks_preview = []
    if context:
        # Context is a string containing all documents concatenated
        # Let's show a preview of what was retrieved
        context_preview = context[:500] + "..." if len(context) > 500 else context
        chunks_preview = [{"text_preview": context_preview, "length": len(context)}]

    # Create span for document grading
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="document_grading",
                input_data={
                    "query": question,
                    "context_length": len(context) if context else 0,
                    "has_context": context is not None,
                    "chunks_received": chunks_preview,
                },
                metadata={
                    "node": "grade_documents",
                    "model": runtime.context.model_name,
                },
            )
            logger.debug("Created Langfuse span for document grading")
        except Exception as e:
            logger.warning(f"Failed to create span for grade_documents node: {e}")

    if not context:
        logger.warning("No context found, routing to rewrite_query")

        # Update span with no context result
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"routing_decision": "rewrite_query", "reason": "no_context"},
                metadata={"execution_time_ms": execution_time},
            )

        return {"routing_decision": "rewrite_query", "grading_results": [], "relevant_sources": []}

    logger.debug(f"Grading context of length {len(context)} characters")

    llm = None
    try:
        llm = runtime.context.ollama_client.get_langchain_model(model=runtime.context.model_name, temperature=0.0)
    except Exception as e:
        logger.error(f"Could not create grading LLM: {e}")

    documents = get_latest_tool_documents(state["messages"])

    if documents:
        # Grade every document separately (concurrently) and keep only relevant ones. The
        # original implementation produced a single yes/no for the whole concatenated context,
        # so irrelevant chunks still reached the answer prompt and the source list.
        grading_results = list(
            await asyncio.gather(*(_grade_one(llm, question, doc, i) for i, doc in enumerate(documents)))
        )
        relevant_docs = [doc for doc, grade in zip(documents, grading_results) if grade.is_relevant]
        is_relevant = bool(relevant_docs)
        relevant_context = format_documents_for_llm(relevant_docs)
        relevant_sources = documents_to_sources(relevant_docs)
    else:
        # No structured artefact (e.g. legacy tool output): grade the whole context at once
        grading_results = [await _grade_text(llm, question, context, "retrieved_docs")]
        is_relevant = grading_results[0].is_relevant
        relevant_context = context if is_relevant else ""
        relevant_sources = []

    score = sum(g.score for g in grading_results) / len(grading_results) if grading_results else 0.0

    # Determine routing
    route = "generate_answer" if is_relevant else "rewrite_query"

    logger.info(
        f"Grading result: {sum(g.is_relevant for g in grading_results)}/{len(grading_results)} relevant, routing to: {route}"
    )

    # Update span with grading result
    if span:
        execution_time = (time.time() - start_time) * 1000
        runtime.context.langfuse_tracer.end_span(
            span,
            output={
                "routing_decision": route,
                "is_relevant": is_relevant,
                "score": score,
                "per_document": [g.model_dump() for g in grading_results],
            },
            metadata={
                "execution_time_ms": execution_time,
                "context_length": len(context),
            },
        )

    return {
        "routing_decision": route,
        "grading_results": grading_results,
        # Only documents graded relevant are used for the answer and listed as sources
        "relevant_sources": relevant_sources,
        "relevant_context": relevant_context,
    }


async def _grade_text(llm, question: str, text: str, document_id: str) -> GradingResult:
    """Binary relevance grade for one piece of text, with a conservative fallback."""
    try:
        if llm is None:
            raise RuntimeError("grading LLM unavailable")
        structured_llm = llm.with_structured_output(GradeDocuments)
        response = await structured_llm.ainvoke(GRADE_DOCUMENTS_PROMPT.format(context=text, question=question))
        is_relevant = response.binary_score == "yes"
        logger.info(f"LLM grading [{document_id}]: {response.binary_score} - {response.reasoning}")
        return GradingResult(
            document_id=document_id, is_relevant=is_relevant, score=1.0 if is_relevant else 0.0, reasoning=response.reasoning
        )
    except Exception as e:
        logger.error(f"LLM grading failed for {document_id}: {e}, falling back to heuristic")
        # Keep non-trivial documents rather than discarding everything when the grader fails
        is_relevant = len((text or "").strip()) > 50
        return GradingResult(
            document_id=document_id,
            is_relevant=is_relevant,
            score=1.0 if is_relevant else 0.0,
            reasoning=f"Fallback heuristic (LLM failed): {'sufficient content' if is_relevant else 'insufficient content'}",
        )


async def _grade_one(llm, question: str, document, index: int) -> GradingResult:
    meta = getattr(document, "metadata", {}) or {}
    document_id = f"{meta.get('arxiv_id', 'doc')}#{index}"
    return await _grade_text(llm, question, format_documents_for_llm([document]), document_id)
