import logging
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..models import ReasoningStep, SourceItem, ToolArtefact

logger = logging.getLogger(__name__)


def extract_sources_from_tool_messages(messages: List) -> List[SourceItem]:
    """Extract sources from tool messages in conversation.

    :param messages: List of messages from graph state
    :returns: List of SourceItem objects
    """
    sources = []

    for msg in messages:
        if isinstance(msg, ToolMessage) and hasattr(msg, "name"):
            if msg.name == "retrieve_papers":
                # Parse tool response for sources
                # This would need to parse the actual document metadata
                # For now, return empty list
                pass

    return sources


def extract_tool_artefacts(messages: List) -> List[ToolArtefact]:
    """Extract tool artifacts from messages.

    :param messages: List of messages from graph state
    :returns: List of ToolArtefact objects
    """
    artefacts = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            artefact = ToolArtefact(
                tool_name=getattr(msg, "name", "unknown"),
                tool_call_id=getattr(msg, "tool_call_id", ""),
                content=msg.content,
                metadata={},
            )
            artefacts.append(artefact)

    return artefacts


def create_reasoning_step(
    step_name: str,
    description: str,
    metadata: Optional[Dict] = None,
) -> ReasoningStep:
    """Create a reasoning step record.

    :param step_name: Name of the step/node
    :param description: Human-readable description
    :param metadata: Additional metadata
    :returns: ReasoningStep object
    """
    return ReasoningStep(
        step_name=step_name,
        description=description,
        metadata=metadata or {},
    )


def filter_messages(messages: List) -> List[AIMessage | HumanMessage]:
    """Filter messages to include only HumanMessage and AIMessage types.

    Excludes tool messages and other internal message types.

    :param messages: List of messages to filter
    :returns: Filtered list of messages
    """
    return [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]


def get_latest_query(messages: List) -> str:
    """Get the latest user query from messages.

    :param messages: List of messages
    :returns: Latest query text
    :raises ValueError: If no user query found
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content

    raise ValueError("No user query found in messages")


def get_original_query(state: Dict) -> str:
    """Return the user's original question (before any rewrites).

    Rewritten queries are optimised for retrieval, not for answering: grading and answer
    generation must target what the user actually asked.
    """
    return state.get("original_query") or get_latest_query(state["messages"])


def get_latest_tool_documents(messages: List) -> List:
    """Return the documents (tool artefact) attached to the latest retrieval ToolMessage."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            artifact = getattr(msg, "artifact", None)
            return list(artifact) if artifact else []
    return []


def documents_to_sources(documents: List) -> List[SourceItem]:
    """Convert retrieved LangChain documents into de-duplicated SourceItem records."""
    sources: Dict[str, SourceItem] = {}
    for doc in documents:
        meta = getattr(doc, "metadata", {}) or {}
        arxiv_id = meta.get("arxiv_id")
        if not arxiv_id or arxiv_id in sources:
            continue
        authors = meta.get("authors") or []
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(",") if a.strip()]
        sources[arxiv_id] = SourceItem(
            arxiv_id=arxiv_id,
            title=meta.get("title", ""),
            authors=authors,
            url=meta.get("source", ""),
            relevance_score=float(meta.get("score", 0.0) or 0.0),
        )
    return list(sources.values())


def get_latest_context(messages: List) -> str:
    """Get the latest context from tool messages.

    :param messages: List of messages
    :returns: Latest context text or empty string
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return msg.content if hasattr(msg, "content") else ""

    return ""
