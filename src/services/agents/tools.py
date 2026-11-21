import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient
from src.services.retrieval import Retriever
from src.utils.arxiv import arxiv_pdf_url
from src.utils.prompting import escape_document_text

logger = logging.getLogger(__name__)


def format_documents_for_llm(documents: List[Document]) -> str:
    """Render retrieved chunks as a compact, citation-friendly context block.

    Previously the ToolMessage content was ``str(list[Document])`` - a Python repr full of
    metadata noise (scores, top_k, search mode) that wasted context and confused small models.
    Each excerpt is wrapped in explicit delimiters so the model can tell retrieved (untrusted)
    text apart from instructions.
    """
    if not documents:
        return ""

    blocks = []
    for i, doc in enumerate(documents, 1):
        meta = doc.metadata
        header = f"[{i}] arXiv:{meta.get('arxiv_id', '')}"
        if meta.get("title"):
            header += f" | {meta['title']}"
        if meta.get("section"):
            header += f" | section: {meta['section']}"
        blocks.append(f"<document>\n{header}\n{escape_document_text(doc.page_content)}\n</document>")
    return "\n\n".join(blocks)


def create_retriever_tool(
    opensearch_client: OpenSearchClient,
    embeddings_client: JinaEmbeddingsClient,
    top_k: int = 3,
    use_hybrid: bool = True,
    reranker=None,
    rerank_candidates: int = 20,
):
    """Create a retriever tool that wraps OpenSearch service.

    The tool returns ``(content, artifact)``: ``content`` is the formatted text the LLM sees,
    ``artifact`` is the list of ``Document`` objects, used downstream to return real sources.

    :param opensearch_client: Existing OpenSearch service
    :param embeddings_client: Existing Jina embeddings service
    :param top_k: Default number of chunks to retrieve
    :param use_hybrid: Default search mode (hybrid BM25 + vector)
    :param reranker: Optional cross-encoder reranker (see src/services/retrieval)
    :param rerank_candidates: Chunks fetched before reranking down to top_k
    :returns: LangChain tool for retrieving papers
    """

    # Per-call arguments below shadow these names, so keep the factory defaults separately.
    default_top_k = top_k
    default_use_hybrid = use_hybrid
    retriever = Retriever(opensearch_client, embeddings_client, reranker=reranker, rerank_candidates=rerank_candidates)

    @tool(response_format="content_and_artifact")
    async def retrieve_papers(
        query: str,
        top_k: Optional[int] = None,
        use_hybrid: Optional[bool] = None,
        categories: Optional[List[str]] = None,
    ) -> Tuple[str, List[Document]]:
        """Search and return relevant arXiv research papers.

        Use this tool when the user asks about:
        - Machine learning concepts or techniques
        - Deep learning architectures
        - Natural language processing
        - Computer vision methods
        - AI research topics
        - Specific algorithms or models

        :param query: The search query describing what papers to find
        :param top_k: Number of chunks to return (defaults to the service setting)
        :param use_hybrid: Use hybrid BM25 + vector search (defaults to the service setting)
        :param categories: Optional arXiv category filter, e.g. ["cs.AI"]
        :returns: Formatted paper excerpts (content) and the underlying documents (artifact)
        """
        k = top_k or default_top_k
        hybrid = default_use_hybrid if use_hybrid is None else use_hybrid

        logger.info(f"Retrieving papers for query: {query[:100]}...")
        logger.debug(f"Search mode: {'hybrid' if hybrid else 'bm25'}, top_k: {k}")

        # Shared retrieval path: embedding with BM25 fallback (the agent used to fail outright when
        # the embedding API was down), OpenSearch in a worker thread, optional reranking.
        result = await retriever.retrieve(query, top_k=k, use_hybrid=hybrid, categories=categories)

        documents = []
        hits = result.hits
        logger.info(f"Found {len(hits)} documents from OpenSearch")

        for hit in hits:
            text = hit.get("chunk_text") or hit.get("abstract") or ""
            arxiv_id = hit.get("arxiv_id", "")
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "arxiv_id": arxiv_id,
                        "title": hit.get("title", ""),
                        "authors": hit.get("authors", ""),
                        "score": hit.get("score", 0.0),
                        "source": arxiv_pdf_url(arxiv_id) if arxiv_id else "",
                        "section": hit.get("section_title") or hit.get("section_name", ""),
                        "search_mode": result.search_mode,
                        "reranked": result.reranked,
                        "top_k": k,
                    },
                )
            )

        logger.info(f"✓ Retrieved {len(documents)} papers successfully")
        return format_documents_for_llm(documents), documents

    return retrieve_papers
