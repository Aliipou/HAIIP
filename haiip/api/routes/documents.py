"""Documents routes — RAG document ingestion and natural-language querying."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, status

from haiip.api.config import get_settings
from haiip.api.deps import CurrentUser, EngineerUser
from haiip.api.schemas import QueryRequest, QueryResponse
from haiip.core.rag import Document, RAGEngine

router = APIRouter()
logger = structlog.get_logger(__name__)
settings = get_settings()

# Module-level RAGEngine instance — shared across requests
_rag_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine(
            persist_dir=settings.model_artifacts_path + "/rag",
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
        )
        _rag_engine.initialize()
    return _rag_engine


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    body: QueryRequest,
    current_user: CurrentUser,
) -> QueryResponse:
    """Query the knowledge base with a natural-language question.

    Returns AI-generated answer (if OpenAI configured) or context-based
    template answer, plus source citations and confidence score.
    """
    engine = get_rag_engine()

    result = engine.query(
        question=body.question,
        machine_id=body.machine_id,
        top_k=body.context_window,
    )

    logger.info(
        "rag.query",
        question=body.question[:80],
        sources=len(result.sources),
        confidence=result.confidence,
        llm_used=result.llm_used,
        tenant_id=current_user.tenant_id,
    )

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        confidence=result.confidence,
        machine_context=result.machine_context,
    )


@router.post("/documents/ingest", status_code=status.HTTP_201_CREATED)
async def ingest_document(
    body: dict,
    current_user: EngineerUser,
) -> dict:
    """Ingest a document into the RAG vector store.

    Engineers and admins only. Documents are chunked automatically.
    Supports: maintenance manuals, fault reports, ISO standards, SOPs.
    """
    title = body.get("title", "")
    content = body.get("content", "")
    source = body.get("source", "manual")
    machine_id = body.get("machine_id")

    if not title or not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Both 'title' and 'content' are required",
        )
    if len(content) < 10:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Content too short (minimum 10 characters)",
        )
    if len(content) > 100_000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Content too large (maximum 100,000 characters)",
        )

    engine = get_rag_engine()
    n_chunks = engine.add_text(
        content=content,
        title=title,
        source=source,
        machine_id=machine_id,
    )

    logger.info(
        "rag.ingest",
        title=title,
        chunks=n_chunks,
        machine_id=machine_id,
        by=current_user.id,
        tenant_id=current_user.tenant_id,
    )

    return {
        "status": "ingested",
        "title": title,
        "chunks_created": n_chunks,
        "total_documents": engine.document_count,
    }


@router.get("/documents/stats")
async def document_stats(current_user: CurrentUser) -> dict:
    """Return knowledge base statistics."""
    engine = get_rag_engine()
    return {
        "total_documents": engine.document_count,
        "index_ready": engine.is_initialized,
        "llm_enabled": engine._llm is not None,
        "embedding_model": engine.model_name,
    }


@router.delete("/documents", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def clear_documents(current_user: EngineerUser):
    """Clear all documents from the knowledge base. Engineers+ only."""
    global _rag_engine
    _rag_engine = None  # Reset — will reinitialize on next request
    logger.warning("rag.cleared", by=current_user.id, tenant_id=current_user.tenant_id)
