"""Documents routes — placeholder for RAG document ingestion.

Full implementation in Phase 5 (RAG engine). This stub ensures the
app starts correctly and routes are registered.
"""

from fastapi import APIRouter, HTTPException, status

from haiip.api.deps import CurrentUser, EngineerUser
from haiip.api.schemas import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    body: QueryRequest,
    current_user: CurrentUser,
) -> QueryResponse:
    """Query the RAG engine with a natural-language question.

    Full implementation available in Phase 5.
    """
    # Phase 5 placeholder — returns a structured stub
    return QueryResponse(
        answer=(
            f"RAG engine not yet initialised. "
            f"Your question: '{body.question}' has been received. "
            f"Full implementation arrives in Phase 5."
        ),
        sources=[],
        confidence=0.0,
        machine_context={"machine_id": body.machine_id} if body.machine_id else None,
    )


@router.post("/documents/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(
    current_user: EngineerUser,
    title: str,
    content: str,
) -> dict:
    """Ingest a document into the RAG vector store. Phase 5 implementation."""
    return {
        "status": "accepted",
        "message": "Document ingestion will be fully implemented in Phase 5.",
        "title": title,
    }
