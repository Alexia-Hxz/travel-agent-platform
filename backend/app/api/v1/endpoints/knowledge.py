"""Endpoints for searching the traveller's own uploaded documents.

Kept out of the files router on purpose: those routes live under
``/trips/{trip_id}/files/{file_id}`` and a sibling path such as ``/files/search`` would be
matched as a file id first.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.session import get_session
from app.schemas.knowledge import (
    KnowledgeChunk,
    KnowledgeRebuildResult,
    KnowledgeSearchResult,
    KnowledgeStatus,
)
from app.services import vector_service
from app.services.file_service import list_trip_files
from app.services.trip_service import get_trip


router = APIRouter(prefix="/trips/{trip_id}/knowledge", tags=["knowledge"])
SessionDependency = Annotated[Session, Depends(get_session)]


@router.get("/status", response_model=KnowledgeStatus)
def knowledge_status_endpoint(trip_id: int) -> KnowledgeStatus:
    return KnowledgeStatus(**vector_service.knowledge_status(trip_id))


@router.get("/search", response_model=KnowledgeSearchResult)
def knowledge_search_endpoint(
    trip_id: int,
    q: Annotated[str, Query(min_length=1, max_length=500)],
    limit: Annotated[int, Query(ge=1, le=20)] = vector_service.DEFAULT_RESULT_LIMIT,
) -> KnowledgeSearchResult:
    outcome = vector_service.search_trip_documents(trip_id, q, limit)
    return KnowledgeSearchResult(
        status=outcome.status,
        reason=outcome.reason,
        chunks=[
            KnowledgeChunk(
                chunk_id=chunk.chunk_id,
                file_id=chunk.file_id,
                source_label=chunk.source_label,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                similarity=chunk.similarity,
            )
            for chunk in outcome.chunks
        ],
    )


@router.post("/rebuild", response_model=KnowledgeRebuildResult)
def knowledge_rebuild_endpoint(trip_id: int, session: SessionDependency) -> KnowledgeRebuildResult:
    """Re-index every parsed file of this trip.

    Exists because the index can legitimately be behind: a document parsed before this
    feature existed, an embedding provider that was down at upload time, or a model
    change. Without it the only way to repair a stale index would be to delete and
    re-upload the documents.
    """
    indexed = skipped = failed = chunks = 0
    for trip_file in list_trip_files(session, trip_id):
        if trip_file.parse_status != "completed":
            continue
        outcome = vector_service.index_trip_file(session, trip_file)
        if outcome.status == "indexed":
            indexed += 1
            chunks += outcome.chunk_count
        elif outcome.status == "skipped":
            skipped += 1
        else:
            failed += 1
    return KnowledgeRebuildResult(
        indexed_files=indexed, skipped_files=skipped, failed_files=failed, indexed_chunks=chunks
    )
