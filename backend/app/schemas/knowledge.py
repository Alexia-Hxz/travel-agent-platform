from pydantic import BaseModel


class KnowledgeStatus(BaseModel):
    """What of this trip's uploaded material is actually searchable."""

    available: bool
    indexed_chunks: int
    indexed_files: int
    backend: str
    collection: str
    path: str
    embedding_model: str
    embedding_dim: int
    error: str | None


class KnowledgeChunk(BaseModel):
    chunk_id: str
    file_id: int
    source_label: str
    chunk_index: int
    text: str
    similarity: float


class KnowledgeSearchResult(BaseModel):
    # "ok" reports real matches. The other values say why there are none, so an empty
    # list is never mistaken for "this traveller has no such document".
    status: str
    reason: str | None
    chunks: list[KnowledgeChunk]


class KnowledgeRebuildResult(BaseModel):
    indexed_files: int
    skipped_files: int
    failed_files: int
    indexed_chunks: int
